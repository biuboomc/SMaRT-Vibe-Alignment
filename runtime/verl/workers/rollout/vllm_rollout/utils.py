# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
import json
import logging
import os
import platform
import signal
import threading
from types import MethodType
from typing import Any, Literal, get_args

import torch
from safetensors.torch import load_file

from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

COMMON_WEIGHT_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "module.",
    "actor_module.",
    "base_model.model.",
    "base_model.",
    "language_model.",
    "model.",
)

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        return current_platform.get_device_uuid(device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


def _sample_weight_names(weight_names: list[str], limit: int = 6) -> str:
    if not weight_names:
        return "[]"
    sample = weight_names[:limit]
    suffix = "..." if len(weight_names) > limit else ""
    return f"{sample}{suffix}"


def _summarize_weight_roots(weight_names: list[str], limit: int = 6) -> str:
    root_counts: dict[str, int] = {}
    for name in weight_names:
        root = name.split(".", 1)[0]
        root_counts[root] = root_counts.get(root, 0) + 1
    ranked_roots = sorted(root_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{root}:{count}" for root, count in ranked_roots)


def _collect_model_weight_names(model: Any) -> list[str]:
    names: list[str] = []
    for accessor_name in ("named_parameters", "named_buffers"):
        accessor = getattr(model, accessor_name, None)
        if accessor is None:
            continue
        try:
            names.extend(name for name, _ in accessor())
        except TypeError:
            continue

    if not names:
        state_dict = getattr(model, "state_dict", None)
        if state_dict is not None:
            try:
                names.extend(list(state_dict().keys()))
            except TypeError:
                pass

    return list(dict.fromkeys(names))


def _collect_base_layer_wrapped_paths(expected_names: list[str]) -> set[str]:
    wrapped_paths: set[str] = set()
    marker = ".base_layer."
    for name in expected_names:
        if marker not in name:
            continue
        wrapped_paths.add(name.split(marker, 1)[0])
    return wrapped_paths


def _iter_weight_name_candidates(name: str):
    seen = {name}
    queue = [name]

    while queue:
        current = queue.pop(0)
        yield current

        for prefix in COMMON_WEIGHT_WRAPPER_PREFIXES:
            if current.startswith(prefix):
                candidate = current[len(prefix) :]
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    queue.append(candidate)


def _resolve_weight_name_against_expected(name: str, expected_names: set[str]) -> str | None:
    if name in expected_names:
        return name

    for candidate in _iter_weight_name_candidates(name):
        if candidate in expected_names:
            return candidate

    exact_suffix_matches = [expected for expected in expected_names if name.endswith(f".{expected}")]
    if len(exact_suffix_matches) == 1:
        return exact_suffix_matches[0]

    reverse_suffix_matches = [expected for expected in expected_names if expected.endswith(f".{name}")]
    if len(reverse_suffix_matches) == 1:
        return reverse_suffix_matches[0]

    parts = name.split(".")
    for index in range(1, len(parts) - 1):
        suffix = ".".join(parts[index:])
        if suffix in expected_names:
            return suffix

    return None


def _maybe_insert_base_layer(name: str, wrapped_paths: set[str]) -> str | None:
    if not wrapped_paths or ".base_layer." in name:
        return None

    parts = name.split(".")
    if len(parts) < 2:
        return None

    module_path = ".".join(parts[:-1])
    param_name = parts[-1]

    candidate_wrapped_paths = {module_path}
    if parts[-2] in {"q_proj", "k_proj", "v_proj"}:
        candidate_wrapped_paths.add(".".join([*parts[:-2], "qkv_proj"]))
    if parts[-2] in {"gate_proj", "up_proj"}:
        candidate_wrapped_paths.add(".".join([*parts[:-2], "gate_up_proj"]))

    if not any(path in wrapped_paths for path in candidate_wrapped_paths):
        return None

    return f"{module_path}.base_layer.{param_name}"


def _resolve_weight_name_with_model_wrappers(
    name: str,
    expected_names: set[str],
    wrapped_paths: set[str],
) -> str | None:
    fallback_name: str | None = None
    for candidate in _iter_weight_name_candidates(name):
        resolved_name = _resolve_weight_name_against_expected(candidate, expected_names)
        if resolved_name is not None:
            return resolved_name

        base_layer_candidate = _maybe_insert_base_layer(candidate, wrapped_paths)
        if base_layer_candidate is None:
            continue

        resolved_name = _resolve_weight_name_against_expected(base_layer_candidate, expected_names)
        if resolved_name is not None:
            return resolved_name

        if fallback_name is None:
            fallback_name = base_layer_candidate

    return fallback_name


def _expand_nested_loader_prefix(
    weights: list[tuple[str, torch.Tensor]],
    model: Any | None,
) -> list[tuple[str, torch.Tensor]] | None:
    if model is None or not weights:
        return None

    child_modules = dict(model.named_children()) if hasattr(model, "named_children") else {}
    root_counts: dict[str, int] = {}
    for name, _ in weights:
        root = name.split(".", 1)[0]
        root_counts[root] = root_counts.get(root, 0) + 1

    if not root_counts:
        return None

    dominant_root = max(root_counts, key=root_counts.get)
    child_module = child_modules.get(dominant_root)
    if child_module is None or not callable(getattr(child_module, "load_weights", None)):
        return None

    base_prefix = f"{dominant_root}."
    nested_prefix = f"{dominant_root}.{dominant_root}."
    remapped_weights: list[tuple[str, torch.Tensor]] = []
    changed = False

    for name, tensor in weights:
        if name.startswith(base_prefix) and not name.startswith(nested_prefix):
            remapped_weights.append((f"{dominant_root}.{name}", tensor))
            changed = True
        else:
            remapped_weights.append((name, tensor))

    return remapped_weights if changed else None


def remap_weight_prefixes_for_vllm_load(
    weights: list[tuple[str, torch.Tensor]],
    missing_key: str | None,
    model: Any | None = None,
) -> list[tuple[str, torch.Tensor]] | None:
    """Retry helper for vLLM weight loading when model wrappers disagree on key prefixes.

    Some vLLM LoRA-enabled model wrappers look up parameters without the leading
    ``model.`` prefix even though the actor-side state dict still uses the
    HuggingFace-style ``model.*`` names. When that happens, ``load_weights``
    raises ``KeyError`` for the first missing key. We use that signal to do one
    conservative retry with a consistent prefix remap for the current bucket.
    """

    if not weights or not missing_key:
        return None

    expected_weight_names = _collect_model_weight_names(model) if model is not None else []
    if expected_weight_names:
        expected_name_set = set(expected_weight_names)
        wrapped_paths = _collect_base_layer_wrapped_paths(expected_weight_names)
        remapped_weights: list[tuple[str, torch.Tensor]] = []
        changed = False
        unresolved_names: list[str] = []
        for name, tensor in weights:
            resolved_name = _resolve_weight_name_with_model_wrappers(name, expected_name_set, wrapped_paths)
            if resolved_name is None:
                unresolved_names.append(name)
                remapped_weights.append((name, tensor))
                continue
            changed = changed or resolved_name != name
            remapped_weights.append((resolved_name, tensor))

        if changed:
            if unresolved_names:
                logger.warning(
                    "Remapped vLLM weight names using expected model keys; some names stayed unresolved. "
                    "missing_key=%s incoming_roots=%s unresolved_sample=%s expected_sample=%s",
                    missing_key,
                    _summarize_weight_roots([name for name, _ in weights]),
                    _sample_weight_names(unresolved_names),
                    _sample_weight_names(expected_weight_names),
                )
            return remapped_weights

    expanded_weights = _expand_nested_loader_prefix(weights, model)
    if expanded_weights is not None:
        return expanded_weights

    known_prefixes = ("model.", "language_model.")
    desired_prefix = next((prefix for prefix in known_prefixes if missing_key.startswith(prefix)), "")

    prefix_counts = {prefix: sum(name.startswith(prefix) for name, _ in weights) for prefix in known_prefixes}
    current_prefix = max(prefix_counts, key=prefix_counts.get)
    if prefix_counts[current_prefix] == 0:
        current_prefix = ""

    if current_prefix == desired_prefix:
        return None

    remapped_weights: list[tuple[str, torch.Tensor]] = []
    changed = False
    for name, tensor in weights:
        new_name = name
        if current_prefix:
            if name.startswith(current_prefix):
                new_name = f"{desired_prefix}{name[len(current_prefix):]}"
        elif desired_prefix and not any(name.startswith(prefix) for prefix in known_prefixes):
            new_name = f"{desired_prefix}{name}"

        changed = changed or new_name != name
        remapped_weights.append((new_name, tensor))

    return remapped_weights if changed else None


def _normalize_lora_tensor_payload(
    lora_tensors: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    if not lora_tensors:
        return {}

    normalized_tensors: dict[str, torch.Tensor] = {}
    converted_names: list[str] = []
    for name, value in lora_tensors.items():
        if isinstance(value, torch.Tensor):
            normalized_tensors[name] = value
            continue

        try:
            tensor = torch.as_tensor(value)
        except Exception as exc:  # pragma: no cover - defensive branch for malformed payloads
            raise TypeError(f"Unsupported LoRA tensor payload for `{name}`: {type(value)!r}") from exc

        if not torch.is_floating_point(tensor):
            tensor = tensor.float()

        normalized_tensors[name] = tensor.contiguous()
        converted_names.append(name)

    if converted_names:
        logger.warning(
            "Normalized async ephemeral LoRA payload values to torch.Tensor for %s entries. sample=%s",
            len(converted_names),
            _sample_weight_names(converted_names),
        )

    return normalized_tensors


def _read_ephemeral_lora_artifacts(lora_path: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")

    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"Missing adapter config: {adapter_config_path}")
    if not os.path.exists(adapter_model_path):
        raise FileNotFoundError(f"Missing adapter weights: {adapter_model_path}")

    with open(adapter_config_path, encoding="utf-8") as f:
        peft_config = json.load(f)
    lora_tensors = load_file(adapter_model_path, device="cpu")
    return peft_config, lora_tensors


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMHijack.hijack()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        return instance

    def monkey_patch_model(self, vocab_size: int):
        # patch compute_logits to avoid sampling OOV token
        monkey_patch_compute_logits(self.model_runner.model, vocab_size)
        # patch weight loader to support MoE model
        patch_vllm_moe_model_weight_loader(self.model_runner.model)

    def update_weights_from_ipc(
        self,
        peft_config: dict = None,
        base_sync_done=False,
        use_shm: bool = False,
        clear_actor_lora: bool = False,
    ):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old actor LoRA is removed before either re-adding it
        # or switching back to dense weight sync.
        if clear_actor_lora or (peft_config and base_sync_done):
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        if self._is_qat_model:
            # QAT: Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            prepare_qat_for_load_weights(self.model_runner.model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            patch_vllm_moe_model_weight_loader(self.model_runner.model)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

        if self._is_qat_model:
            # QAT: call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            manual_process_weights_after_loading(self.model_runner.model)
            logger.info("QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            model = self.model_runner.model
            model_config = self.model_runner.vllm_config.model_config
            process_weights_after_loading(model, model_config, self.device)

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                weight_bucket = list(weights)
                try:
                    self.model_runner.model.load_weights(weight_bucket)
                except KeyError as exc:
                    missing_key = exc.args[0] if exc.args else None
                    remapped_weights = remap_weight_prefixes_for_vllm_load(
                        weight_bucket,
                        missing_key,
                        model=self.model_runner.model,
                    )
                    if remapped_weights is None:
                        raise
                    logger.warning(
                        "Retrying standard vLLM weight load after key remap. missing_key=%s incoming_sample=%s "
                        "incoming_roots=%s remapped_sample=%s",
                        missing_key,
                        _sample_weight_names([name for name, _ in weight_bucket]),
                        _summarize_weight_roots([name for name, _ in weight_bucket]),
                        _sample_weight_names([name for name, _ in remapped_weights]),
                    )
                    self.model_runner.model.load_weights(remapped_weights)

    def load_ephemeral_lora_direct(
        self,
        peft_config: dict,
        lora_tensors: dict[str, torch.Tensor],
        *,
        lora_name: str = VLLM_LORA_NAME,
        lora_int_id: int = VLLM_LORA_INT_ID,
        lora_request_path: str = VLLM_LORA_PATH,
    ):
        """Load a temporary LoRA adapter directly into the async vLLM engine."""
        self.remove_lora(lora_int_id)
        normalized_tensors = _normalize_lora_tensor_payload(lora_tensors)
        lora_request = TensorLoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_request_path,
            peft_config=peft_config,
            lora_tensors=normalized_tensors,
        )
        self.add_lora(lora_request)
        logger.info(
            "vLLM load_ephemeral_lora_direct, loaded_params: %s, lora_name=%s, lora_int_id=%s",
            len(normalized_tensors),
            lora_name,
            lora_int_id,
        )

    def load_ephemeral_lora_from_path_direct(
        self,
        lora_path: str,
        *,
        lora_name: str = VLLM_LORA_NAME,
        lora_int_id: int = VLLM_LORA_INT_ID,
        lora_request_path: str = VLLM_LORA_PATH,
    ):
        """Load a temporary LoRA adapter from shared storage into the async vLLM engine."""
        peft_config, lora_tensors = _read_ephemeral_lora_artifacts(lora_path)
        self.load_ephemeral_lora_direct(
            peft_config=peft_config,
            lora_tensors=lora_tensors,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_request_path=lora_request_path,
        )
        logger.info(
            "vLLM load_ephemeral_lora_from_path_direct, lora_path=%s, lora_name=%s, lora_int_id=%s",
            lora_path,
            lora_name,
            lora_int_id,
        )

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication."""
        if not hasattr(self, "device_uuid") or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        return f"ipc:///tmp/rl-colocate-zmq-{self.device_uuid}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


