import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


_LORA_A_MARKER = ".lora_A."
_LORA_B_MARKER = ".lora_B."


def _normalize_optional_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if text in {"", "None", "none", "null"}:
        return None
    return text


def _sequence_get(values: Any, index: int) -> Any:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.ndim == 0 or index >= int(values.shape[0]):
            return None
        return values[index].item()
    if hasattr(values, "tolist") and not isinstance(values, (list, tuple)):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        if index >= len(values):
            return None
        return values[index]
    if index == 0:
        return values
    return None


def extract_ephemeral_lora_paths_from_micro_batch(
    micro_batch: dict[str, Any],
    *,
    batch_size: int,
    path_key: str = "ephemeral_lora_path",
) -> list[str | None]:
    candidate_keys = [path_key, "ephemeral_lora_loss_path", "ephemeral_lora_path"]
    candidate_keys = list(dict.fromkeys(candidate_keys))
    extra_infos = micro_batch.get("extra_info")
    paths: list[str | None] = []
    for row_index in range(batch_size):
        resolved_path = None
        for candidate_key in candidate_keys:
            resolved_path = _normalize_optional_path(_sequence_get(micro_batch.get(candidate_key), row_index))
            if resolved_path is not None:
                break
        if resolved_path is None:
            extra_info = _sequence_get(extra_infos, row_index)
            if isinstance(extra_info, dict):
                for candidate_key in candidate_keys:
                    resolved_path = _normalize_optional_path(extra_info.get(candidate_key))
                    if resolved_path is not None:
                        break
        paths.append(resolved_path)
    return paths


def _strip_peft_prefix(module_name: str) -> str:
    prefixes = (
        "base_model.model.",
        "base_model.",
    )
    for prefix in prefixes:
        if module_name.startswith(prefix):
            return module_name[len(prefix) :]
    return module_name


def _adapter_file(lora_path: str) -> str:
    if os.path.isdir(lora_path):
        return os.path.join(lora_path, "adapter_model.safetensors")
    return lora_path


def _adapter_config_file(lora_path: str) -> str:
    if os.path.isdir(lora_path):
        return os.path.join(lora_path, "adapter_config.json")
    return os.path.join(os.path.dirname(lora_path), "adapter_config.json")


def _load_ephemeral_lora_adapter_uncached(lora_path: str) -> dict[str, Any]:
    from safetensors.torch import safe_open

    adapter_path = _adapter_file(lora_path)
    config_path = _adapter_config_file(lora_path)
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"ephemeral LoRA adapter weights not found: {adapter_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ephemeral LoRA adapter config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    modules: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if _LORA_A_MARKER in key:
                module_name = _strip_peft_prefix(key.split(_LORA_A_MARKER, 1)[0])
                modules.setdefault(module_name, {})["A"] = f.get_tensor(key).contiguous()
            elif _LORA_B_MARKER in key:
                module_name = _strip_peft_prefix(key.split(_LORA_B_MARKER, 1)[0])
                modules.setdefault(module_name, {})["B"] = f.get_tensor(key).contiguous()

    complete_modules: dict[str, dict[str, Any]] = {}
    use_rslora = bool(config.get("use_rslora", False))
    default_alpha = float(config.get("lora_alpha", 1.0))
    alpha_pattern = config.get("alpha_pattern", {}) or {}
    for module_name, tensors in modules.items():
        if "A" not in tensors or "B" not in tensors:
            continue
        rank = max(1, int(tensors["A"].shape[0]))
        alpha = float(alpha_pattern.get(module_name, default_alpha))
        scale = alpha / (rank**0.5 if use_rslora else rank)
        complete_modules[module_name] = {"A": tensors["A"], "B": tensors["B"], "scale": float(scale)}

    if not complete_modules:
        raise ValueError(f"no complete LoRA A/B tensor pairs found in {adapter_path}")
    return {"path": os.path.abspath(os.fspath(lora_path)), "config": config, "modules": complete_modules}


def load_ephemeral_lora_adapter(
    lora_path: str,
    *,
    cache: OrderedDict[str, dict[str, Any]] | None = None,
    max_cache_entries: int = 1,
) -> dict[str, Any]:
    normalized_path = os.path.abspath(os.fspath(lora_path))
    if cache is None or max_cache_entries <= 0:
        return _load_ephemeral_lora_adapter_uncached(normalized_path)
    if normalized_path in cache:
        adapter = cache.pop(normalized_path)
        cache[normalized_path] = adapter
        return adapter
    adapter = _load_ephemeral_lora_adapter_uncached(normalized_path)
    cache[normalized_path] = adapter
    while len(cache) > max_cache_entries:
        cache.popitem(last=False)
    return adapter


def _layer_fsdp_wrapped_name(module_name: str) -> str | None:
    parts = module_name.split(".")
    wrapped: list[str] = []
    inserted = False
    index = 0
    while index < len(parts):
        wrapped.append(parts[index])
        if parts[index] == "layers" and index + 1 < len(parts):
            index += 1
            wrapped.append(parts[index])
            wrapped.append("_fsdp_wrapped_module")
            inserted = True
        index += 1
    if not inserted:
        return None
    return ".".join(wrapped)


def _module_lookup_candidates(module_name: str) -> list[str]:
    base_names = [module_name]
    layer_wrapped = _layer_fsdp_wrapped_name(module_name)
    if layer_wrapped is not None:
        base_names.append(layer_wrapped)
    if module_name.startswith("model."):
        stripped = module_name[len("model.") :]
        base_names.append(stripped)
        stripped_layer_wrapped = _layer_fsdp_wrapped_name(stripped)
        if stripped_layer_wrapped is not None:
            base_names.append(stripped_layer_wrapped)

    candidates: list[str] = []
    for base_name in base_names:
        candidates.extend([
            base_name,
            f"_fsdp_wrapped_module.{base_name}",
            f"{base_name}.base_layer",
            f"{base_name}._fsdp_wrapped_module",
            f"{base_name}.base_layer._fsdp_wrapped_module",
            f"_fsdp_wrapped_module.{base_name}.base_layer",
            f"_fsdp_wrapped_module.{base_name}._fsdp_wrapped_module",
            f"_fsdp_wrapped_module.{base_name}.base_layer._fsdp_wrapped_module",
        ])
    return list(dict.fromkeys(candidates))


def _find_module_by_suffix(model: nn.Module, module_name: str) -> nn.Module | None:
    named_modules = dict(model.named_modules())
    candidate_names = _module_lookup_candidates(module_name)
    for candidate_name in candidate_names:
        if candidate_name in named_modules:
            return named_modules[candidate_name]

    suffixes = [f".{candidate_name}" for candidate_name in candidate_names]
    suffixes = list(dict.fromkeys(suffixes))
    matches = [
        (name, module)
        for name, module in named_modules.items()
        if any(name.endswith(suffix) for suffix in suffixes)
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: len(item[0]))
    return matches[0][1]


def _format_module_lookup_diagnostics(model: nn.Module, missing_modules: list[str]) -> str:
    named_items = list(model.named_modules())
    first_missing = missing_modules[0] if missing_modules else ""
    fragments = [part for part in first_missing.split(".") if part]
    interesting_terms = set(fragments[-4:]) | {"down_proj", "up_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj", "base_layer", "fsdp"}
    matching = []
    for name, module in named_items:
        lowered = name.lower()
        module_type = type(module).__name__
        if any(term.lower() in lowered or term.lower() in module_type.lower() for term in interesting_terms):
            matching.append(f"{name}:{module_type}")
    if not matching:
        matching = [f"{name}:{type(module).__name__}" for name, module in named_items[:80]]
    root_children = [f"{name}:{type(module).__name__}" for name, module in list(model.named_children())[:40]]
    return (
        f"available_module_count={len(named_items)}; "
        f"model_type={type(model).__name__}; "
        f"root_children={root_children}; "
        f"interesting_module_samples={matching[:120]}"
    )


@contextmanager
def apply_ephemeral_lora_to_model(
    model: nn.Module,
    lora_path: str | None,
    *,
    cache: OrderedDict[str, dict[str, Any]] | None = None,
    max_cache_entries: int = 1,
):
    normalized_path = _normalize_optional_path(lora_path)
    if normalized_path is None:
        yield
        return

    adapter = load_ephemeral_lora_adapter(
        normalized_path,
        cache=cache,
        max_cache_entries=max_cache_entries,
    )
    handles = []
    missing_modules: list[str] = []
    for module_name, lora_state in adapter["modules"].items():
        module = _find_module_by_suffix(model, module_name)
        if module is None:
            missing_modules.append(module_name)
            continue

        lora_a_cpu = lora_state["A"]
        lora_b_cpu = lora_state["B"]
        scale = float(lora_state["scale"])

        def hook(current_module, inputs, output, *, lora_a=lora_a_cpu, lora_b=lora_b_cpu, lora_scale=scale):
            if not inputs:
                return output
            hidden_states = inputs[0]
            if not torch.is_tensor(hidden_states):
                return output
            # Keep the hook stateless so activation checkpoint recomputation sees the same graph.
            lora_a_device = lora_a.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True).detach()
            lora_b_device = lora_b.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True).detach()
            delta = F.linear(F.linear(hidden_states, lora_a_device), lora_b_device) * lora_scale
            return output + delta

        handles.append(module.register_forward_hook(hook))

    if missing_modules:
        diagnostics = _format_module_lookup_diagnostics(model, missing_modules)
        raise ValueError(
            f"failed to apply ephemeral LoRA {normalized_path}: {len(missing_modules)} target modules were not found; "
            f"first_missing={missing_modules[0]!r}; {diagnostics}"
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def slice_micro_batch_rows(micro_batch: dict[str, Any], indices: list[int], *, batch_size: int) -> dict[str, Any]:
    index_tensor_cache: dict[torch.device, torch.Tensor] = {}
    sliced: dict[str, Any] = {}
    for key, value in micro_batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            index_tensor = index_tensor_cache.get(value.device)
            if index_tensor is None:
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=value.device)
                index_tensor_cache[value.device] = index_tensor
            sliced[key] = value.index_select(0, index_tensor)
        elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            as_list = value.tolist()
            sliced[key] = [as_list[index] for index in indices] if len(as_list) == batch_size else value
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            sliced[key] = [value[index] for index in indices]
        else:
            sliced[key] = value
    return sliced
