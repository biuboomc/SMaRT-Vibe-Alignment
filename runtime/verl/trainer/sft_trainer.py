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


import os
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from tensordict.tensorclass import NonTensorData

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker
from scripts.build_ephemeral_lora_adapters import (
    KNOWLEDGE_LORA_VARIANT,
    adapter_is_complete,
    resolve_dtype,
    resolve_lora_build_spec,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        log_gpu_memory_usage(f"rank {torch.distributed.get_rank()}: Before SFTTrainer init", logger=logger)

        self.rank = torch.distributed.get_rank()

        self._build_config()
        self._build_runtime_lora_manager_config()
        self._build_dataset()

        self._build_engine()

        self._build_dataloader()

        self._init_engine()

        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        if self.rank == 0:
            print(self.config)

        log_gpu_memory_usage(f"rank {self.rank}: After SFTTrainer init", logger=logger)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)
        lora_train_meta = self._get_lora_train_meta()

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
            lora_train_meta=lora_train_meta,
        )

    def _get_lora_train_meta(self):
        lora_adapter_path = self.config.model.get("lora_adapter_path", None)
        lora_rank = int(getattr(self.config.model, "lora_rank", 0) or 0)

        if lora_adapter_path is None and lora_rank <= 0:
            return None

        raw_lora_alpha = self.config.model.get("lora_alpha", None)
        if raw_lora_alpha is None:
            log_with_rank(
                "LoRA is enabled but `model.lora_alpha` is not set; fallback to 0 in checkpoint metadata.",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            lora_alpha = 0
        else:
            lora_alpha = int(raw_lora_alpha)
            if lora_alpha == 0:
                log_with_rank(
                    "LoRA is enabled but `model.lora_alpha` is 0; this may lead to ineffective LoRA scaling.",
                    logger=logger,
                    rank=self.rank,
                    level=logging.WARNING,
                    log_only_rank_0=True,
                )

        task_type = self.config.model.get("task_type", None)
        if task_type is None:
            task_type = "CAUSAL_LM"

        return {
            "r": lora_rank,
            "lora_alpha": int(lora_alpha or 0),
            "task_type": str(task_type),
        }

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

        # check profile interval
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_candidate_list(raw_value, *, cast_fn, default):
        if raw_value in (None, "", []):
            return list(default)
        if isinstance(raw_value, (list, tuple)):
            return [cast_fn(item) for item in raw_value]
        text = str(raw_value).strip()
        if text.startswith("[") and text.endswith("]"):
            return [cast_fn(item) for item in ast.literal_eval(text)]
        return [cast_fn(item.strip()) for item in text.split(",") if item.strip()]

    @staticmethod
    def _parse_module_list(raw_value) -> list[str]:
        if raw_value in (None, "", []):
            return []
        if isinstance(raw_value, (list, tuple)):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        text = str(raw_value).strip()
        if text.startswith("[") and text.endswith("]"):
            return [str(item).strip() for item in ast.literal_eval(text) if str(item).strip()]
        return [item.strip() for item in text.split(",") if item.strip()]

    def _build_runtime_lora_manager_config(self):
        data_config = self.config.data
        model_lora_config = self.config.model.get("lora", {}) or {}
        self.runtime_lora_enabled = self._coerce_bool(model_lora_config.get("runtime_swap", False)) and self._coerce_bool(
            data_config.get("knowledge_lora_on_demand_build", False)
        )
        self.runtime_lora_path_key = str(
            data_config.get("knowledge_lora_path_key", model_lora_config.get("runtime_path_key", "knowledge_lora_path"))
        )
        self.runtime_knowledge_id_key = str(data_config.get("knowledge_id_key", "knowledge_id"))
        self.runtime_lora_root_dir = data_config.get("knowledge_lora_root_dir", None)
        self.runtime_lora_delete_after_use = self._coerce_bool(data_config.get("knowledge_lora_delete_after_use", False))
        self.runtime_lora_variant = str(data_config.get("knowledge_lora_variant", KNOWLEDGE_LORA_VARIANT)).strip().lower()
        self.runtime_lora_model_path = str(data_config.get("knowledge_lora_model_path", "")).strip()
        self.runtime_lora_base_adapter_path = data_config.get("knowledge_lora_base_adapter_path", None)
        self.runtime_lora_steps = int(data_config.get("knowledge_lora_steps", 2))
        self.runtime_lora_learning_rate = float(data_config.get("knowledge_lora_learning_rate", 5e-4))
        self.runtime_lora_fallback_learning_rate = float(
            data_config.get("knowledge_lora_fallback_learning_rate", max(self.runtime_lora_learning_rate / 5.0, 1e-6))
        )
        self.runtime_lora_max_length = int(data_config.get("knowledge_lora_max_length", 768))
        self.runtime_lora_rank = int(data_config.get("knowledge_lora_rank", 8))
        self.runtime_lora_alpha = int(data_config.get("knowledge_lora_alpha", 16))
        self.runtime_lora_dropout = float(data_config.get("knowledge_lora_dropout", 0.0))
        self.runtime_lora_gradient_clip_norm = float(data_config.get("knowledge_lora_gradient_clip_norm", 1.0))
        self.runtime_lora_seed = int(data_config.get("knowledge_lora_seed", 42))
        self.runtime_lora_randomize_config = self._coerce_bool(data_config.get("knowledge_lora_randomize_config", False))
        self.runtime_lora_randomization_base_seed = data_config.get("knowledge_lora_randomization_base_seed", None)
        self.runtime_lora_lock_timeout_seconds = float(data_config.get("knowledge_lora_lock_timeout_seconds", 7200.0))
        self.runtime_lora_lock_poll_interval_seconds = float(
            data_config.get("knowledge_lora_lock_poll_interval_seconds", 1.0)
        )
        self.runtime_lora_preferred_dtype = resolve_dtype(
            str(data_config.get("knowledge_lora_dtype", "auto")).strip().lower()
        )
        fallback_dtype_name = str(data_config.get("knowledge_lora_fallback_dtype", "fp32")).strip().lower()
        self.runtime_lora_fallback_dtype = None
        if fallback_dtype_name not in ("", "none"):
            self.runtime_lora_fallback_dtype = resolve_dtype(fallback_dtype_name)
        self.runtime_lora_target_modules = self._parse_module_list(
            data_config.get(
                "knowledge_lora_target_modules",
                "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            )
        )
        self.runtime_lora_rank_candidates = self._parse_candidate_list(
            data_config.get("knowledge_lora_rank_candidates", None),
            cast_fn=int,
            default=[self.runtime_lora_rank],
        )
        self.runtime_lora_alpha_candidates = self._parse_candidate_list(
            data_config.get("knowledge_lora_alpha_candidates", None),
            cast_fn=int,
            default=[self.runtime_lora_alpha],
        )
        self.runtime_lora_dropout_candidates = self._parse_candidate_list(
            data_config.get("knowledge_lora_dropout_candidates", None),
            cast_fn=float,
            default=[self.runtime_lora_dropout],
        )
        self.runtime_lora_build_pool_size = int(data_config.get("knowledge_lora_build_pool_size", 1))
        raw_build_cuda_devices = data_config.get("knowledge_lora_build_cuda_devices", None)
        if raw_build_cuda_devices in (None, "", []):
            raw_device_tokens = []
        elif isinstance(raw_build_cuda_devices, (list, tuple)):
            raw_device_tokens = list(raw_build_cuda_devices)
        else:
            text = str(raw_build_cuda_devices).strip()
            if text.startswith("[") and text.endswith("]"):
                raw_device_tokens = ast.literal_eval(text)
            else:
                raw_device_tokens = [item.strip() for item in text.split(",") if item.strip()]
        self.runtime_lora_build_cuda_devices: list[str | None] = []
        for token in raw_device_tokens:
            if token in (None, "", "none", "cpu"):
                self.runtime_lora_build_cuda_devices.append(None)
            else:
                self.runtime_lora_build_cuda_devices.append(str(token))
        if not self.runtime_lora_build_cuda_devices:
            if torch.cuda.is_available():
                local_rank = os.environ.get("LOCAL_RANK")
                current_device = local_rank if local_rank not in (None, "") else str(torch.cuda.current_device())
                self.runtime_lora_build_cuda_devices = [str(current_device)]
            else:
                self.runtime_lora_build_cuda_devices = [None]

    @staticmethod
    def _stacked_non_tensor_to_list(value):
        if value is None:
            return []
        if hasattr(value, "tolist"):
            try:
                output = value.tolist()
                if isinstance(output, list):
                    return output
            except TypeError:
                pass
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        try:
            return list(value)
        except TypeError:
            return [value]

    @staticmethod
    def _pad_optional_values(values, size, fill_value=None):
        values = list(values or [])
        if len(values) >= size:
            return values[:size]
        return values + [fill_value] * (size - len(values))

    def _batch_size_from_raw_batch(self, batch) -> int:
        for value in batch.values():
            try:
                return len(value)
            except TypeError:
                continue
        return 0

    @staticmethod
    def _normalize_optional_path(value) -> str | None:
        if value in (None, "", "None"):
            return None
        return os.path.abspath(os.fspath(value))

    def _runtime_lora_path_has_required_files(self, path: str) -> bool:
        return adapter_is_complete(Path(path).expanduser())

    def _get_runtime_lora_build_pool_spec(self) -> tuple[int, list[str | None]]:
        device_tokens = list(self.runtime_lora_build_cuda_devices)
        if not device_tokens:
            device_tokens = [None]
        effective_pool_size = max(1, self.runtime_lora_build_pool_size)
        if any(device_token is not None for device_token in device_tokens):
            effective_pool_size = min(effective_pool_size, len(device_tokens))
        else:
            effective_pool_size = 1
        return effective_pool_size, device_tokens

    def _extract_runtime_lora_build_requests(self, raw_batch) -> list[dict]:
        if (
            not self.runtime_lora_enabled
            or self.runtime_lora_variant != KNOWLEDGE_LORA_VARIANT
            or not self.runtime_lora_model_path
        ):
            return []
        batch_size = self._batch_size_from_raw_batch(raw_batch)
        if batch_size <= 0:
            return []

        paths = self._pad_optional_values(
            self._stacked_non_tensor_to_list(raw_batch.get(self.runtime_lora_path_key)),
            batch_size,
            fill_value=None,
        )
        knowledge_ids = self._pad_optional_values(
            self._stacked_non_tensor_to_list(raw_batch.get(self.runtime_knowledge_id_key)),
            batch_size,
            fill_value="",
        )
        titles = self._pad_optional_values(self._stacked_non_tensor_to_list(raw_batch.get("title")), batch_size, fill_value="")
        categories = self._pad_optional_values(
            self._stacked_non_tensor_to_list(raw_batch.get("category")),
            batch_size,
            fill_value="",
        )
        subcategories = self._pad_optional_values(
            self._stacked_non_tensor_to_list(raw_batch.get("subcategory")),
            batch_size,
            fill_value="",
        )
        contexts = self._pad_optional_values(self._stacked_non_tensor_to_list(raw_batch.get("context")), batch_size, fill_value="")
        questions = self._pad_optional_values(
            self._stacked_non_tensor_to_list(raw_batch.get("question")),
            batch_size,
            fill_value="",
        )
        answers = self._pad_optional_values(self._stacked_non_tensor_to_list(raw_batch.get("answer")), batch_size, fill_value="")

        build_requests: list[dict] = []
        for row_index in range(batch_size):
            resolved_path = self._normalize_optional_path(paths[row_index])
            knowledge_id = str(knowledge_ids[row_index] or "").strip()
            if resolved_path is None or not knowledge_id:
                continue
            if self._runtime_lora_path_has_required_files(resolved_path):
                continue
            sample = {
                "knowledge_id": knowledge_id,
                "title": str(titles[row_index] or ""),
                "category": str(categories[row_index] or ""),
                "subcategory": str(subcategories[row_index] or ""),
                "context": str(contexts[row_index] or ""),
                "question": str(questions[row_index] or ""),
                "answer": str(answers[row_index] or ""),
            }
            build_requests.append(
                {
                    "request_key": resolved_path,
                    "output_dir": resolved_path,
                    "knowledge_id": knowledge_id,
                    "lora_variant": self.runtime_lora_variant,
                    "sample": sample,
                }
            )
        return build_requests

    def _build_runtime_lora_subprocess(self, *, request: dict, temp_dir: str, device_token: str | None) -> dict:
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "build_ephemeral_lora_adapters.py"
        safe_knowledge_id = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in str(request["knowledge_id"])
        ).strip("._")
        if not safe_knowledge_id:
            safe_knowledge_id = "knowledge"
        sample_file = Path(temp_dir) / f"{safe_knowledge_id}_{os.getpid()}_{self.rank}.json"
        sample_file.write_text(json.dumps(request["sample"], ensure_ascii=False), encoding="utf-8")
        build_spec = resolve_lora_build_spec(
            knowledge_id=f"{request['knowledge_id']}:{request['lora_variant']}",
            seed=self.runtime_lora_seed,
            randomize_lora_config=self.runtime_lora_randomize_config,
            randomization_base_seed=self.runtime_lora_randomization_base_seed,
            lora_rank=self.runtime_lora_rank,
            lora_alpha=self.runtime_lora_alpha,
            lora_dropout=self.runtime_lora_dropout,
            lora_rank_candidates=self.runtime_lora_rank_candidates,
            lora_alpha_candidates=self.runtime_lora_alpha_candidates,
            lora_dropout_candidates=self.runtime_lora_dropout_candidates,
            target_modules=self.runtime_lora_target_modules,
        )
        command = [
            sys.executable,
            os.fspath(script_path),
            "--model-path",
            self.runtime_lora_model_path,
            "--sample-file",
            os.fspath(sample_file),
            "--knowledge-id",
            str(request["knowledge_id"]),
            "--lora-variant",
            str(request["lora_variant"]),
            "--output-dir",
            os.fspath(request["output_dir"]),
            "--steps",
            str(self.runtime_lora_steps),
            "--learning-rate",
            str(self.runtime_lora_learning_rate),
            "--fallback-learning-rate",
            str(self.runtime_lora_fallback_learning_rate),
            "--max-length",
            str(self.runtime_lora_max_length),
            "--lora-rank",
            str(build_spec["lora_rank"]),
            "--lora-alpha",
            str(build_spec["lora_alpha"]),
            "--lora-dropout",
            str(build_spec["lora_dropout"]),
            "--gradient-clip-norm",
            str(self.runtime_lora_gradient_clip_norm),
            "--target-modules",
            ",".join(build_spec["target_modules"]),
            "--seed",
            str(build_spec["seed"]),
            "--dtype",
            str(self.config.data.get("knowledge_lora_dtype", "auto")),
            "--fallback-dtype",
            str(self.config.data.get("knowledge_lora_fallback_dtype", "fp32")),
            "--lock-timeout-seconds",
            str(self.runtime_lora_lock_timeout_seconds),
            "--lock-poll-interval-seconds",
            str(self.runtime_lora_lock_poll_interval_seconds),
        ]
        if self.runtime_lora_base_adapter_path not in (None, "", "None"):
            command.extend(["--base-lora-adapter-path", str(self.runtime_lora_base_adapter_path)])
        if self.runtime_lora_randomize_config:
            command.append("--randomize-lora-config")
            if self.runtime_lora_randomization_base_seed is not None:
                command.extend(["--randomization-base-seed", str(self.runtime_lora_randomization_base_seed)])
            command.extend(["--lora-rank-candidates", ",".join(str(v) for v in self.runtime_lora_rank_candidates)])
            command.extend(["--lora-alpha-candidates", ",".join(str(v) for v in self.runtime_lora_alpha_candidates)])
            command.extend(["--lora-dropout-candidates", ",".join(str(v) for v in self.runtime_lora_dropout_candidates)])
        env = os.environ.copy()
        if device_token is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device_token)
        completed = subprocess.run(
            command,
            cwd=os.fspath(repo_root),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        payload = json.loads(stdout_lines[-1]) if stdout_lines else {}
        payload["device_token"] = device_token
        return payload

    def _prepare_runtime_loras_for_batch(self, raw_batch) -> None:
        build_requests = self._extract_runtime_lora_build_requests(raw_batch)
        if not build_requests:
            return
        deduped_requests: dict[str, dict] = {}
        for request in build_requests:
            deduped_requests.setdefault(request["request_key"], request)
        pending_requests = [request for request in deduped_requests.values() if not self._runtime_lora_path_has_required_files(request["output_dir"])]
        if not pending_requests:
            return
        effective_pool_size, device_tokens = self._get_runtime_lora_build_pool_spec()
        if effective_pool_size <= 1 or len(pending_requests) <= 1:
            with tempfile.TemporaryDirectory(prefix="knowledge_update_sft_runtime_lora_") as temp_dir:
                for request in pending_requests:
                    self._build_runtime_lora_subprocess(request=request, temp_dir=temp_dir, device_token=device_tokens[0])
            return
        with tempfile.TemporaryDirectory(prefix="knowledge_update_sft_runtime_lora_") as temp_dir:
            future_to_request = {}
            with ThreadPoolExecutor(max_workers=effective_pool_size) as executor:
                for request_index, request in enumerate(pending_requests):
                    future = executor.submit(
                        self._build_runtime_lora_subprocess,
                        request=request,
                        temp_dir=temp_dir,
                        device_token=device_tokens[request_index % len(device_tokens)],
                    )
                    future_to_request[future] = request
                for future in as_completed(future_to_request):
                    future.result()

    def _should_delete_runtime_lora_path(self, path: str | None) -> bool:
        if not self.runtime_lora_delete_after_use:
            return False
        normalized_path = self._normalize_optional_path(path)
        if normalized_path is None:
            return False
        if self.runtime_lora_root_dir in (None, "", "None"):
            return False
        root = Path(str(self.runtime_lora_root_dir)).expanduser()
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank not in (None, ""):
            root = root / f"rank_{local_rank}"
        try:
            Path(normalized_path).resolve().relative_to(root.resolve())
        except ValueError:
            return False
        return True

    def _delete_runtime_loras_for_batch(self, raw_batch) -> None:
        if not self.runtime_lora_delete_after_use:
            return
        batch_size = self._batch_size_from_raw_batch(raw_batch)
        if batch_size <= 0:
            return
        paths = self._pad_optional_values(
            self._stacked_non_tensor_to_list(raw_batch.get(self.runtime_lora_path_key)),
            batch_size,
            fill_value=None,
        )
        for path in {self._normalize_optional_path(item) for item in paths}:
            if path is None or not self._should_delete_runtime_lora_path(path):
                continue
            shutil.rmtree(path, ignore_errors=True)

    def _validate_profiler_interval(self):
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        # Note that in SPMD world, this abstraction has to break
        self.engine = self.training_client.engine

    def _init_engine(self):
        # patch optimizer config
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.training_client.reset()

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def _get_batch_seqlens(self, data):
        # mean over dp group
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        batch_seqlens = batch_seqlens.to(self.device_name)  # (global_bsz // dp)

        dp_group = self.engine.get_data_parallel_group()
        dp_size = self.engine.get_data_parallel_size()

        if dp_size == 1 or dp_group is None:
            return batch_seqlens.tolist()

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * dp_size,),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )  # (global_bsz,)

        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=dp_group,
        )

        batch_seqlens = output_tensor.tolist()
        return batch_seqlens

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        # TODO: add a unified tracking
        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"rank {self.rank}: At start of epoch {epoch}", logger=logger)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                raw_batch = data
                self._prepare_runtime_loras_for_batch(raw_batch)
                try:
                    # construct tensordict
                    data = tu.get_tensordict(tensor_dict=raw_batch, non_tensor_dict=meta_info)
                    batch_seqlens = self._get_batch_seqlens(data=data)
                    # this is necessary. Otherwise, it is interpreted as NonTensorStack
                    batch_seqlens_ntd = NonTensorData(batch_seqlens)

                    tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                    # start profile in SPMD mode
                    if global_step == self.start_profile_step:
                        self.training_client.start_profile()
                    # train for on batch
                    output = self.training_client.train_batch(data=data)

                    if global_step == self.end_profile_step:
                        self.training_client.stop_profile()
                finally:
                    self._delete_runtime_loras_for_batch(raw_batch)

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = tu.get(output, "metrics")

                    # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics
                    for k in ["loss", "grad_norm", "lr", "mfu"]:
                        if k in metrics.keys():
                            value = metrics.pop(k)
                            metrics[f"train/{k}"] = value

                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                # early exit or validation step
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        raw_val_batch = val_data
                        self._prepare_runtime_loras_for_batch(raw_val_batch)
                        try:
                            val_data = tu.get_tensordict(tensor_dict=raw_val_batch, non_tensor_dict=meta_info)
                            output = self.training_client.infer_batch(val_data)
                        finally:
                            self._delete_runtime_loras_for_batch(raw_val_batch)

                        if self.engine.is_mp_src_rank_with_outputs():
                            metrics = tu.get(output, "metrics")
                            val_losses.append(metrics["loss"])

                    if self.engine.is_mp_src_rank_with_outputs():
                        val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                        # average over data parallel group
                        dp_group = self.engine.get_data_parallel_group()
                        if dp_group is not None:
                            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

                    if is_logging:
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    aggressive_empty_cache(force_sync=True)
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset


if __name__ == "__main__":
    main()
