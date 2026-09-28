from __future__ import annotations

import gc
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data._utils.collate import default_collate

from verl.protocol import DataProto
from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.knowledge_update_negative_responses import (
    DEFAULT_NO_CHANGE_RESPONSES,
    DEFAULT_RANDOM_CHANGE_RESPONSES,
    load_response_templates,
)
from verl.utils.model import compute_position_id_with_mask

_COMPANION_TEACHER_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "utils"
    / "dataset"
    / "knowledge_update_teacher_trace_dataset.py"
)
if _COMPANION_TEACHER_DATASET_PATH.exists():
    _teacher_spec = importlib.util.spec_from_file_location(
        "reader_balanced24_teacher_trace_dataset",
        _COMPANION_TEACHER_DATASET_PATH,
    )
    if _teacher_spec is None or _teacher_spec.loader is None:
        raise ImportError(f"Could not load companion teacher dataset: {_COMPANION_TEACHER_DATASET_PATH}")
    _teacher_module = importlib.util.module_from_spec(_teacher_spec)
    _teacher_spec.loader.exec_module(_teacher_module)
    KnowledgeUpdateTeacherTraceDataset = _teacher_module.KnowledgeUpdateTeacherTraceDataset
else:
    from verl.utils.dataset.knowledge_update_teacher_trace_dataset import KnowledgeUpdateTeacherTraceDataset

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

DEFAULT_EPHEMERAL_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
KNOWLEDGE_LORA_VARIANT = "knowledge"
NO_OP_LORA_VARIANT = "no_op"
RANDOM_LORA_VARIANT = "random"
SUPPORTED_LORA_VARIANTS = (
    KNOWLEDGE_LORA_VARIANT,
    NO_OP_LORA_VARIANT,
    RANDOM_LORA_VARIANT,
)
PROCESS_ONLY_REWARD_MODE = "process_only"
DEFAULT_REWARD_MODE = "default"
ALL_VARIANT_REWARD_KEYS = ("existence_reward", "process_reward", "evaluation_reward")
PROCESS_ONLY_VARIANT_REWARD_KEYS = ("process_reward",)
PROCESS_ONLY_HIDDEN_REWARD_KEYS = ("existence_reward", "evaluation_reward", "process_evaluation_reward")


class _PersistentEphemeralLoraBuildWorker:
    def __init__(
        self,
        *,
        command: list[str],
        cwd: str,
        env: dict[str, str],
        device_token: str | None,
        timeout_seconds: float,
    ) -> None:
        self.command = list(command)
        self.cwd = cwd
        self.env = dict(env)
        self.device_token = device_token
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=1,
        )
        ready_payload = self._read_response()
        if not ready_payload.get("ok") or ready_payload.get("event") != "ready":
            self.close()
            raise RuntimeError(
                "Persistent ephemeral LoRA build worker did not become ready: "
                f"device={device_token!r} payload={ready_payload!r}"
            )

    def build(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Persistent ephemeral LoRA build worker exited before request: "
                    f"device={self.device_token!r} returncode={self.process.returncode}"
                )
            if self.process.stdin is None:
                raise RuntimeError("Persistent ephemeral LoRA build worker stdin is closed.")
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            payload = self._read_response()
            if not payload.get("ok"):
                raise RuntimeError(
                    "Persistent ephemeral LoRA build worker failed: "
                    f"device={self.device_token!r} error={payload.get('error')} "
                    f"traceback={payload.get('traceback')}"
                )
            payload.setdefault("device_token", self.device_token)
            return payload

    def close(self) -> None:
        process = self.process
        if process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
        except Exception:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def _read_response(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("Persistent ephemeral LoRA build worker stdout is closed.")
        deadline = time.monotonic() + max(30.0, self.timeout_seconds)
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for persistent ephemeral LoRA build worker response: "
                    f"device={self.device_token!r}"
                )
            line = self.process.stdout.readline()
            if line == "":
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Persistent ephemeral LoRA build worker exited while waiting for response: "
                        f"device={self.device_token!r} returncode={self.process.returncode}"
                    )
                time.sleep(0.1)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring non-JSON persistent LoRA worker stdout line: %s", line[:500])
                continue
            if isinstance(payload, dict):
                return payload


def _resolve_reward_mode() -> str:
    raw_value = os.getenv("KNOWLEDGE_UPDATE_REWARD_MODE", DEFAULT_REWARD_MODE)
    normalized_value = str(raw_value).strip().lower().replace("-", "_")
    if normalized_value in {"process", "process_reward", PROCESS_ONLY_REWARD_MODE}:
        return PROCESS_ONLY_REWARD_MODE
    return DEFAULT_REWARD_MODE


def _resolve_variant_reward_keys() -> tuple[str, ...]:
    if _resolve_reward_mode() == PROCESS_ONLY_REWARD_MODE:
        return PROCESS_ONLY_VARIANT_REWARD_KEYS
    return ALL_VARIANT_REWARD_KEYS


class KnowledgeUpdatePPOTrainer(RayPPOTrainer):
    """PPO trainer scaffold for update-awareness experiments.

    This trainer keeps the existing VeRL PPO loop intact while inserting a
    dedicated hook around rollout generation. The hook is where per-QA
    ephemeral LoRA lifecycle management will live.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_update_config = self.config.trainer.get("knowledge_update", OmegaConf.create({}))
        self.enable_ephemeral_lora = bool(self.knowledge_update_config.get("enable_ephemeral_lora", False))
        self.require_single_knowledge_id = bool(self.knowledge_update_config.get("require_single_knowledge_id", True))
        self.ephemeral_lora_path_field = str(
            self.knowledge_update_config.get("ephemeral_lora_path_field", "ephemeral_lora_path")
        )
        self.ephemeral_lora_dir = self.knowledge_update_config.get("ephemeral_lora_dir", None)
        self.ephemeral_lora_request_field = str(
            self.knowledge_update_config.get("ephemeral_lora_request_field", "ephemeral_lora_request")
        )
        self.ephemeral_lora_variants_field = str(
            self.knowledge_update_config.get("ephemeral_lora_variants_field", "ephemeral_lora_variants")
        )
        self.ephemeral_lora_variant_field = str(
            self.knowledge_update_config.get("ephemeral_lora_variant_field", "ephemeral_lora_variant")
        )
        self.multi_knowledge_strategy = str(self.knowledge_update_config.get("multi_knowledge_strategy", "error"))
        self.max_simultaneous_ephemeral_loras = int(
            self.knowledge_update_config.get("max_simultaneous_ephemeral_loras", 1)
        )
        self.max_ephemeral_loras_per_async_server = int(
            self.knowledge_update_config.get("max_ephemeral_loras_per_async_server", 1)
        )
        self.allow_missing_ephemeral_lora = bool(
            self.knowledge_update_config.get("allow_missing_ephemeral_lora", False)
        )
        self.build_missing_ephemeral_lora = bool(
            self.knowledge_update_config.get("build_missing_ephemeral_lora", False)
        )
        self.ephemeral_lora_build_source = str(
            self.knowledge_update_config.get("ephemeral_lora_build_source", "base")
        ).strip().lower()
        self.ephemeral_lora_build_model_path = self._resolve_ephemeral_lora_build_model_path()
        self.ephemeral_lora_build_policy_lora_adapter_path = self._normalize_optional_path(
            self.knowledge_update_config.get("ephemeral_lora_build_policy_lora_adapter_path", None)
        )
        self.ephemeral_lora_build_steps = int(self.knowledge_update_config.get("ephemeral_lora_build_steps", 96))
        self.ephemeral_lora_build_learning_rate = float(
            self.knowledge_update_config.get("ephemeral_lora_build_learning_rate", 2e-5)
        )
        self.ephemeral_lora_build_fallback_learning_rate = self.knowledge_update_config.get(
            "ephemeral_lora_build_fallback_learning_rate",
            None,
        )
        self.ephemeral_lora_build_max_length = int(
            self.knowledge_update_config.get("ephemeral_lora_build_max_length", 512)
        )
        self.ephemeral_lora_build_lora_rank = int(
            self.knowledge_update_config.get("ephemeral_lora_build_lora_rank", 256)
        )
        self.ephemeral_lora_build_lora_alpha = int(
            self.knowledge_update_config.get("ephemeral_lora_build_lora_alpha", 32)
        )
        self.ephemeral_lora_build_lora_dropout = float(
            self.knowledge_update_config.get("ephemeral_lora_build_lora_dropout", 0.0)
        )
        self.ephemeral_lora_build_use_rslora = self._normalize_bool(
            self.knowledge_update_config.get("ephemeral_lora_build_use_rslora", False)
        )
        self.ephemeral_lora_build_train_prompt_mode = str(
            self.knowledge_update_config.get("ephemeral_lora_build_train_prompt_mode", "qwen_bare")
        ).strip()
        self.ephemeral_lora_build_chat_target_mode = str(
            self.knowledge_update_config.get("ephemeral_lora_build_chat_target_mode", "answer_only")
        ).strip()
        self.ephemeral_lora_build_answer_field = str(
            self.knowledge_update_config.get("ephemeral_lora_build_answer_field", "answer")
        ).strip()
        self.ephemeral_lora_build_randomize_config = self._normalize_bool(
            self.knowledge_update_config.get("ephemeral_lora_build_randomize_config", False)
        )
        self.ephemeral_lora_build_random_base_seed = int(
            self.knowledge_update_config.get("ephemeral_lora_build_random_base_seed", 42)
        )
        self.ephemeral_lora_build_randomization_mode = str(
            self.knowledge_update_config.get("ephemeral_lora_build_randomization_mode", "random")
        ).strip().lower()
        if self.ephemeral_lora_build_randomization_mode not in {"random", "deterministic"}:
            raise ValueError(
                "trainer.knowledge_update.ephemeral_lora_build_randomization_mode must be "
                "'random' or 'deterministic'."
            )
        self.ephemeral_lora_build_randomization_seed = int(
            self.knowledge_update_config.get("ephemeral_lora_build_randomization_seed", 0)
        )
        self.ephemeral_lora_build_lora_rank_candidates = self._normalize_int_candidates(
            self.knowledge_update_config.get("ephemeral_lora_build_lora_rank_candidates", None)
        )
        self.ephemeral_lora_build_lora_alpha_candidates = self._normalize_int_candidates(
            self.knowledge_update_config.get("ephemeral_lora_build_lora_alpha_candidates", None)
        )
        self.ephemeral_lora_build_lora_dropout_candidates = self._normalize_float_candidates(
            self.knowledge_update_config.get("ephemeral_lora_build_lora_dropout_candidates", None)
        )
        self.ephemeral_lora_build_lora_recipe_pool = str(
            self.knowledge_update_config.get("ephemeral_lora_build_lora_recipe_pool", "")
        ).strip()
        self.ephemeral_lora_build_quality_max_last_loss = self._normalize_optional_float(
            self.knowledge_update_config.get("ephemeral_lora_build_quality_max_last_loss", None)
        )
        self.ephemeral_lora_build_quality_min_l2_norm = self._normalize_optional_float(
            self.knowledge_update_config.get("ephemeral_lora_build_quality_min_l2_norm", None)
        )
        self.ephemeral_lora_build_quality_max_l2_norm = self._normalize_optional_float(
            self.knowledge_update_config.get("ephemeral_lora_build_quality_max_l2_norm", None)
        )
        self.ephemeral_lora_build_quality_retry_step_multiplier = float(
            self.knowledge_update_config.get("ephemeral_lora_build_quality_retry_step_multiplier", 2.0)
        )
        self.ephemeral_lora_build_quality_safe_lora_rank = int(
            self.knowledge_update_config.get("ephemeral_lora_build_quality_safe_lora_rank", 256)
        )
        self.ephemeral_lora_build_quality_safe_lora_alpha = int(
            self.knowledge_update_config.get("ephemeral_lora_build_quality_safe_lora_alpha", 32)
        )
        self.ephemeral_lora_build_telemetry_csv = str(
            self.knowledge_update_config.get("ephemeral_lora_build_telemetry_csv", "")
        ).strip()
        self.ephemeral_lora_build_early_stop_loss = self._normalize_optional_float(
            self.knowledge_update_config.get("ephemeral_lora_build_early_stop_loss", None)
        )
        self.ephemeral_lora_build_early_stop_min_steps = int(
            self.knowledge_update_config.get("ephemeral_lora_build_early_stop_min_steps", 8)
        )
        self.ephemeral_lora_build_gradient_clip_norm = self.knowledge_update_config.get(
            "ephemeral_lora_build_gradient_clip_norm",
            1.0,
        )
        self.ephemeral_lora_build_dtype = str(self.knowledge_update_config.get("ephemeral_lora_build_dtype", "auto"))
        self.ephemeral_lora_build_fallback_dtype = str(
            self.knowledge_update_config.get("ephemeral_lora_build_fallback_dtype", "fp32")
        )
        self.ephemeral_lora_build_target_modules = self._normalize_target_modules(
            self.knowledge_update_config.get("ephemeral_lora_build_target_modules", None)
        )
        self.ephemeral_lora_build_backend = str(
            self.knowledge_update_config.get("ephemeral_lora_build_backend", "subprocess")
        ).strip().lower()
        if self.ephemeral_lora_build_backend not in {"subprocess", "persistent"}:
            raise ValueError(
                "`trainer.knowledge_update.ephemeral_lora_build_backend` must be 'subprocess' or 'persistent', "
                f"got {self.ephemeral_lora_build_backend!r}."
            )
        self._ephemeral_lora_build_workers: list[_PersistentEphemeralLoraBuildWorker] | None = None
        self.ephemeral_lora_build_pool_size = int(
            self.knowledge_update_config.get("ephemeral_lora_build_pool_size", 1)
        )
        self.ephemeral_lora_build_timeout_seconds = float(
            self.knowledge_update_config.get("ephemeral_lora_build_timeout_seconds", 7200)
        )
        self.ephemeral_lora_build_poll_interval_seconds = float(
            self.knowledge_update_config.get("ephemeral_lora_build_poll_interval_seconds", 1.0)
        )
        self.ephemeral_lora_build_cuda_devices = self._resolve_ephemeral_lora_build_cuda_devices(
            self.knowledge_update_config.get("ephemeral_lora_build_cuda_devices", None)
        )
        self.verify_ephemeral_lora_state = bool(
            self.knowledge_update_config.get("verify_ephemeral_lora_state", False)
        )
        self.log_ephemeral_lora_events = bool(self.knowledge_update_config.get("log_ephemeral_lora_events", True))
        self.memory_diagnostics = bool(self.knowledge_update_config.get("memory_diagnostics", False))
        self.default_lora_variant = self._normalize_lora_variant(
            self.knowledge_update_config.get("default_lora_variant", KNOWLEDGE_LORA_VARIANT)
        )
        self.enabled_lora_variants = self._normalize_lora_variants(
            self.knowledge_update_config.get("enabled_lora_variants", [self.default_lora_variant])
        )
        self.lora_variant_weights = self._normalize_lora_variant_weights(
            self.knowledge_update_config.get("lora_variant_weights", None),
            enabled_variants=self.enabled_lora_variants,
        )
        self.lora_variant_selection_seed = int(self.knowledge_update_config.get("lora_variant_selection_seed", 0))
        self.no_op_reward_gate_threshold = float(self.knowledge_update_config.get("no_op_reward_gate_threshold", 0.95))
        self.no_op_reward_gate_window = int(self.knowledge_update_config.get("no_op_reward_gate_window", 32))
        self.log_variant_metrics = bool(self.knowledge_update_config.get("log_variant_metrics", True))
        self.log_variant_entropy = bool(self.knowledge_update_config.get("log_variant_entropy", True))
        self.log_variant_lora_norm = bool(self.knowledge_update_config.get("log_variant_lora_norm", True))
        self.delete_ephemeral_lora_after_use = bool(
            self.knowledge_update_config.get("delete_ephemeral_lora_after_use", False)
        )
        self.use_ephemeral_lora_for_loss = bool(
            self.knowledge_update_config.get("use_ephemeral_lora_for_loss", self.enable_ephemeral_lora)
        )
        self._pending_ephemeral_lora_deletes: dict[str, dict[str, Any]] = {}
        self.process_reward_positive_sample_limit = int(
            self.knowledge_update_config.get("process_reward_positive_sample_limit", 10)
        )
        self.process_reward_positive_samples_path = self._resolve_process_reward_positive_samples_path()
        self._process_reward_positive_samples: list[dict[str, Any]] = []
        self._process_reward_positive_sample_keys: set[str] = set()
        self._load_existing_process_reward_positive_samples()
        self.variant_reward_keys = _resolve_variant_reward_keys()
        self._no_op_reward_gate_open = NO_OP_LORA_VARIANT not in self.enabled_lora_variants
        self._knowledge_reward_history = {
            reward_key: deque(maxlen=max(1, self.no_op_reward_gate_window))
            for reward_key in self.variant_reward_keys
        }
        self._ephemeral_lora_events: list[dict[str, Any]] = []
        self._ephemeral_lora_norm_cache: dict[str, float] = {}
        self.luffy_config = self.knowledge_update_config.get("luffy", OmegaConf.create({}))
        self.enable_luffy_teacher_loss = bool(self.luffy_config.get("enable", False))
        self.luffy_teacher_dataset = None
        self.luffy_teacher_seed = int(self.luffy_config.get("seed", 0))
        self.luffy_teacher_rng = np.random.default_rng(self.luffy_teacher_seed)
        self.luffy_negative_sample_seed = int(self.luffy_config.get("negative_sample_seed", 0))
        self.luffy_negative_rng = np.random.default_rng(self.luffy_negative_sample_seed)
        self.luffy_teacher_max_length = int(
            self.luffy_config.get("max_length", self.config.data.get("max_prompt_length", 2048))
        )
        self.luffy_per_type_schedule_file = str(
            self.luffy_config.get("per_type_schedule_file", "")
        ).strip()
        self.luffy_per_type_source_file = str(
            self.luffy_config.get("per_type_source_file", "")
        ).strip()
        self.luffy_per_type_schedule_strict = bool(
            self.luffy_config.get("per_type_schedule_strict", True)
        )
        self._luffy_per_type_schedule: list[dict[str, Any]] = []
        self._luffy_per_type_source_by_knowledge: dict[str, dict[str, Any]] = {}
        self.luffy_teacher_advantage_value = float(self.luffy_config.get("teacher_advantage_value", 1.0))
        self.luffy_no_change_responses = load_response_templates(
            self.luffy_config.get("no_change_responses_file", None),
            defaults=DEFAULT_NO_CHANGE_RESPONSES,
            name="no_change",
        )
        self.luffy_random_change_responses = load_response_templates(
            self.luffy_config.get("random_change_responses_file", None),
            defaults=DEFAULT_RANDOM_CHANGE_RESPONSES,
            name="random_change",
        )
        self._luffy_teacher_indices_by_knowledge: dict[str, list[int]] = {}
        self._luffy_teacher_indices_by_pair: dict[tuple[str, str], list[int]] = {}
        self._luffy_teacher_indices_by_sample_hash_pair: dict[tuple[str, str], list[int]] = {}
        if self.enable_luffy_teacher_loss:
            self._enable_luffy_teacher_loss_config()
            self.luffy_teacher_dataset = self._build_luffy_teacher_dataset()
            self._build_luffy_teacher_lookup_maps()
            self._load_luffy_per_type_schedule()
        self._ensure_rollout_correction_bypass_mode()

    def _enable_luffy_teacher_loss_config(self) -> None:
        with open_dict(self.config):
            actor_cfg = self.config.actor_rollout_ref.actor
            use_sft_multitask = bool(actor_cfg.get("use_sft_multitask_loss", False))
            use_off_policy = bool(actor_cfg.get("use_off_policy_loss", False))
            if use_sft_multitask and use_off_policy:
                raise ValueError("LUFFY teacher loss cannot enable both use_sft_multitask_loss and use_off_policy_loss")
            if not use_sft_multitask and not use_off_policy:
                actor_cfg.use_sft_multitask_loss = True
                use_sft_multitask = True
            if use_sft_multitask and actor_cfg.get("sft_loss_coef", None) is None:
                actor_cfg.sft_loss_coef = 1.0
            if use_off_policy:
                if actor_cfg.get("off_policy_loss_impl", None) is None:
                    actor_cfg.off_policy_loss_impl = "token"
                if actor_cfg.get("off_policy_cliprange", None) is None:
                    actor_cfg.off_policy_cliprange = 0.2
                if actor_cfg.get("clip_upper_bound", None) is None:
                    actor_cfg.clip_upper_bound = 1.0

    def _build_luffy_teacher_dataset(self):
        teacher_files = self.luffy_config.get("train_files", None)
        if teacher_files is None:
            raise ValueError("trainer.knowledge_update.luffy.train_files must be set when luffy.enable=True")
        if isinstance(teacher_files, str):
            teacher_files = [teacher_files]
        teacher_files = [str(item) for item in teacher_files if str(item).strip()]
        if not teacher_files:
            raise ValueError("trainer.knowledge_update.luffy.train_files is empty")

        dataset_config = OmegaConf.create({
            "messages_key": str(self.luffy_config.get("messages_key", "messages")),
            "max_length": int(self.luffy_teacher_max_length),
            "truncation": str(self.luffy_config.get("truncation", "right")),
            "pad_mode": str(self.luffy_config.get("pad_mode", "right")),
            "ignore_input_ids_mismatch": bool(self.luffy_config.get("ignore_input_ids_mismatch", True)),
            "knowledge_id_key": str(self.luffy_config.get("knowledge_id_key", "knowledge_id")),
            "knowledge_lora_path_key": str(self.luffy_config.get("knowledge_lora_path_key", "knowledge_lora_path")),
            "knowledge_lora_root_dir": self.luffy_config.get("knowledge_lora_root_dir", None),
            "knowledge_lora_variant": str(self.luffy_config.get("knowledge_lora_variant", "knowledge")),
            "knowledge_lora_rank_local_root": bool(self.luffy_config.get("knowledge_lora_rank_local_root", True)),
            "target_probs_key": self.luffy_config.get("target_probs_key", None),
        })
        return KnowledgeUpdateTeacherTraceDataset(
            parquet_files=teacher_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=dataset_config,
            max_samples=int(self.luffy_config.get("max_samples", -1)),
        )

    def _build_luffy_teacher_lookup_maps(self) -> None:
        self._luffy_teacher_indices_by_knowledge = {}
        self._luffy_teacher_indices_by_pair = {}
        self._luffy_teacher_indices_by_sample_hash_pair = {}
        if self.luffy_teacher_dataset is None:
            return
        dataframe = getattr(self.luffy_teacher_dataset, "dataframe", None)
        if dataframe is None:
            return
        knowledge_id_key = getattr(self.luffy_teacher_dataset, "knowledge_id_key", "knowledge_id")
        has_meta_query = "meta_query" in dataframe.columns
        has_sample_hash = "sample_hash" in dataframe.columns
        for row_index, row in enumerate(dataframe.to_dict(orient="records")):
            knowledge_id = str(row.get(knowledge_id_key, "")).strip()
            if not knowledge_id:
                continue
            self._luffy_teacher_indices_by_knowledge.setdefault(knowledge_id, []).append(row_index)
            if has_meta_query:
                meta_query = str(row.get("meta_query", "")).strip()
                if meta_query:
                    self._luffy_teacher_indices_by_pair.setdefault((knowledge_id, meta_query), []).append(row_index)
                    if has_sample_hash:
                        sample_hash = str(row.get("sample_hash", "")).strip()
                        if sample_hash:
                            self._luffy_teacher_indices_by_sample_hash_pair.setdefault(
                                (sample_hash, meta_query),
                                [],
                            ).append(row_index)

    def _load_luffy_per_type_schedule(self) -> None:
        if not self.luffy_per_type_schedule_file:
            return
        if self.luffy_teacher_dataset is None:
            raise ValueError("A per-type LUFFY schedule requires an initialized teacher dataset.")
        if not self._normalize_bool(self.luffy_config.get("warmup_teacher_only", False)):
            raise ValueError("A per-type LUFFY schedule is only supported with warmup_teacher_only=true.")

        schedule_path = Path(self.luffy_per_type_schedule_file).expanduser()
        if not schedule_path.exists():
            raise FileNotFoundError(f"Per-type LUFFY schedule does not exist: {schedule_path}")
        source_path = Path(self.luffy_per_type_source_file).expanduser() if self.luffy_per_type_source_file else None
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(
                "A per-type LUFFY schedule requires per_type_source_file with the full source metadata."
            )

        source_by_knowledge: dict[str, dict[str, Any]] = {}
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                knowledge_id = str(row.get("knowledge_id") or row.get("reader_id") or "").strip()
                if not knowledge_id:
                    raise ValueError(f"Missing knowledge_id in {source_path}:{line_number}")
                if knowledge_id in source_by_knowledge:
                    raise ValueError(f"Duplicate knowledge_id in per-type source file: {knowledge_id}")
                source_by_knowledge[knowledge_id] = row

        dataframe = getattr(self.luffy_teacher_dataset, "dataframe", None)
        if dataframe is None:
            raise ValueError("Per-type LUFFY schedule requires a teacher dataset dataframe.")
        if "update_type" not in dataframe.columns:
            raise KeyError("Per-type LUFFY schedule requires update_type in the teacher parquet.")
        knowledge_id_key = getattr(self.luffy_teacher_dataset, "knowledge_id_key", "knowledge_id")
        if knowledge_id_key not in dataframe.columns:
            raise KeyError(f"Per-type LUFFY schedule requires {knowledge_id_key!r} in the teacher parquet.")
        dataset_size = len(dataframe)
        update_types = dataframe["update_type"].astype(str).tolist()
        teacher_knowledge_ids = dataframe[knowledge_id_key].astype(str).tolist()

        schedule: list[dict[str, Any]] = []
        required_index_fields = (
            "knowledge_teacher_indices",
            "behavior_teacher_indices",
            "no_op_teacher_indices",
            "random_teacher_indices",
        )
        with schedule_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                expected_global_step = len(schedule) + 1
                if int(row.get("global_step", -1)) != expected_global_step:
                    raise ValueError(
                        f"Non-sequential per-type schedule step at {schedule_path}:{line_number}; "
                        f"expected {expected_global_step}."
                    )
                index_groups: dict[str, list[int]] = {}
                for field in required_index_fields:
                    values = row.get(field)
                    if not isinstance(values, list):
                        raise TypeError(f"Per-type schedule field {field!r} must be a list.")
                    indices = [int(value) for value in values]
                    if any(index < 0 or index >= dataset_size for index in indices):
                        raise IndexError(f"Per-type schedule field {field!r} contains an out-of-range index.")
                    index_groups[field] = indices
                if len(index_groups["knowledge_teacher_indices"]) != int(row.get("knowledge_count", -1)):
                    raise ValueError("Per-type schedule knowledge_count does not match its index list.")
                if len(index_groups["behavior_teacher_indices"]) != int(row.get("behavior_count", -1)):
                    raise ValueError("Per-type schedule behavior_count does not match its index list.")
                if len(index_groups["no_op_teacher_indices"]) != int(row.get("no_op_count", -1)):
                    raise ValueError("Per-type schedule no_op_count does not match its index list.")
                if len(index_groups["random_teacher_indices"]) != int(row.get("random_count", -1)):
                    raise ValueError("Per-type schedule random_count does not match its index list.")
                flattened = [index for field in required_index_fields for index in index_groups[field]]
                if len(flattened) != int(row.get("batch_size", -1)) or len(set(flattened)) != len(flattened):
                    raise ValueError("Per-type schedule rows must contain one disjoint teacher index per batch row.")
                if any(update_types[index] != "knowledge" for index in index_groups["knowledge_teacher_indices"]):
                    raise ValueError("Knowledge schedule indices do not point to knowledge teacher rows.")
                if any(update_types[index] != "behavior" for index in index_groups["behavior_teacher_indices"]):
                    raise ValueError("Behavior schedule indices do not point to behavior teacher rows.")
                missing_source_ids = {
                    teacher_knowledge_ids[index]
                    for index in flattened
                    if teacher_knowledge_ids[index] not in source_by_knowledge
                }
                if missing_source_ids:
                    raise KeyError(
                        f"Per-type source file is missing {len(missing_source_ids)} scheduled knowledge IDs."
                    )
                normalized_row = dict(row)
                normalized_row.update(index_groups)
                schedule.append(normalized_row)

        if not schedule:
            raise ValueError(f"Per-type LUFFY schedule is empty: {schedule_path}")
        warmup_steps = int(self.luffy_config.get("warmup_steps", 0))
        if self.luffy_per_type_schedule_strict and warmup_steps != len(schedule):
            raise ValueError(
                f"Per-type schedule rows ({len(schedule)}) must equal warmup_steps ({warmup_steps})."
            )
        self._luffy_per_type_source_by_knowledge = source_by_knowledge
        self._luffy_per_type_schedule = schedule
        logger.info(
            "Loaded %d strict per-type LUFFY steps from %s with %d source rows.",
            len(schedule),
            schedule_path,
            len(source_by_knowledge),
        )

    def _current_luffy_per_type_schedule_row(self) -> dict[str, Any] | None:
        if not self._luffy_per_type_schedule:
            return None
        global_step = int(getattr(self, "global_steps", 0))
        schedule_index = global_step - 1
        if schedule_index < 0 or schedule_index >= len(self._luffy_per_type_schedule):
            if self.luffy_per_type_schedule_strict:
                raise IndexError(
                    f"No per-type LUFFY schedule row for global_step={global_step}; "
                    f"available steps={len(self._luffy_per_type_schedule)}."
                )
            return None
        row = self._luffy_per_type_schedule[schedule_index]
        if int(row["global_step"]) != global_step:
            raise AssertionError("Per-type LUFFY schedule/global step mismatch.")
        return row



    def _select_luffy_teacher_dataset_index(
        self,
        *,
        knowledge_id: str | None,
        meta_query: str | None,
        sample_hash: str | None = None,
    ) -> int | None:
        dataset_index, _ = self._select_luffy_teacher_dataset_index_with_status(
            knowledge_id=knowledge_id,
            meta_query=meta_query,
            sample_hash=sample_hash,
        )
        return dataset_index

    def _select_luffy_teacher_dataset_index_with_status(
        self,
        *,
        knowledge_id: str | None,
        meta_query: str | None,
        sample_hash: str | None = None,
    ) -> tuple[int | None, str]:
        normalized_knowledge_id = self._normalize_optional_id(knowledge_id)
        normalized_meta_query = str(meta_query or "").strip()
        normalized_sample_hash = str(sample_hash or "").strip()
        candidate_indices: list[int] = []
        match_status = "miss"
        if normalized_sample_hash and normalized_meta_query:
            candidate_indices = self._luffy_teacher_indices_by_sample_hash_pair.get(
                (normalized_sample_hash, normalized_meta_query),
                [],
            )
            if candidate_indices:
                match_status = "sample_hash_meta_query"
        if normalized_knowledge_id and normalized_meta_query:
            if not candidate_indices:
                candidate_indices = self._luffy_teacher_indices_by_pair.get(
                    (normalized_knowledge_id, normalized_meta_query),
                    [],
                )
                if candidate_indices:
                    match_status = "knowledge_meta_query"
        strict_match = bool(self.luffy_config.get("strict_match", True))
        if not candidate_indices and strict_match:
            return None, "miss"
        if not candidate_indices and normalized_knowledge_id:
            candidate_indices = self._luffy_teacher_indices_by_knowledge.get(normalized_knowledge_id, [])
            if candidate_indices:
                match_status = "knowledge_id"
        if not candidate_indices:
            dataset_len = len(self.luffy_teacher_dataset) if self.luffy_teacher_dataset is not None else 0
            if dataset_len <= 0:
                return None, "miss"
            return int(self.luffy_teacher_rng.integers(low=0, high=dataset_len)), "fallback_random"
        selected_offset = int(self.luffy_teacher_rng.integers(low=0, high=len(candidate_indices)))
        return int(candidate_indices[selected_offset]), match_status

    def _resolve_luffy_base_teacher_sample_size(self, *, template_batch_size: int) -> int:
        if template_batch_size <= 0:
            return 0
        configured_batch_size = int(self.luffy_config.get("batch_size", template_batch_size))
        if configured_batch_size > 0:
            return configured_batch_size

        batch_size_source = str(self.luffy_config.get("batch_size_source", "pre_repeat")).strip().lower()
        if batch_size_source in {"pre_repeat", "prompt", "per_prompt"}:
            rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 1))
            rollout_n = max(1, rollout_n)
            return max(1, (template_batch_size + rollout_n - 1) // rollout_n)
        if batch_size_source not in {"post_repeat", "batch", "rollout"}:
            logger.warning(
                "Unknown LUFFY teacher batch_size_source=%s; falling back to post_repeat.",
                batch_size_source,
            )
        return template_batch_size

    def _luffy_teacher_warmup_active(self) -> bool:
        warmup_steps = int(self.luffy_config.get("warmup_steps", 0))
        # global_steps is 1-based inside fit(), so the configured final step is inclusive.
        return warmup_steps > 0 and int(getattr(self, "global_steps", 0)) <= warmup_steps

    def _luffy_teacher_warmup_allows_replacement(self) -> bool:
        if not self._luffy_teacher_warmup_active():
            return False
        return bool(
            self.luffy_config.get(
                "warmup_sample_with_replacement",
                bool(self.luffy_config.get("warmup_teacher_only", False)),
            )
        )

    def _resolve_luffy_teacher_warmup_quota_counts(self) -> dict[str, int] | None:
        if not self._luffy_teacher_warmup_active():
            return None
        counts = {
            "changed": max(0, int(self.luffy_config.get("warmup_changed_count", 0))),
            "no_change": max(0, int(self.luffy_config.get("warmup_no_change_count", 0))),
            "random_change": max(0, int(self.luffy_config.get("warmup_random_change_count", 0))),
        }
        if sum(counts.values()) <= 0:
            return None
        return counts

    def _resolve_luffy_teacher_sample_size(self, *, template_batch_size: int) -> int:
        base_sample_size = self._resolve_luffy_base_teacher_sample_size(template_batch_size=template_batch_size)
        if base_sample_size <= 0 or not self._luffy_teacher_warmup_active():
            return base_sample_size

        quota_counts = self._resolve_luffy_teacher_warmup_quota_counts()
        if quota_counts is not None:
            return max(1, sum(quota_counts.values()))

        allow_replacement = self._luffy_teacher_warmup_allows_replacement()
        warmup_batch_size = int(self.luffy_config.get("warmup_batch_size", 0))
        if warmup_batch_size > 0:
            return max(1, warmup_batch_size) if allow_replacement else min(template_batch_size, warmup_batch_size)

        warmup_multiplier = float(self.luffy_config.get("warmup_teacher_multiplier", 1.0))
        if warmup_multiplier <= 1.0:
            return base_sample_size
        sample_size = max(base_sample_size, int(round(base_sample_size * warmup_multiplier)))
        warmup_max_batch_size = int(self.luffy_config.get("warmup_max_batch_size", 0))
        if warmup_max_batch_size > 0:
            sample_size = min(sample_size, warmup_max_batch_size)
        if not allow_replacement:
            sample_size = min(template_batch_size, sample_size)
        return max(1, sample_size)

    @staticmethod
    def _build_luffy_teacher_sampling_stats(*, sample_size: int, match_statuses: list[str], matched_count: int) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for status in match_statuses:
            normalized_status = str(status or "unknown")
            status_counts[normalized_status] = status_counts.get(normalized_status, 0) + 1
        miss_count = int(status_counts.get("miss", 0))
        fallback_count = int(status_counts.get("fallback_random", 0))
        return {
            "sampled_rows": int(sample_size),
            "matched_rows": int(matched_count),
            "miss_rows": miss_count,
            "fallback_rows": fallback_count,
            "miss_fraction": float(miss_count / max(sample_size, 1)),
            "fallback_fraction": float(fallback_count / max(sample_size, 1)),
            "match_status_counts": status_counts,
        }

    def _resolve_luffy_teacher_scalar_rewards(
        self,
        *,
        collated_batch: dict[str, Any],
        batch_size: int,
    ) -> list[float]:
        reward_column_map = {
            "existence_reward": "judge_existence_reward",
            "process_reward": "judge_process_reward",
            "evaluation_reward": "judge_evaluation_reward",
        }
        selected_columns = [
            reward_column_map[reward_key]
            for reward_key in self.variant_reward_keys
            if reward_column_map.get(reward_key) in collated_batch
        ]
        if not selected_columns:
            selected_columns = [column_name for column_name in reward_column_map.values() if column_name in collated_batch]

        reward_values: list[float] = []
        for row_index in range(batch_size):
            row_scores: list[float] = []
            for column_name in selected_columns:
                column_values = collated_batch.get(column_name, None)
                if column_values is None or row_index >= len(column_values):
                    continue
                try:
                    score_value = float(column_values[row_index])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(score_value):
                    row_scores.append(score_value)
            reward_values.append(float(np.mean(row_scores)) if row_scores else 0.0)
        return reward_values

    @staticmethod
    def _flatten_token_ids(value: Any) -> list[int]:
        if isinstance(value, torch.Tensor):
            return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, dict):
            if "input_ids" in value:
                return KnowledgeUpdatePPOTrainer._flatten_token_ids(value["input_ids"])
            return []
        if hasattr(value, "input_ids"):
            return KnowledgeUpdatePPOTrainer._flatten_token_ids(value.input_ids)
        if hasattr(value, "ids"):
            return [int(item) for item in value.ids]
        if isinstance(value, np.ndarray):
            return [int(item) for item in value.reshape(-1).tolist()]
        if isinstance(value, (list, tuple)):
            flattened: list[int] = []
            for item in value:
                if isinstance(item, (list, tuple, np.ndarray, torch.Tensor)):
                    flattened.extend(KnowledgeUpdatePPOTrainer._flatten_token_ids(item))
                else:
                    flattened.append(int(item))
            return flattened
        return [int(value)]

    def _apply_luffy_chat_template(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        return self._flatten_token_ids(token_ids)

    def _extract_luffy_prompt_messages(self, row_dict: dict[str, Any]) -> list[dict[str, str]]:
        messages_key = str(self.luffy_config.get("messages_key", "messages"))
        raw_messages = row_dict.get(messages_key)
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except json.JSONDecodeError:
                raw_messages = None
        if hasattr(raw_messages, "tolist"):
            raw_messages = raw_messages.tolist()
        if isinstance(raw_messages, list):
            prompt_messages: list[dict[str, str]] = []
            for raw_message in raw_messages:
                if not isinstance(raw_message, Mapping):
                    continue
                role = str(raw_message.get("role", "")).strip()
                content = str(raw_message.get("content", "")).strip()
                if role == "assistant":
                    break
                if role and content:
                    prompt_messages.append({"role": role, "content": content})
            if prompt_messages:
                return prompt_messages

        meta_query = str(row_dict.get("meta_query", "")).strip()
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": meta_query},
        ]

    def _tokenize_luffy_messages(
        self,
        *,
        prompt_messages: list[dict[str, str]],
        assistant_response: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_messages = prompt_messages + [{"role": "assistant", "content": str(assistant_response).strip()}]
        prompt_ids = self._apply_luffy_chat_template(prompt_messages, add_generation_prompt=True)
        full_ids = self._apply_luffy_chat_template(full_messages, add_generation_prompt=False)
        response_start = min(len(prompt_ids), len(full_ids))
        while response_start > 0 and full_ids[:response_start] != prompt_ids[:response_start]:
            response_start -= 1
        loss_mask_values = [0] * len(full_ids)
        for token_index in range(response_start, len(full_ids)):
            loss_mask_values[token_index] = 1

        max_length = int(self.luffy_teacher_max_length)
        truncation = str(self.luffy_config.get("truncation", "right")).strip().lower()
        if len(full_ids) > max_length:
            if truncation == "left":
                full_ids = full_ids[-max_length:]
                loss_mask_values = loss_mask_values[-max_length:]
            else:
                full_ids = full_ids[:max_length]
                loss_mask_values = loss_mask_values[:max_length]

        pad_mode = str(self.luffy_config.get("pad_mode", "right")).strip().lower()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        attention_values = [1] * len(full_ids)
        pad_length = max(0, max_length - len(full_ids))
        if pad_length > 0 and pad_mode == "left":
            full_ids = [int(pad_token_id)] * pad_length + full_ids
            attention_values = [0] * pad_length + attention_values
            loss_mask_values = [0] * pad_length + loss_mask_values
        elif pad_length > 0:
            full_ids = full_ids + [int(pad_token_id)] * pad_length
            attention_values = attention_values + [0] * pad_length
            loss_mask_values = loss_mask_values + [0] * pad_length

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(attention_values, dtype=torch.long),
            torch.tensor(loss_mask_values, dtype=torch.long),
        )

    def _select_luffy_negative_response(self, sample_type: str) -> str:
        if sample_type == "no_change":
            templates = self.luffy_no_change_responses
        elif sample_type == "random_change":
            templates = self.luffy_random_change_responses
        else:
            raise ValueError(f"Unsupported LUFFY negative sample_type={sample_type!r}")
        template_index = int(self.luffy_negative_rng.integers(low=0, high=len(templates)))
        return str(templates[template_index])

    def _select_scheduled_luffy_negative_response(
        self,
        *,
        sample_type: str,
        global_step: int,
        sample_position: int,
    ) -> str:
        if sample_type == "no_change":
            templates = self.luffy_no_change_responses
        elif sample_type == "random_change":
            templates = self.luffy_random_change_responses
        else:
            raise ValueError(f"Unsupported scheduled LUFFY negative sample_type={sample_type!r}")
        seed = self.luffy_negative_sample_seed + global_step * 1000 + sample_position
        template_index = int(np.random.default_rng(seed).integers(low=0, high=len(templates)))
        return str(templates[template_index])

    def _build_luffy_negative_teacher_sample(
        self,
        *,
        dataset_index: int,
        sample_type: str,
        response_text: str,
    ) -> dict[str, Any]:
        if self.luffy_teacher_dataset is None:
            raise ValueError("LUFFY teacher dataset is not initialized.")
        dataframe = getattr(self.luffy_teacher_dataset, "dataframe", None)
        if dataframe is None:
            raise ValueError("LUFFY teacher dataset does not expose a dataframe.")
        row_dict = dataframe.iloc[int(dataset_index)].to_dict()
        prompt_messages = self._extract_luffy_prompt_messages(row_dict)
        input_ids, attention_mask, loss_mask = self._tokenize_luffy_messages(
            prompt_messages=prompt_messages,
            assistant_response=response_text,
        )
        knowledge_id_key = getattr(self.luffy_teacher_dataset, "knowledge_id_key", "knowledge_id")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            knowledge_id_key: str(row_dict.get(knowledge_id_key, "")),
            "title": str(row_dict.get("title", "")),
            "category": str(row_dict.get("category", "")),
            "subcategory": str(row_dict.get("subcategory", "")),
            "context": str(row_dict.get("context", "")),
            "question": str(row_dict.get("question", "")),
            "answer": str(row_dict.get("answer", "")),
            "meta_query": str(row_dict.get("meta_query", "")),
            "sample_hash": str(row_dict.get("sample_hash", "")),
            "reader_id": str(row_dict.get("reader_id", row_dict.get(knowledge_id_key, ""))),
            "update_type": str(row_dict.get("update_type", "")),
            "reader_target": str(row_dict.get("reader_target", "")),
            "reader_target_kind": str(row_dict.get("reader_target_kind", "")),
            "luffy_sample_type": sample_type,
            "luffy_negative_target": response_text,
            "judge_process_reward": 1.0,
            "judge_existence_reward": 1.0,
            "judge_evaluation_reward": 1.0,
        }

    def _collate_luffy_teacher_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {}
        knowledge_id_key = getattr(self.luffy_teacher_dataset, "knowledge_id_key", "knowledge_id")
        collate_keys = [
            "input_ids",
            "attention_mask",
            "loss_mask",
            knowledge_id_key,
            "title",
            "category",
            "subcategory",
            "context",
            "question",
            "answer",
            "meta_query",
            "sample_hash",
            "reader_id",
            "update_type",
            "reader_target",
            "reader_target_kind",
            "judge_process_reward",
            "judge_existence_reward",
            "judge_evaluation_reward",
            "luffy_sample_type",
            "luffy_negative_target",
        ]
        if any("target_probs" in sample for sample in samples):
            collate_keys.append("target_probs")

        normalized_samples: list[dict[str, Any]] = []
        for sample in samples:
            normalized: dict[str, Any] = {}
            for key in collate_keys:
                if key in sample:
                    normalized[key] = sample[key]
                    continue
                if key == "target_probs":
                    normalized[key] = torch.zeros_like(sample["input_ids"], dtype=torch.float32)
                elif key in {"judge_process_reward", "judge_existence_reward", "judge_evaluation_reward"}:
                    normalized[key] = float("nan")
                elif key in {"input_ids", "attention_mask", "loss_mask"}:
                    raise KeyError(f"LUFFY teacher sample is missing required tensor key {key!r}")
                else:
                    normalized[key] = ""
            normalized_samples.append(normalized)
        return default_collate(normalized_samples)

    def _sample_luffy_teacher_batch_from_schedule(
        self,
        *,
        batch: DataProto,
        schedule_row: dict[str, Any],
    ) -> DataProto:
        template_batch_size = len(batch)
        dataset_indices = (
            list(schedule_row["knowledge_teacher_indices"])
            + list(schedule_row["behavior_teacher_indices"])
            + list(schedule_row["no_op_teacher_indices"])
            + list(schedule_row["random_teacher_indices"])
        )
        sample_types = (
            ["changed"] * len(schedule_row["knowledge_teacher_indices"])
            + ["changed"] * len(schedule_row["behavior_teacher_indices"])
            + ["no_change"] * len(schedule_row["no_op_teacher_indices"])
            + ["random_change"] * len(schedule_row["random_teacher_indices"])
        )
        if self.luffy_per_type_schedule_strict and template_batch_size != len(dataset_indices):
            raise ValueError(
                f"Per-type schedule batch size {len(dataset_indices)} does not match "
                f"the runtime template batch size {template_batch_size}."
            )

        samples: list[dict[str, Any]] = []
        sample_lora_variants: list[str] = []
        sample_mount_lora: list[bool] = []
        sample_targets: list[str] = []
        global_step = int(schedule_row["global_step"])
        for sample_position, (dataset_index, sample_type) in enumerate(zip(dataset_indices, sample_types)):
            if sample_type == "changed":
                sample = self.luffy_teacher_dataset[int(dataset_index)]
                lora_variant = KNOWLEDGE_LORA_VARIANT
                mount_lora = True
                target_text = ""
            else:
                target_text = self._select_scheduled_luffy_negative_response(
                    sample_type=sample_type,
                    global_step=global_step,
                    sample_position=sample_position,
                )
                sample = self._build_luffy_negative_teacher_sample(
                    dataset_index=int(dataset_index),
                    sample_type=sample_type,
                    response_text=target_text,
                )
                lora_variant = NO_OP_LORA_VARIANT if sample_type == "no_change" else RANDOM_LORA_VARIANT
                mount_lora = sample_type == "random_change"
            samples.append(sample)
            sample_lora_variants.append(lora_variant)
            sample_mount_lora.append(mount_lora)
            sample_targets.append(target_text)

        match_statuses = ["per_type_schedule"] * len(dataset_indices)
        sampling_stats = self._build_luffy_teacher_sampling_stats(
            sample_size=len(dataset_indices),
            match_statuses=match_statuses,
            matched_count=len(dataset_indices),
        )
        sampling_stats.update(
            {
                "template_batch_size": int(template_batch_size),
                "base_sampled_rows": self._resolve_luffy_base_teacher_sample_size(
                    template_batch_size=template_batch_size
                ),
                "sample_with_replacement": 0.0,
                "warmup_active": float(self._luffy_teacher_warmup_active()),
                "warmup_sample_with_replacement": 0.0,
                "warmup_teacher_multiplier": float(
                    self.luffy_config.get("warmup_teacher_multiplier", 1.0)
                ),
                "per_type_schedule_active": 1.0,
                "scheduled_knowledge_changed": int(schedule_row["knowledge_count"]),
                "scheduled_behavior_changed": int(schedule_row["behavior_count"]),
                "scheduled_no_change": int(schedule_row["no_op_count"]),
                "scheduled_random_change": int(schedule_row["random_count"]),
                "scheduled_global_step": global_step,
            }
        )
        batch.meta_info["luffy_teacher_sampling_stats"] = sampling_stats
        collated = self._collate_luffy_teacher_samples(samples)
        return self._convert_luffy_teacher_collated_batch(
            collated,
            template_batch=batch,
            selected_row_indices=list(range(len(dataset_indices))),
            match_statuses=match_statuses,
            sample_types=sample_types,
            sample_lora_variants=sample_lora_variants,
            sample_mount_lora=sample_mount_lora,
            sample_targets=sample_targets,
            teacher_dataset_indices=[int(index) for index in dataset_indices],
        )

    def _sample_luffy_teacher_batch(self, batch: DataProto) -> DataProto | None:
        if self.luffy_teacher_dataset is None:
            return None
        template_batch_size = len(batch)
        if template_batch_size <= 0:
            return None
        sample_size = self._resolve_luffy_teacher_sample_size(template_batch_size=template_batch_size)
        if sample_size <= 0:
            return None
        schedule_row = self._current_luffy_per_type_schedule_row()
        if schedule_row is not None:
            return self._sample_luffy_teacher_batch_from_schedule(batch=batch, schedule_row=schedule_row)
        quota_counts = self._resolve_luffy_teacher_warmup_quota_counts()
        sample_types: list[str]
        if quota_counts is not None:
            sample_types = (
                ["changed"] * int(quota_counts.get("changed", 0))
                + ["no_change"] * int(quota_counts.get("no_change", 0))
                + ["random_change"] * int(quota_counts.get("random_change", 0))
            )
            sample_size = len(sample_types)
        else:
            sample_types = ["changed"] * sample_size

        batch_size_source = str(self.luffy_config.get("batch_size_source", "pre_repeat")).strip().lower()
        rollout_n = max(1, int(self.config.actor_rollout_ref.rollout.get("n", 1)))
        should_sample_pre_repeat_groups = (
            batch_size_source in {"pre_repeat", "prompt", "per_prompt"}
            and rollout_n > 1
            and sample_size == self._resolve_luffy_base_teacher_sample_size(template_batch_size=template_batch_size)
            and not self._luffy_teacher_warmup_active()
        )
        if should_sample_pre_repeat_groups:
            # The rollout batch has already been interleaved by rollout.n. For LUFFY 15+1,
            # add exactly one teacher row per original query, not random rows from the 15 repeats.
            selected_row_indices = list(range(0, template_batch_size, rollout_n))[:sample_size]
        else:
            replace = sample_size > template_batch_size or (
                quota_counts is not None and self._luffy_teacher_warmup_allows_replacement()
            )
            selected_row_indices = self.luffy_teacher_rng.choice(template_batch_size, size=sample_size, replace=replace)
            selected_row_indices = [int(index) for index in selected_row_indices.tolist()]

        non_tensor_batch = getattr(batch, "non_tensor_batch", None) or {}
        extra_infos = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get("extra_info", None)),
            template_batch_size,
        )

        dataset_indices: list[int] = []
        matched_row_indices: list[int] = []
        match_statuses: list[str] = []
        sampled_match_statuses: list[str] = []
        samples: list[dict[str, Any]] = []
        sample_type_values: list[str] = []
        sample_lora_variants: list[str] = []
        sample_mount_lora: list[bool] = []
        sample_target_values: list[str] = []
        for sample_position, row_index in enumerate(selected_row_indices):
            sample_type = sample_types[sample_position] if sample_position < len(sample_types) else "changed"
            extra_info = extra_infos[row_index] if isinstance(extra_infos[row_index], dict) else {}
            knowledge_id = self._normalize_optional_id(extra_info.get("knowledge_id"))
            sample_hash = str(extra_info.get("sample_hash", "")).strip()
            meta_query = str(extra_info.get("meta_query", "")).strip()
            dataset_index, match_status = self._select_luffy_teacher_dataset_index_with_status(
                knowledge_id=knowledge_id,
                meta_query=meta_query,
                sample_hash=sample_hash,
            )
            sampled_match_statuses.append(match_status)
            if dataset_index is None:
                continue
            dataset_indices.append(dataset_index)
            matched_row_indices.append(row_index)
            match_statuses.append(match_status)
            if sample_type == "changed":
                sample = self.luffy_teacher_dataset[int(dataset_index)]
                sample_lora_variant = KNOWLEDGE_LORA_VARIANT
                mount_lora = True
                target_text = ""
            elif sample_type in {"no_change", "random_change"}:
                target_text = self._select_luffy_negative_response(sample_type)
                sample = self._build_luffy_negative_teacher_sample(
                    dataset_index=int(dataset_index),
                    sample_type=sample_type,
                    response_text=target_text,
                )
                sample_lora_variant = NO_OP_LORA_VARIANT if sample_type == "no_change" else RANDOM_LORA_VARIANT
                mount_lora = sample_type == "random_change"
            else:
                raise ValueError(f"Unsupported LUFFY teacher sample_type={sample_type!r}")
            samples.append(sample)
            sample_type_values.append(sample_type)
            sample_lora_variants.append(sample_lora_variant)
            sample_mount_lora.append(mount_lora)
            sample_target_values.append(target_text)

        sampling_stats = self._build_luffy_teacher_sampling_stats(
            sample_size=sample_size,
            match_statuses=sampled_match_statuses,
            matched_count=len(dataset_indices),
        )
        sampling_stats["template_batch_size"] = int(template_batch_size)
        sampling_stats["base_sampled_rows"] = self._resolve_luffy_base_teacher_sample_size(
            template_batch_size=template_batch_size
        )
        sampling_stats["sample_with_replacement"] = float(sample_size > template_batch_size)
        sampling_stats["warmup_active"] = float(self._luffy_teacher_warmup_active())
        sampling_stats["warmup_sample_with_replacement"] = float(self._luffy_teacher_warmup_allows_replacement())
        sampling_stats["warmup_teacher_multiplier"] = float(
            self.luffy_config.get("warmup_teacher_multiplier", 1.0)
        )
        sampling_stats["warmup_quota_changed"] = int((quota_counts or {}).get("changed", 0))
        sampling_stats["warmup_quota_no_change"] = int((quota_counts or {}).get("no_change", 0))
        sampling_stats["warmup_quota_random_change"] = int((quota_counts or {}).get("random_change", 0))
        sampling_stats["actual_changed"] = int(sum(1 for item in sample_type_values if item == "changed"))
        sampling_stats["actual_no_change"] = int(sum(1 for item in sample_type_values if item == "no_change"))
        sampling_stats["actual_random_change"] = int(sum(1 for item in sample_type_values if item == "random_change"))
        batch.meta_info["luffy_teacher_sampling_stats"] = sampling_stats

        if not samples:
            return None

        collated = self._collate_luffy_teacher_samples(samples)
        return self._convert_luffy_teacher_collated_batch(
            collated,
            template_batch=batch,
            selected_row_indices=matched_row_indices,
            match_statuses=match_statuses,
            sample_types=sample_type_values,
            sample_lora_variants=sample_lora_variants,
            sample_mount_lora=sample_mount_lora,
            sample_targets=sample_target_values,
        )

    def _convert_luffy_teacher_collated_batch(
        self,
        collated_batch: dict[str, Any],
        *,
        template_batch: DataProto,
        selected_row_indices: list[int],
        match_statuses: list[str] | None = None,
        sample_types: list[str] | None = None,
        sample_lora_variants: list[str] | None = None,
        sample_mount_lora: list[bool] | None = None,
        sample_targets: list[str] | None = None,
        teacher_dataset_indices: list[int] | None = None,
    ) -> DataProto | None:
        input_ids = collated_batch.get("input_ids", None)
        attention_mask = collated_batch.get("attention_mask", None)
        loss_mask = collated_batch.get("loss_mask", None)
        target_probs = collated_batch.get("target_probs", None)
        if input_ids is None or attention_mask is None or loss_mask is None:
            return None

        template_tensors = template_batch.batch
        target_sequence_length = int(template_tensors["input_ids"].shape[-1])
        target_response_length = int(template_tensors["responses"].shape[-1])
        if target_sequence_length <= 0 or target_response_length <= 0:
            return None

        batch_size = int(input_ids.shape[0])
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        valid_lengths = attention_mask.long().sum(dim=-1)
        response_lengths = loss_mask.long().sum(dim=-1)
        teacher_reward_values = self._resolve_luffy_teacher_scalar_rewards(
            collated_batch=collated_batch,
            batch_size=batch_size,
        )

        teacher_input_ids = torch.full(
            (batch_size, target_sequence_length),
            pad_token_id,
            dtype=template_tensors["input_ids"].dtype,
        )
        teacher_attention_mask = torch.zeros(
            (batch_size, target_sequence_length),
            dtype=template_tensors["attention_mask"].dtype,
        )
        teacher_responses = torch.full(
            (batch_size, target_response_length),
            pad_token_id,
            dtype=template_tensors["responses"].dtype,
        )
        teacher_response_mask = torch.zeros(
            (batch_size, target_response_length),
            dtype=template_tensors["response_mask"].dtype,
        )
        reward_dtype = template_tensors.get("token_level_rewards", torch.zeros((), dtype=torch.float32)).dtype
        teacher_token_level_rewards = torch.zeros(
            (batch_size, target_response_length),
            dtype=reward_dtype,
        )
        teacher_token_level_scores = torch.zeros(
            (batch_size, target_response_length),
            dtype=reward_dtype,
        )
        teacher_target_probs = None
        if target_probs is not None:
            teacher_target_probs = torch.zeros(
                (batch_size, target_response_length),
                dtype=torch.float32,
            )

        prompt_budget = max(0, target_sequence_length - target_response_length)
        for row_index in range(batch_size):
            valid_length = int(valid_lengths[row_index].item())
            response_length = int(response_lengths[row_index].item())
            if response_length <= 0:
                continue
            valid_tokens = input_ids[row_index, :valid_length]
            prompt_tokens = valid_tokens[:-response_length]
            response_tokens = valid_tokens[-response_length:]

            if prompt_budget > 0 and prompt_tokens.shape[0] > prompt_budget:
                prompt_tokens = prompt_tokens[-prompt_budget:]
            elif prompt_budget == 0:
                prompt_tokens = prompt_tokens[:0]

            response_tokens = response_tokens[:target_response_length]
            current_response_length = int(response_tokens.shape[0])
            if current_response_length <= 0:
                continue
            teacher_responses[row_index, :current_response_length] = response_tokens.to(teacher_responses.dtype)
            teacher_response_mask[row_index, :current_response_length] = 1
            reward_value = float(teacher_reward_values[row_index])
            teacher_token_level_rewards[row_index, current_response_length - 1] = reward_value
            teacher_token_level_scores[row_index, current_response_length - 1] = reward_value
            if teacher_target_probs is not None:
                valid_target_probs = target_probs[row_index, :valid_length]
                response_target_probs = valid_target_probs[-response_length:][:target_response_length]
                teacher_target_probs[row_index, :current_response_length] = response_target_probs[:current_response_length].to(dtype=teacher_target_probs.dtype)

            prompt_length = int(prompt_tokens.shape[0])
            if prompt_budget > 0 and prompt_length > 0:
                prompt_start = max(0, prompt_budget - prompt_length)
                teacher_input_ids[row_index, prompt_start:prompt_budget] = prompt_tokens.to(teacher_input_ids.dtype)
                teacher_attention_mask[row_index, prompt_start:prompt_budget] = 1

            response_start = prompt_budget
            response_end = min(target_sequence_length, response_start + current_response_length)
            if response_end > response_start:
                response_fill_length = response_end - response_start
                teacher_input_ids[row_index, response_start:response_end] = response_tokens[
                    :response_fill_length
                ].to(teacher_input_ids.dtype)
                teacher_attention_mask[row_index, response_start:response_end] = 1

        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
        prefix_mask = teacher_response_mask.to(dtype=torch.bool)
        teacher_tensors = {
            "responses": teacher_responses,
            "response_mask": teacher_response_mask,
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "prefix_mask": prefix_mask,
            "token_level_scores": teacher_token_level_scores,
            "token_level_rewards": teacher_token_level_rewards,
        }
        if "old_log_probs" in template_tensors.keys():
            teacher_tensors["old_log_probs"] = torch.zeros(
                (batch_size, target_response_length),
                dtype=template_tensors["old_log_probs"].dtype,
            )
        if "ref_log_prob" in template_tensors.keys():
            teacher_tensors["ref_log_prob"] = torch.zeros(
                (batch_size, target_response_length),
                dtype=template_tensors["ref_log_prob"].dtype,
            )
        if "rollout_log_probs" in template_tensors.keys():
            teacher_tensors["rollout_log_probs"] = torch.zeros(
                (batch_size, target_response_length),
                dtype=template_tensors["rollout_log_probs"].dtype,
            )
        if "rollout_is_weights" in template_tensors.keys():
            teacher_tensors["rollout_is_weights"] = torch.ones(
                (batch_size, target_response_length),
                dtype=template_tensors["rollout_is_weights"].dtype,
            )
        if teacher_target_probs is not None:
            teacher_tensors["target_probs"] = teacher_target_probs

        template_batch_size = self._infer_batch_size(template_batch) or len(selected_row_indices)
        teacher_non_tensors: dict[str, np.ndarray] = {}
        for key, raw_values in (template_batch.non_tensor_batch or {}).items():
            expanded_values = self._pad_optional_values(self._expand_batch_field(raw_values), template_batch_size)
            sampled_values = [expanded_values[row_index] for row_index in selected_row_indices]
            teacher_non_tensors[key] = self._as_object_vector(sampled_values)

        scheduled_mode = teacher_dataset_indices is not None
        if scheduled_mode:
            if len(teacher_dataset_indices) != batch_size:
                raise ValueError("Scheduled teacher dataset index count does not match the collated batch size.")
            metadata_keys = (
                getattr(self.luffy_teacher_dataset, "knowledge_id_key", "knowledge_id"),
                "title",
                "category",
                "subcategory",
                "context",
                "question",
                "answer",
                "meta_query",
                "sample_hash",
                "reader_id",
                "update_type",
                "reader_target",
                "reader_target_kind",
            )
            for key in metadata_keys:
                if key in collated_batch:
                    teacher_non_tensors[key] = self._as_object_vector(
                        self._pad_optional_values(self._expand_batch_field(collated_batch[key]), batch_size)
                    )

        match_status_values = self._pad_optional_values(match_statuses or [], batch_size, fill_value="unknown")
        teacher_non_tensors["luffy_teacher_match_status"] = self._as_object_vector(match_status_values)
        sample_type_values = self._pad_optional_values(sample_types or [], batch_size, fill_value="changed")
        sample_lora_variant_values = self._pad_optional_values(
            sample_lora_variants or [],
            batch_size,
            fill_value=KNOWLEDGE_LORA_VARIANT,
        )
        sample_mount_lora_values = self._pad_optional_values(sample_mount_lora or [], batch_size, fill_value=True)
        sample_target_values = self._pad_optional_values(sample_targets or [], batch_size, fill_value="")
        teacher_non_tensors["luffy_teacher_sample_type"] = self._as_object_vector(sample_type_values)
        teacher_non_tensors["luffy_teacher_lora_variant"] = self._as_object_vector(sample_lora_variant_values)
        teacher_non_tensors["luffy_teacher_mount_lora"] = self._as_object_vector(sample_mount_lora_values)
        teacher_non_tensors["lora_variant"] = self._as_object_vector(sample_lora_variant_values)
        teacher_non_tensors[self.ephemeral_lora_variant_field] = self._as_object_vector(sample_lora_variant_values)
        if scheduled_mode:
            teacher_non_tensors[self.ephemeral_lora_path_field] = self._as_object_vector([None] * batch_size)
            teacher_non_tensors[self.ephemeral_lora_variants_field] = self._as_object_vector([None] * batch_size)
            scheduled_requests = [
                {
                    "knowledge_id": str(collated_batch[
                        getattr(self.luffy_teacher_dataset, "knowledge_id_key", "knowledge_id")
                    ][row_index]),
                    "lora_variant": str(sample_lora_variant_values[row_index]),
                }
                if bool(sample_mount_lora_values[row_index])
                else None
                for row_index in range(batch_size)
            ]
            teacher_non_tensors[self.ephemeral_lora_request_field] = self._as_object_vector(scheduled_requests)

        sampled_extra_infos: list[dict[str, Any]] = []
        sampled_meta_queries = collated_batch.get("meta_query", [""] * batch_size)
        sampled_knowledge_ids = collated_batch.get(
            getattr(self.luffy_teacher_dataset, "knowledge_id_key", "knowledge_id"),
            [""] * batch_size,
        )
        for row_index, sampled_row_index in enumerate(selected_row_indices):
            base_extra_info: dict[str, Any] = {}
            knowledge_id = str(sampled_knowledge_ids[row_index])
            if scheduled_mode:
                source_row = self._luffy_per_type_source_by_knowledge.get(knowledge_id)
                if source_row is None and self.luffy_per_type_schedule_strict:
                    raise KeyError(f"Scheduled teacher row has no source metadata for knowledge_id={knowledge_id}")
                if source_row is not None:
                    base_extra_info = dict(source_row)
            elif "extra_info" in teacher_non_tensors and row_index < len(teacher_non_tensors["extra_info"]):
                raw_extra_info = teacher_non_tensors["extra_info"][row_index]
                if isinstance(raw_extra_info, Mapping):
                    base_extra_info = dict(raw_extra_info)
            base_extra_info["is_luffy_teacher"] = True
            base_extra_info["knowledge_id"] = knowledge_id
            base_extra_info["meta_query"] = str(sampled_meta_queries[row_index])
            base_extra_info["teacher_reward"] = float(teacher_reward_values[row_index])
            base_extra_info["source_row_index"] = int(sampled_row_index)
            base_extra_info["teacher_match_status"] = str(match_status_values[row_index])
            base_extra_info["luffy_sample_type"] = str(sample_type_values[row_index])
            base_extra_info["luffy_negative_target"] = str(sample_target_values[row_index] or "")
            base_extra_info["mount_ephemeral_lora"] = bool(sample_mount_lora_values[row_index])
            base_extra_info["lora_variant"] = str(sample_lora_variant_values[row_index])
            base_extra_info[self.ephemeral_lora_variant_field] = str(sample_lora_variant_values[row_index])
            if scheduled_mode:
                base_extra_info["teacher_dataset_index"] = int(teacher_dataset_indices[row_index])
                for metadata_key in (
                    "title",
                    "category",
                    "subcategory",
                    "context",
                    "question",
                    "answer",
                    "sample_hash",
                    "reader_id",
                    "update_type",
                    "reader_target",
                    "reader_target_kind",
                ):
                    values = collated_batch.get(metadata_key)
                    if values is not None:
                        base_extra_info[metadata_key] = str(values[row_index])
                if bool(sample_mount_lora_values[row_index]):
                    base_extra_info[self.ephemeral_lora_request_field] = {
                        "knowledge_id": knowledge_id,
                        "lora_variant": str(sample_lora_variant_values[row_index]),
                    }
                else:
                    base_extra_info.pop(self.ephemeral_lora_request_field, None)
                    base_extra_info.pop(self.ephemeral_lora_path_field, None)
            sampled_extra_infos.append(base_extra_info)
        if sampled_extra_infos:
            teacher_non_tensors["extra_info"] = self._as_object_vector(sampled_extra_infos)
        if scheduled_mode:
            teacher_non_tensors["uid"] = self._as_object_vector(teacher_dataset_indices)
        elif "uid" not in teacher_non_tensors:
            teacher_non_tensors["uid"] = self._as_object_vector(selected_row_indices)

        teacher_proto = DataProto.from_dict(
            tensors=teacher_tensors,
            non_tensors=teacher_non_tensors,
            meta_info=dict(template_batch.meta_info),
        )
        if self.enable_ephemeral_lora and self.use_ephemeral_lora_for_loss:
            teacher_plan_items = self._build_ephemeral_lora_execution_plan(teacher_proto)
            self._materialize_ephemeral_lora_requests_for_plan_items(teacher_plan_items)
            self._annotate_gen_batch_with_execution_plan(
                gen_batch=teacher_proto,
                execution_plan=teacher_plan_items,
            )
            teacher_paths = self._expand_batch_field(
                (teacher_proto.non_tensor_batch or {}).get(self.ephemeral_lora_path_field, None)
            )
            teacher_paths = self._pad_optional_values(teacher_paths, batch_size)
            teacher_proto.non_tensor_batch["ephemeral_lora_loss_path"] = self._as_object_vector(teacher_paths)
            for plan_item in teacher_plan_items:
                self._delete_ephemeral_lora_path_after_use(
                    lora_path=plan_item.get("lora_path"),
                    knowledge_id=plan_item.get("knowledge_id"),
                    lora_variant=plan_item.get("lora_variant"),
                )
            teacher_proto.meta_info = dict(teacher_proto.meta_info)
            teacher_proto.meta_info["luffy_teacher_lora_loss_paths"] = int(
                sum(1 for path in teacher_paths if self._normalize_optional_path(path) is not None)
            )
            teacher_proto.meta_info["luffy_teacher_lora_loss_groups"] = int(
                len(
                    {
                        self._normalize_optional_path(path)
                        for path in teacher_paths
                        if self._normalize_optional_path(path) is not None
                    }
                )
            )
        return teacher_proto

    def _build_luffy_concat_padding_tensor(
        self,
        *,
        key: str,
        reference_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        fill_value = 1 if key in {"rollout_is_weights", "target_probs"} else 0
        return reference_tensor.new_full((batch_size, *reference_tensor.shape[1:]), fill_value=fill_value)

    def _align_luffy_dataproto_for_concat(self, left: DataProto, right: DataProto) -> tuple[DataProto, DataProto]:
        left_keys = set(left.batch.keys())
        right_keys = set(right.batch.keys())
        left_non_tensor_batch = getattr(left, "non_tensor_batch", None) or {}
        right_non_tensor_batch = getattr(right, "non_tensor_batch", None) or {}
        left_non_keys = set(left_non_tensor_batch.keys())
        right_non_keys = set(right_non_tensor_batch.keys())

        all_keys = sorted(left_keys | right_keys)
        left_batch_size = len(left)
        right_batch_size = len(right)
        left_tensors: dict[str, torch.Tensor] = {}
        right_tensors: dict[str, torch.Tensor] = {}

        for key in all_keys:
            left_tensor = left.batch[key] if key in left_keys else None
            right_tensor = right.batch[key] if key in right_keys else None
            if left_tensor is None:
                left_tensor = self._build_luffy_concat_padding_tensor(
                    key=key,
                    reference_tensor=right_tensor,
                    batch_size=left_batch_size,
                )
            if right_tensor is None:
                right_tensor = self._build_luffy_concat_padding_tensor(
                    key=key,
                    reference_tensor=left_tensor,
                    batch_size=right_batch_size,
                )
            left_tensors[key] = left_tensor
            right_tensors[key] = right_tensor

        all_non_keys = sorted(left_non_keys | right_non_keys)
        left_non_tensors: dict[str, np.ndarray] = {}
        right_non_tensors: dict[str, np.ndarray] = {}
        for key in all_non_keys:
            if key in left_non_tensor_batch:
                left_values = self._pad_optional_values(self._expand_batch_field(left_non_tensor_batch[key]), left_batch_size)
            else:
                left_values = [None] * left_batch_size
            if key in right_non_tensor_batch:
                right_values = self._pad_optional_values(self._expand_batch_field(right_non_tensor_batch[key]), right_batch_size)
            else:
                right_values = [None] * right_batch_size
            left_non_tensors[key] = self._as_object_vector(left_values)
            right_non_tensors[key] = self._as_object_vector(right_values)

        return (
            DataProto.from_dict(tensors=left_tensors, non_tensors=left_non_tensors, meta_info=dict(left.meta_info)),
            DataProto.from_dict(tensors=right_tensors, non_tensors=right_non_tensors, meta_info=dict(right.meta_info)),
        )

    def _ensure_luffy_prefix_mask(self, batch: DataProto) -> DataProto:
        if "prefix_mask" in batch.batch.keys():
            return batch
        zero_prefix_mask = DataProto.from_dict(
            tensors={
                "prefix_mask": torch.zeros_like(batch.batch["response_mask"], dtype=torch.bool),
            },
            meta_info=dict(batch.meta_info),
        )
        return batch.union(zero_prefix_mask)

    def _drop_luffy_teacher_source_rows(self, batch: DataProto, teacher_batch: DataProto) -> DataProto:
        if len(batch) <= 0 or len(teacher_batch) <= 0:
            return batch

        source_row_indices: set[int] = set()
        extra_infos = self._expand_batch_field((teacher_batch.non_tensor_batch or {}).get("extra_info", None))
        for extra_info in extra_infos:
            if not isinstance(extra_info, Mapping):
                continue
            try:
                source_row_index = int(extra_info.get("source_row_index"))
            except (TypeError, ValueError):
                continue
            if 0 <= source_row_index < len(batch):
                source_row_indices.add(source_row_index)
        if not source_row_indices:
            return batch

        keep_mask = np.ones(len(batch), dtype=bool)
        keep_mask[list(source_row_indices)] = False
        kept_batch = batch.select_idxs(keep_mask)
        kept_batch.meta_info = dict(kept_batch.meta_info)
        kept_batch.meta_info["luffy_teacher_replaced_on_policy_rows"] = len(source_row_indices)
        return kept_batch

    @staticmethod
    def _flatten_luffy_metric_value(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            flattened = []
            for item in value:
                item_flat = KnowledgeUpdatePPOTrainer._flatten_luffy_metric_value(item)
                if isinstance(item_flat, list):
                    flattened.extend(item_flat)
                elif isinstance(item_flat, (int, float, bool, np.integer, np.floating, np.bool_)):
                    flattened.append(float(item_flat))
            return flattened
        if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            return [float(value)]
        return value

    def _normalize_luffy_actor_metrics(self, actor_output: DataProto) -> None:
        metrics = actor_output.meta_info.get("metrics") if actor_output is not None else None
        if not isinstance(metrics, dict):
            return
        for key, value in list(metrics.items()):
            normalized = self._flatten_luffy_metric_value(value)
            if isinstance(normalized, list) and normalized:
                metrics[key] = normalized
            elif isinstance(normalized, list):
                metrics[key] = [0.0]
            else:
                metrics.pop(key, None)

    def _finalize_luffy_category_loss_metrics(self, actor_output: DataProto) -> None:
        metrics = actor_output.meta_info.get("metrics") if actor_output is not None else None
        if not isinstance(metrics, dict):
            return
        all_teacher_tokens = sum(
            self._flatten_luffy_metric_value(metrics.get("actor/sft_category_all_teacher_tokens", [])) or []
        )
        for category in ("knowledge", "behavior", "no_op", "random"):
            nll_values = self._flatten_luffy_metric_value(
                metrics.get(f"actor/sft_category_nll_sum/{category}", [])
            )
            token_values = self._flatten_luffy_metric_value(
                metrics.get(f"actor/sft_category_token_count/{category}", [])
            )
            weighted_values = self._flatten_luffy_metric_value(
                metrics.get(f"actor/sft_category_weighted_nll_sum/{category}", [])
            )
            weight_values = self._flatten_luffy_metric_value(
                metrics.get(f"actor/sft_category_weight/{category}", [])
            )
            nll_sum = sum(nll_values) if isinstance(nll_values, list) else 0.0
            token_count = sum(token_values) if isinstance(token_values, list) else 0.0
            weighted_sum = sum(weighted_values) if isinstance(weighted_values, list) else 0.0
            mean_weight = (
                float(sum(weight_values) / len(weight_values))
                if isinstance(weight_values, list) and weight_values
                else 0.0
            )
            metrics[f"actor/sft_category_loss_mean/{category}"] = [
                float(nll_sum / max(token_count, 1.0))
            ]
            metrics[f"actor/sft_category_token_count_total/{category}"] = [float(token_count)]
            metrics[f"actor/sft_category_weight_mean/{category}"] = [mean_weight]
            metrics[f"actor/sft_category_weighted_contribution/{category}"] = [
                float(weighted_sum / max(all_teacher_tokens, 1.0))
            ]

    def _collect_luffy_mixed_batch_metrics(self, batch: DataProto) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if "prefix_mask" not in batch.batch.keys() or "response_mask" not in batch.batch.keys():
            return metrics

        prefix_mask = batch.batch["prefix_mask"].bool()
        response_mask = batch.batch["response_mask"].bool()
        if prefix_mask.ndim < 2 or response_mask.ndim < 2:
            return metrics

        total_rows = int(prefix_mask.shape[0])
        teacher_row_mask = prefix_mask.any(dim=-1)
        teacher_rows = int(teacher_row_mask.long().sum().item())
        on_policy_rows = max(0, total_rows - teacher_rows)
        teacher_tokens = int((prefix_mask & response_mask).long().sum().item())
        total_tokens = int(response_mask.long().sum().item())
        on_policy_tokens = max(0, total_tokens - teacher_tokens)

        metrics["knowledge_update/luffy/on_policy_rows"] = float(on_policy_rows)
        metrics["knowledge_update/luffy/teacher_rows"] = float(teacher_rows)
        metrics["knowledge_update/luffy/teacher_row_fraction"] = float(teacher_rows / max(total_rows, 1))
        metrics["knowledge_update/luffy/on_policy_tokens"] = float(on_policy_tokens)
        metrics["knowledge_update/luffy/teacher_tokens"] = float(teacher_tokens)
        metrics["knowledge_update/luffy/teacher_token_fraction"] = float(teacher_tokens / max(total_tokens, 1))

        sampling_stats = batch.meta_info.get("luffy_teacher_sampling_stats", {})
        if isinstance(sampling_stats, Mapping):
            for key in (
                "sampled_rows",
                "base_sampled_rows",
                "template_batch_size",
                "matched_rows",
                "miss_rows",
                "fallback_rows",
                "miss_fraction",
                "fallback_fraction",
                "sample_with_replacement",
                "warmup_active",
                "warmup_sample_with_replacement",
                "warmup_teacher_multiplier",
                "per_type_schedule_active",
                "scheduled_knowledge_changed",
                "scheduled_behavior_changed",
                "scheduled_no_change",
                "scheduled_random_change",
                "scheduled_global_step",
            ):
                if key in sampling_stats:
                    metrics[f"knowledge_update/luffy/{key}"] = float(sampling_stats[key])
            status_counts = sampling_stats.get("match_status_counts", {})
            if isinstance(status_counts, Mapping):
                sampled_rows = float(sampling_stats.get("sampled_rows", sum(float(value) for value in status_counts.values())))
                for raw_status, raw_count in status_counts.items():
                    status = str(raw_status or "unknown")
                    count = float(raw_count)
                    metrics[f"knowledge_update/luffy/sampling_status/{status}"] = count
                    metrics[f"knowledge_update/luffy/sampling_status/{status}_fraction"] = count / max(sampled_rows, 1.0)

        non_tensor_batch = getattr(batch, "non_tensor_batch", None) or {}
        match_status_values = non_tensor_batch.get("luffy_teacher_match_status", None)
        if match_status_values is not None:
            for raw_status in self._expand_batch_field(match_status_values):
                if raw_status in (None, "", "None"):
                    continue
                status = str(raw_status)
                metric_key = f"knowledge_update/luffy/match_status/{status}"
                metrics[metric_key] = metrics.get(metric_key, 0.0) + 1.0
            status_total = sum(
                value
                for key, value in metrics.items()
                if key.startswith("knowledge_update/luffy/match_status/")
            )
            for key, value in list(metrics.items()):
                if key.startswith("knowledge_update/luffy/match_status/"):
                    metrics[f"{key}_fraction"] = float(value / max(status_total, 1.0))
        return metrics

    @contextmanager
    def _patch_compute_advantage(self):
        original_compute_advantage = ray_trainer_module.compute_advantage

        def wrapped_compute_advantage(batch, *args, **kwargs):
            if not self.enable_luffy_teacher_loss or self.luffy_teacher_dataset is None:
                return original_compute_advantage(batch, *args, **kwargs)

            mixed_batch = self._ensure_luffy_prefix_mask(batch)
            teacher_batch = self._sample_luffy_teacher_batch(mixed_batch)
            if teacher_batch is not None:
                warmup_teacher_only = bool(self.luffy_config.get("warmup_teacher_only", False))
                if warmup_teacher_only and self._luffy_teacher_warmup_active():
                    teacher_batch.meta_info = dict(teacher_batch.meta_info)
                    teacher_batch.meta_info["luffy_teacher_mixed"] = True
                    teacher_batch.meta_info["luffy_teacher_rows"] = len(teacher_batch)
                    teacher_batch.meta_info["luffy_teacher_only_warmup"] = True
                    teacher_batch.meta_info["luffy_teacher_dropped_on_policy_rows"] = len(mixed_batch)
                    sampling_stats = mixed_batch.meta_info.get("luffy_teacher_sampling_stats")
                    if isinstance(sampling_stats, Mapping):
                        teacher_batch.meta_info["luffy_teacher_sampling_stats"] = dict(sampling_stats)
                    return original_compute_advantage(teacher_batch, *args, **kwargs)
                if bool(self.luffy_config.get("replace_sampled_on_policy", True)):
                    mixed_batch = self._drop_luffy_teacher_source_rows(mixed_batch, teacher_batch)
                mixed_batch, teacher_batch = self._align_luffy_dataproto_for_concat(mixed_batch, teacher_batch)
                mixed_batch = DataProto.concat([mixed_batch, teacher_batch])
                mixed_batch.meta_info["luffy_teacher_mixed"] = True
                mixed_batch.meta_info["luffy_teacher_rows"] = len(teacher_batch)
            else:
                mixed_batch.meta_info["luffy_teacher_mixed"] = False
                mixed_batch.meta_info["luffy_teacher_rows"] = 0
            return original_compute_advantage(mixed_batch, *args, **kwargs)

        ray_trainer_module.compute_advantage = wrapped_compute_advantage
        try:
            yield
        finally:
            ray_trainer_module.compute_advantage = original_compute_advantage

    def _update_actor(self, batch: DataProto) -> DataProto:
        luffy_metrics: dict[str, float] = {}
        if self.enable_luffy_teacher_loss:
            batch = self._ensure_luffy_prefix_mask(batch)
            batch.meta_info = dict(batch.meta_info)
            batch.meta_info["luffy_global_step"] = int(getattr(self, "global_steps", 0))
            luffy_metrics = self._collect_luffy_mixed_batch_metrics(batch)
        try:
            actor_output = super()._update_actor(batch)
        finally:
            self._flush_pending_ephemeral_lora_deletes()
        if self.enable_luffy_teacher_loss:
            self._normalize_luffy_actor_metrics(actor_output)
            self._finalize_luffy_category_loss_metrics(actor_output)
        if luffy_metrics:
            actor_output.meta_info.setdefault("metrics", {}).update(luffy_metrics)
        return actor_output

    def fit(self):
        with self._patch_reward_extraction():
            with self._patch_rollout_generation():
                with self._patch_compute_advantage():
                    return super().fit()

    @contextmanager
    def _patch_reward_extraction(self):
        original_extract_reward = ray_trainer_module.extract_reward

        def wrapped_extract_reward(batch, *args, **kwargs):
            if self._should_skip_on_policy_reward_for_luffy_warmup(batch=batch):
                reward_tensor, reward_extra_infos_dict = self._build_skipped_luffy_warmup_reward(batch=batch)
            else:
                reward_tensor, reward_extra_infos_dict = original_extract_reward(batch, *args, **kwargs)
            self._augment_reward_extra_infos_with_lora_metadata(
                batch=batch,
                reward_extra_infos_dict=reward_extra_infos_dict,
            )
            self._update_no_op_reward_gate(batch=batch, reward_extra_infos_dict=reward_extra_infos_dict)
            self._record_positive_process_reward_samples(batch=batch, reward_extra_infos_dict=reward_extra_infos_dict)
            self._filter_reward_extra_infos_for_logging(reward_extra_infos_dict=reward_extra_infos_dict)
            return reward_tensor, reward_extra_infos_dict

        ray_trainer_module.extract_reward = wrapped_extract_reward
        try:
            yield
        finally:
            ray_trainer_module.extract_reward = original_extract_reward

    def _should_skip_on_policy_reward_for_luffy_warmup(self, *, batch) -> bool:
        if not self.enable_luffy_teacher_loss or self.luffy_teacher_dataset is None:
            return False
        if not bool(self.luffy_config.get("warmup_teacher_only", False)):
            return False
        if not bool(self.luffy_config.get("warmup_skip_on_policy_reward", True)):
            return False
        if not self._luffy_teacher_warmup_active():
            return False
        batch_tensors = getattr(batch, "batch", None)
        if batch_tensors is None:
            return False
        return "responses" in batch_tensors.keys()

    def _should_skip_rollout_lora_for_luffy_warmup(self) -> bool:
        if not self.enable_luffy_teacher_loss or self.luffy_teacher_dataset is None:
            return False
        if not bool(self.luffy_config.get("warmup_teacher_only", False)):
            return False
        if not bool(self.luffy_config.get("warmup_skip_rollout_lora", True)):
            return False
        return self._luffy_teacher_warmup_active()

    def _build_luffy_teacher_only_warmup_rollout_stub(self, gen_batch) -> DataProto:
        batch_size = self._infer_batch_size(gen_batch)
        if batch_size is None:
            raise ValueError("Cannot infer batch size for teacher-only warmup rollout stub.")
        batch_size = int(batch_size)
        response_length = max(1, int(self.config.data.get("max_response_length", 1)))
        prompt_budget = max(1, int(self.config.data.get("max_prompt_length", 1)))
        sequence_length = prompt_budget + response_length
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids = torch.full((batch_size, sequence_length), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long)
        responses = torch.full((batch_size, response_length), int(pad_token_id), dtype=torch.long)
        response_mask = torch.zeros((batch_size, response_length), dtype=torch.long)
        rollout_log_probs = torch.zeros((batch_size, response_length), dtype=torch.float32)

        non_tensor_batch = getattr(gen_batch, "non_tensor_batch", None) or {}
        raw_prompts = self._expand_batch_field(non_tensor_batch.get("raw_prompt", None))
        if not raw_prompts:
            raw_prompts = self._expand_batch_field(non_tensor_batch.get("prompt", None))
        raw_prompts = self._pad_optional_values(raw_prompts, batch_size, fill_value=None)
        for row_index, raw_prompt in enumerate(raw_prompts[:batch_size]):
            if raw_prompt is None:
                continue
            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    raw_prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_ids = self.tokenizer.apply_chat_template(
                    raw_prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = str(raw_prompt)
                prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_ids = list(prompt_ids)[-prompt_budget:]
            if not prompt_ids:
                continue
            start = prompt_budget - len(prompt_ids)
            input_ids[row_index, start:prompt_budget] = torch.tensor(prompt_ids, dtype=torch.long)
            attention_mask[row_index, start:prompt_budget] = 1

        position_ids = compute_position_id_with_mask(attention_mask)
        non_tensors = {key: value for key, value in non_tensor_batch.items()}
        if "multi_modal_inputs" not in non_tensors:
            non_tensors["multi_modal_inputs"] = np.array([{} for _ in range(batch_size)], dtype=object)
        meta_info = dict(getattr(gen_batch, "meta_info", {}) or {})
        meta_info["timing"] = {"gen": 0.0, "luffy_teacher_only_rollout_stub": 1.0}
        meta_info["luffy_teacher_only_rollout_stub"] = True
        return DataProto.from_dict(
            tensors={
                "responses": responses,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "rollout_log_probs": rollout_log_probs,
            },
            non_tensors=non_tensors,
            meta_info=meta_info,
        )

    def _build_skipped_luffy_warmup_reward(self, *, batch) -> tuple[torch.Tensor, dict[str, list[float]]]:
        responses = batch.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        try:
            batch_size = self._infer_batch_size(batch)
        except Exception:
            batch_size = None
        batch_size = batch_size or int(responses.shape[0])
        zeros = [0.0] * batch_size
        ones = [1.0] * batch_size
        return reward_tensor, {
            "score": list(zeros),
            "process_reward": list(zeros),
            "knowledge_content_gate_passed": list(zeros),
            "judge_used": list(zeros),
            "judge_failed": list(zeros),
            "judge_fallback_used": list(zeros),
            "luffy_warmup_reward_skipped": ones,
        }

    def _resolve_process_reward_positive_samples_path(self) -> str | None:
        if self.process_reward_positive_sample_limit <= 0:
            return None
        trainer_default_local_dir = getattr(getattr(self.config, "trainer", None), "default_local_dir", None)
        output_dir = self._normalize_optional_path(trainer_default_local_dir)
        if output_dir is None:
            return None
        return os.path.abspath(os.path.join(output_dir, "process_reward_positive_samples.json"))

    def _load_existing_process_reward_positive_samples(self) -> None:
        path = self.process_reward_positive_samples_path
        if not path:
            return
        path_obj = Path(path)
        if not path_obj.exists():
            return
        try:
            payload = json.loads(path_obj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load existing process-reward-positive sample log from %s", path)
            return
        if not isinstance(payload, list):
            return
        trimmed_payload = payload[: self.process_reward_positive_sample_limit]
        self._process_reward_positive_samples = [entry for entry in trimmed_payload if isinstance(entry, dict)]
        self._process_reward_positive_sample_keys = {
            str(entry.get("sample_key"))
            for entry in self._process_reward_positive_samples
            if entry.get("sample_key") is not None
        }

    def _persist_process_reward_positive_samples(self) -> None:
        path = self.process_reward_positive_samples_path
        if not path:
            return
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        payload = self._process_reward_positive_samples[: self.process_reward_positive_sample_limit]
        path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_positive_process_reward_samples(self, *, batch, reward_extra_infos_dict: dict[str, list[Any]]) -> None:
        if self.process_reward_positive_sample_limit <= 0:
            return
        if len(self._process_reward_positive_samples) >= self.process_reward_positive_sample_limit:
            return

        reward_values = reward_extra_infos_dict.get("process_reward")
        if reward_values is None:
            return
        if len(reward_values) == 0:
            return

        batch_size = self._infer_batch_size(batch) or len(reward_values)
        if batch_size <= 0:
            return

        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not non_tensor_batch:
            return

        extra_infos = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get("extra_info", None)),
            batch_size,
        )
        responses = [""] * batch_size
        batch_tensordict = getattr(batch, "batch", None)
        response_tensor = batch_tensordict.get("responses", None) if batch_tensordict is not None else None
        if response_tensor is not None and self.tokenizer is not None:
            try:
                responses = self.tokenizer.batch_decode(response_tensor.detach().cpu(), skip_special_tokens=True)
            except Exception as exc:
                logger.warning("Failed to decode responses for positive process-reward logging: %s", exc)

        recorded = False
        for row_index, raw_value in enumerate(reward_values[:batch_size]):
            if len(self._process_reward_positive_samples) >= self.process_reward_positive_sample_limit:
                break
            try:
                process_reward = float(raw_value)
            except (TypeError, ValueError):
                continue
            if process_reward <= 0.0:
                continue

            extra_info = extra_infos[row_index] if isinstance(extra_infos[row_index], dict) else {}
            response_text = responses[row_index].strip() if row_index < len(responses) else ""
            sample_key = json.dumps(
                {
                    "step": int(self.global_steps),
                    "knowledge_id": extra_info.get("knowledge_id"),
                    "meta_query": extra_info.get("meta_query"),
                    "response": response_text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if sample_key in self._process_reward_positive_sample_keys:
                continue

            record = {
                "sample_key": sample_key,
                "step": int(self.global_steps),
                "process_reward": process_reward,
                "knowledge_id": extra_info.get("knowledge_id"),
                "title": extra_info.get("title"),
                "question": extra_info.get("question"),
                "answer": extra_info.get("answer"),
                "meta_query": extra_info.get("meta_query"),
                "response": response_text,
                "lora_variant": extra_info.get("lora_variant") or extra_info.get(self.ephemeral_lora_variant_field),
            }
            self._process_reward_positive_samples.append(record)
            self._process_reward_positive_sample_keys.add(sample_key)
            recorded = True

        if recorded:
            self._persist_process_reward_positive_samples()

    def _filter_reward_extra_infos_for_logging(self, *, reward_extra_infos_dict: dict[str, list[Any]]) -> None:
        if _resolve_reward_mode() != PROCESS_ONLY_REWARD_MODE:
            return
        for reward_key in PROCESS_ONLY_HIDDEN_REWARD_KEYS:
            reward_extra_infos_dict.pop(reward_key, None)

    def _update_no_op_reward_gate(self, *, batch, reward_extra_infos_dict: dict[str, list[Any]]) -> None:
        if NO_OP_LORA_VARIANT not in self.enabled_lora_variants or self._no_op_reward_gate_open:
            return

        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not non_tensor_batch:
            return
        lora_variants = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_variant_field, None)),
            self._infer_batch_size(batch) or 0,
        )
        if not lora_variants:
            return

        for reward_key in self.variant_reward_keys:
            reward_values = reward_extra_infos_dict.get(reward_key)
            if reward_values is None:
                continue
            if len(reward_values) == 0:
                continue
            for lora_variant, raw_value in zip(lora_variants, reward_values, strict=False):
                if self._normalize_lora_variant(lora_variant) != KNOWLEDGE_LORA_VARIANT:
                    continue
                try:
                    self._knowledge_reward_history[reward_key].append(float(raw_value))
                except (TypeError, ValueError):
                    continue

        if not all(self._knowledge_reward_history[reward_key] for reward_key in self.variant_reward_keys):
            return
        means = {
            reward_key: float(np.mean(list(history)))
            for reward_key, history in self._knowledge_reward_history.items()
        }
        if all(mean_value >= self.no_op_reward_gate_threshold for mean_value in means.values()):
            self._no_op_reward_gate_open = True
            self._record_ephemeral_lora_event(
                event="no_op_gate_open",
                knowledge_id=None,
                reward_threshold=self.no_op_reward_gate_threshold,
                reward_means=means,
            )

    def _augment_reward_extra_infos_with_lora_metadata(
        self,
        *,
        batch,
        reward_extra_infos_dict: dict[str, list[Any]],
    ) -> None:
        variant_values, lora_paths = self._extract_batch_lora_metadata(batch)
        if not variant_values:
            return

        reward_extra_infos_dict.setdefault("lora_variant", list(variant_values))
        reward_extra_infos_dict.setdefault(self.ephemeral_lora_variant_field, list(variant_values))
        if self.log_variant_lora_norm:
            reward_extra_infos_dict["lora_norm"] = [
                self._resolve_ephemeral_lora_norm(lora_path) for lora_path in lora_paths
            ]

    def _should_collect_custom_entropy_metrics(self) -> bool:
        return self.log_variant_metrics and self.log_variant_entropy

    def _collect_custom_training_metrics(
        self,
        *,
        batch,
        reward_extra_infos_dict: dict[str, list[Any]],
        entropy_tensor: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if not self.log_variant_metrics:
            return {}

        batch_variant_values, lora_paths = self._extract_batch_lora_metadata(batch)
        variant_values = self._to_python_list(reward_extra_infos_dict.get("lora_variant", []))
        if not variant_values:
            variant_values = batch_variant_values
        if not variant_values:
            return {}

        metrics = self._collect_variant_reward_metrics(
            variant_values=variant_values,
            reward_extra_infos_dict=reward_extra_infos_dict,
            metric_prefix="knowledge_update/train",
        )
        if self.log_variant_lora_norm:
            metrics.update(
                self._collect_lora_norm_metrics(
                    variant_values=variant_values,
                    lora_norm_values=reward_extra_infos_dict.get("lora_norm", []),
                    lora_paths=lora_paths,
                    metric_prefix="knowledge_update/train",
                )
            )
        if self.log_variant_entropy and entropy_tensor is not None:
            metrics.update(
                self._collect_variant_entropy_metrics(
                    batch=batch,
                    variant_values=variant_values,
                    entropy_tensor=entropy_tensor,
                    metric_prefix="knowledge_update/train",
                )
            )
        return metrics

    def _collect_custom_validation_metrics(
        self,
        *,
        data_sources,
        sample_uids,
        reward_extra_infos_dict: dict[str, list[Any]],
        sample_turns,
    ) -> dict[str, float]:
        del data_sources, sample_uids, sample_turns
        if not self.log_variant_metrics:
            return {}

        variant_values = self._to_python_list(reward_extra_infos_dict.get("lora_variant", []))
        if not variant_values:
            return {}

        return self._collect_variant_reward_metrics(
            variant_values=variant_values,
            reward_extra_infos_dict=reward_extra_infos_dict,
            metric_prefix="val-aux/variant",
        )

    def _collect_variant_reward_metrics(
        self,
        *,
        variant_values: list[Any],
        reward_extra_infos_dict: dict[str, list[Any]],
        metric_prefix: str,
    ) -> dict[str, float]:
        normalized_variants = [self._normalize_optional_lora_variant(value) for value in variant_values]
        metrics: dict[str, float] = {}
        for variant_name in self._iter_present_lora_variants(normalized_variants):
            variant_indices = [index for index, value in enumerate(normalized_variants) if value == variant_name]
            if not variant_indices:
                continue
            metrics[f"{metric_prefix}/{variant_name}/sample_count"] = float(len(variant_indices))
            for reward_key in self.variant_reward_keys:
                reward_values = self._coerce_numeric_sequence(
                    reward_extra_infos_dict.get(reward_key, []),
                    variant_indices=variant_indices,
                )
                if not reward_values:
                    continue
                metrics[f"{metric_prefix}/{variant_name}/{reward_key}/mean"] = float(np.mean(reward_values))
                metrics[f"{metric_prefix}/{variant_name}/{reward_key}/std"] = float(np.std(reward_values))
        return metrics

    def _collect_lora_norm_metrics(
        self,
        *,
        variant_values: list[Any],
        lora_norm_values: Any | None,
        lora_paths: list[str | None],
        metric_prefix: str,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        normalized_variants = [self._normalize_optional_lora_variant(value) for value in variant_values]
        sequence_norms = self._to_python_list(lora_norm_values) if lora_norm_values is not None else []

        if sequence_norms:
            variant_norms_map: dict[str, list[float]] = {}
            for variant_name in self._iter_present_lora_variants(normalized_variants):
                variant_indices = [index for index, value in enumerate(normalized_variants) if value == variant_name]
                numeric_norms = self._coerce_numeric_sequence(sequence_norms, variant_indices=variant_indices)
                if numeric_norms:
                    variant_norms_map[variant_name] = numeric_norms
        else:
            variant_norms_map = {}
            unique_norms: dict[tuple[str, str], float] = {}
            for raw_variant, lora_path in zip(variant_values, lora_paths, strict=False):
                variant_name = self._normalize_optional_lora_variant(raw_variant)
                normalized_path = self._normalize_optional_path(lora_path)
                if variant_name is None or normalized_path is None:
                    continue
                unique_norms.setdefault(
                    (variant_name, normalized_path),
                    self._resolve_ephemeral_lora_norm(normalized_path),
                )
            for variant_name in self._iter_present_lora_variants([variant for variant, _ in unique_norms]):
                numeric_norms = [
                    norm
                    for (current_variant, _), norm in unique_norms.items()
                    if current_variant == variant_name and np.isfinite(norm)
                ]
                if numeric_norms:
                    variant_norms_map[variant_name] = numeric_norms

        if not variant_norms_map:
            return metrics

        all_norms = [norm for norms in variant_norms_map.values() for norm in norms if np.isfinite(norm)]
        if all_norms:
            metrics[f"{metric_prefix}/all/lora_norm/mean"] = float(np.mean(all_norms))
            metrics[f"{metric_prefix}/all/lora_norm/std"] = float(np.std(all_norms))

        for variant_name in self._iter_present_lora_variants(list(variant_norms_map)):
            variant_norms = variant_norms_map.get(variant_name, [])
            if not variant_norms:
                continue
            metrics[f"{metric_prefix}/{variant_name}/lora_norm/mean"] = float(np.mean(variant_norms))
            metrics[f"{metric_prefix}/{variant_name}/lora_norm/std"] = float(np.std(variant_norms))
        return metrics

    def _collect_variant_entropy_metrics(
        self,
        *,
        batch,
        variant_values: list[Any],
        entropy_tensor: torch.Tensor,
        metric_prefix: str,
    ) -> dict[str, float]:
        if "response_mask" not in batch.batch:
            return {}

        response_mask = batch.batch["response_mask"]
        batch_limit = min(
            int(response_mask.shape[0]),
            int(entropy_tensor.shape[0]),
            len(variant_values),
        )
        if batch_limit <= 0:
            return {}
        response_mask = response_mask[:batch_limit]
        entropy_tensor = entropy_tensor[:batch_limit]
        variant_values = list(variant_values[:batch_limit])

        if entropy_tensor.shape != response_mask.shape:
            response_length = response_mask.shape[-1]
            if entropy_tensor.shape[-1] == response_length:
                entropy_view = entropy_tensor
            else:
                entropy_view = entropy_tensor[..., -response_length:]
                if entropy_view.shape != response_mask.shape:
                    return {}
            entropy_tensor = entropy_view

        entropy_values = (
            (entropy_tensor.float() * response_mask.float()).sum(-1)
            / response_mask.float().sum(-1).clamp_min(1.0)
        ).detach().cpu()
        entropy_list = entropy_values.tolist()
        normalized_variants = [self._normalize_optional_lora_variant(value) for value in variant_values]
        metrics: dict[str, float] = {}
        for variant_name in self._iter_present_lora_variants(normalized_variants):
            variant_indices = [index for index, value in enumerate(normalized_variants) if value == variant_name]
            variant_entropy = self._coerce_numeric_sequence(entropy_list, variant_indices=variant_indices)
            if not variant_entropy:
                continue
            metrics[f"{metric_prefix}/{variant_name}/response_entropy/mean"] = float(np.mean(variant_entropy))
            metrics[f"{metric_prefix}/{variant_name}/response_entropy/std"] = float(np.std(variant_entropy))
        return metrics

    def _extract_batch_lora_metadata(self, batch) -> tuple[list[str | None], list[str | None]]:
        batch_size = self._infer_batch_size(batch) or 0
        if batch_size <= 0:
            return [], []

        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if not non_tensor_batch:
            return [], []

        extra_infos = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get("extra_info", None)),
            batch_size,
        )
        request_items = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_request_field, None)),
            batch_size,
        )
        selected_variants = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_variant_field, None)),
            batch_size,
        )
        selected_paths = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_path_field, None)),
            batch_size,
        )

        variant_values: list[str | None] = []
        lora_paths: list[str | None] = []
        for row_index in range(batch_size):
            extra_info = extra_infos[row_index] if isinstance(extra_infos[row_index], dict) else {}
            request_item = request_items[row_index] if isinstance(request_items[row_index], dict) else {}
            raw_variant = (
                selected_variants[row_index]
                or extra_info.get(self.ephemeral_lora_variant_field)
                or extra_info.get("lora_variant")
                or request_item.get("lora_variant")
            )
            variant_name = self._normalize_optional_lora_variant(raw_variant)
            if variant_name is None and self.enable_ephemeral_lora:
                variant_name = self.default_lora_variant
            knowledge_id = self._normalize_optional_id(
                extra_info.get("knowledge_id")
                or request_item.get("knowledge_id")
                or getattr(batch, "meta_info", {}).get("knowledge_id")
            )
            resolved_path = self._normalize_optional_path(
                selected_paths[row_index]
                or extra_info.get(self.ephemeral_lora_path_field)
                or request_item.get("lora_path")
                or request_item.get(self.ephemeral_lora_path_field)
            )
            if resolved_path is None and knowledge_id is not None:
                resolved_path = self._build_ephemeral_lora_dir_from_knowledge_id(
                    knowledge_id,
                    lora_variant=variant_name,
                )
            variant_values.append(variant_name)
            lora_paths.append(resolved_path)

        return variant_values, lora_paths

    def _resolve_ephemeral_lora_norm(self, lora_path: str | None) -> float:
        normalized_path = self._normalize_optional_path(lora_path)
        if normalized_path is None:
            return float("nan")

        cached_value = self._ephemeral_lora_norm_cache.get(normalized_path)
        if cached_value is not None:
            return cached_value

        metadata_path = Path(normalized_path) / "knowledge_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                metadata = {}
            metadata_norm = metadata.get("lora_weight_l2_norm")
            if metadata_norm is not None:
                try:
                    resolved_norm = float(metadata_norm)
                except (TypeError, ValueError):
                    resolved_norm = float("nan")
                self._ephemeral_lora_norm_cache[normalized_path] = resolved_norm
                return resolved_norm

        model_path = Path(normalized_path) / "adapter_model.safetensors"
        if not model_path.exists():
            self._ephemeral_lora_norm_cache[normalized_path] = float("nan")
            return float("nan")

        try:
            from safetensors.torch import load_file as load_safetensors_file

            state_dict = load_safetensors_file(os.fspath(model_path), device="cpu")
            squared_norm = 0.0
            for tensor in state_dict.values():
                if not torch.is_tensor(tensor):
                    continue
                squared_norm += float(torch.sum(tensor.detach().float() ** 2).item())
            resolved_norm = squared_norm**0.5
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("Failed to read LoRA norm from %s: %s", normalized_path, exc)
            resolved_norm = float("nan")

        self._ephemeral_lora_norm_cache[normalized_path] = resolved_norm
        return resolved_norm

    def _cache_ephemeral_lora_norm(
        self,
        lora_path: str | None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> float:
        normalized_path = self._normalize_optional_path(lora_path)
        if normalized_path is None:
            return float("nan")

        if metadata is not None:
            metadata_norm = metadata.get("lora_weight_l2_norm")
            if metadata_norm is not None:
                try:
                    resolved_norm = float(metadata_norm)
                except (TypeError, ValueError):
                    resolved_norm = float("nan")
                self._ephemeral_lora_norm_cache[normalized_path] = resolved_norm
                return resolved_norm

        if self.log_variant_lora_norm or self.delete_ephemeral_lora_after_use:
            return self._resolve_ephemeral_lora_norm(normalized_path)
        return self._ephemeral_lora_norm_cache.get(normalized_path, float("nan"))

    def _should_delete_ephemeral_lora_path(self, lora_path: str | None) -> bool:
        if not self.delete_ephemeral_lora_after_use:
            return False
        normalized_path = self._normalize_optional_path(lora_path)
        if normalized_path is None or not self.ephemeral_lora_dir:
            return False
        root_path = Path(os.path.abspath(os.fspath(self.ephemeral_lora_dir)))
        path_obj = Path(normalized_path)
        try:
            path_obj.relative_to(root_path)
        except ValueError:
            return False
        return path_obj != root_path

    def _delete_ephemeral_lora_path_after_use(
        self,
        *,
        lora_path: str | None,
        knowledge_id: str | None,
        lora_variant: str | None = None,
    ) -> None:
        normalized_path = self._normalize_optional_path(lora_path)
        if not self._should_delete_ephemeral_lora_path(normalized_path):
            return
        self._cache_ephemeral_lora_norm(normalized_path)
        if self.use_ephemeral_lora_for_loss:
            self._pending_ephemeral_lora_deletes[normalized_path] = {
                "knowledge_id": knowledge_id,
                "lora_variant": lora_variant,
            }
            self._record_ephemeral_lora_event(
                event="defer_delete_cache",
                knowledge_id=knowledge_id,
                lora_variant=self._normalize_optional_lora_variant(lora_variant),
                lora_path=normalized_path,
            )
            return
        self._delete_ephemeral_lora_path_now(
            lora_path=normalized_path,
            knowledge_id=knowledge_id,
            lora_variant=lora_variant,
        )

    def _flush_pending_ephemeral_lora_deletes(self) -> None:
        if not self._pending_ephemeral_lora_deletes:
            return
        pending_items = list(self._pending_ephemeral_lora_deletes.items())
        self._pending_ephemeral_lora_deletes.clear()
        for normalized_path, metadata in pending_items:
            self._delete_ephemeral_lora_path_now(
                lora_path=normalized_path,
                knowledge_id=metadata.get("knowledge_id"),
                lora_variant=metadata.get("lora_variant"),
            )

    def _delete_ephemeral_lora_path_now(
        self,
        *,
        lora_path: str | None,
        knowledge_id: str | None,
        lora_variant: str | None = None,
    ) -> None:
        normalized_path = self._normalize_optional_path(lora_path)
        if not self._should_delete_ephemeral_lora_path(normalized_path):
            return

        self._cache_ephemeral_lora_norm(normalized_path)
        path_obj = Path(normalized_path)
        if not path_obj.exists():
            return

        try:
            shutil.rmtree(path_obj)
        except OSError as exc:
            logger.warning("Failed to delete ephemeral LoRA cache %s: %s", normalized_path, exc)
            self._record_ephemeral_lora_event(
                event="delete_cache_failed",
                knowledge_id=knowledge_id,
                lora_variant=self._normalize_optional_lora_variant(lora_variant),
                lora_path=normalized_path,
                error=str(exc),
            )
            return

        root_path = Path(os.path.abspath(os.fspath(self.ephemeral_lora_dir)))
        parent_path = path_obj.parent
        while parent_path != root_path:
            try:
                parent_path.relative_to(root_path)
            except ValueError:
                break
            try:
                parent_path.rmdir()
            except OSError:
                break
            parent_path = parent_path.parent

        self._record_ephemeral_lora_event(
            event="delete_cache",
            knowledge_id=knowledge_id,
            lora_variant=self._normalize_optional_lora_variant(lora_variant),
            lora_path=normalized_path,
        )

    @staticmethod
    def _coerce_numeric_sequence(values: Any, *, variant_indices: list[int]) -> list[float]:
        sequence = KnowledgeUpdatePPOTrainer._to_python_list(values)
        numeric_values: list[float] = []
        for index in variant_indices:
            if index >= len(sequence):
                continue
            try:
                numeric_value = float(sequence[index])
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                numeric_values.append(numeric_value)
        return numeric_values

    @staticmethod
    def _normalize_optional_lora_variant(value: Any) -> str | None:
        if value in (None, "", "None"):
            return None
        return KnowledgeUpdatePPOTrainer._normalize_lora_variant(value)

    @staticmethod
    def _iter_present_lora_variants(values: Sequence[str | None]) -> list[str]:
        normalized_values = [value for value in values if value is not None]
        ordered_variants = [variant_name for variant_name in SUPPORTED_LORA_VARIANTS if variant_name in normalized_values]
        remaining_variants = [
            variant_name for variant_name in dict.fromkeys(normalized_values) if variant_name not in ordered_variants
        ]
        return ordered_variants + remaining_variants

    @contextmanager
    def _patch_rollout_generation(self):
        original_generate_sequences = self.async_rollout_manager.generate_sequences

        def wrapped_generate_sequences(gen_batch, *args, **kwargs):
            if self._should_skip_rollout_lora_for_luffy_warmup():
                self._record_ephemeral_lora_event(
                    event="skip_rollout_generation_warmup",
                    knowledge_id=None,
                    reason="luffy_teacher_only_warmup",
                    batch_size=self._infer_batch_size(gen_batch),
                )
                return self._build_luffy_teacher_only_warmup_rollout_stub(gen_batch)
            execution_plan = self._build_ephemeral_lora_execution_plan(gen_batch)
            self._annotate_gen_batch_with_execution_plan(gen_batch=gen_batch, execution_plan=execution_plan)
            if execution_plan:
                self._record_ephemeral_lora_event(event="plan", knowledge_id=None, execution_plan=execution_plan)
            return self._execute_ephemeral_lora_execution_plan(
                gen_batch=gen_batch,
                execution_plan=execution_plan,
                original_generate_sequences=original_generate_sequences,
                *args,
                **kwargs,
            )

        self.async_rollout_manager.generate_sequences = wrapped_generate_sequences
        try:
            yield
        finally:
            self.async_rollout_manager.generate_sequences = original_generate_sequences

    def _ensure_rollout_correction_bypass_mode(self) -> None:
        rollout_corr = self.config.algorithm.get("rollout_correction", None)
        if rollout_corr is None:
            with open_dict(self.config.algorithm):
                self.config.algorithm.rollout_correction = OmegaConf.create({"bypass_mode": True})
            return

        with open_dict(rollout_corr):
            rollout_corr.bypass_mode = True

    def _execute_ephemeral_lora_execution_plan(
        self,
        gen_batch,
        execution_plan: list[dict[str, Any]],
        original_generate_sequences,
        *args,
        **kwargs,
    ) -> DataProto:
        execution_schedule = self._build_rollout_execution_schedule(execution_plan)
        self._validate_rollout_execution_schedule(
            execution_schedule=execution_schedule,
            batch_size=self._infer_batch_size(gen_batch),
        )
        if execution_plan:
            self._record_ephemeral_lora_event(
                event="schedule",
                knowledge_id=None,
                execution_schedule=execution_schedule,
            )
        return self._execute_rollout_execution_schedule(
            gen_batch=gen_batch,
            execution_schedule=execution_schedule,
            original_generate_sequences=original_generate_sequences,
            *args,
            **kwargs,
        )

    def _resolve_rollout_execution_strategy(self, execution_plan: list[dict[str, Any]]) -> str:
        if len(execution_plan) <= 1:
            return "single_group"

        knowledge_ids = [item["knowledge_id"] for item in execution_plan]
        if self.multi_knowledge_strategy == "serial_groups":
            if self.max_simultaneous_ephemeral_loras != 1:
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer staged-v1 only supports one active ephemeral LoRA at a time. "
                    f"Received max_simultaneous_ephemeral_loras={self.max_simultaneous_ephemeral_loras}."
                )
            return "serial_groups"
        if self.multi_knowledge_strategy == "serial_waves":
            if self.max_simultaneous_ephemeral_loras < 1:
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer requires max_simultaneous_ephemeral_loras >= 1 "
                    f"for serial_waves. Received {self.max_simultaneous_ephemeral_loras}."
                )
            return "serial_waves"
        if self.multi_knowledge_strategy == "parallel_waves":
            if self.max_simultaneous_ephemeral_loras < 1:
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer requires max_simultaneous_ephemeral_loras >= 1 "
                    f"for parallel_waves. Received {self.max_simultaneous_ephemeral_loras}."
                )
            self._require_async_rollout_targeting("parallel_waves")
            return "parallel_waves"

        raise ValueError(
            "KnowledgeUpdatePPOTrainer staged-v1 expects at most one knowledge group per rollout batch. "
            f"Received grouped knowledge_ids={knowledge_ids}. "
            "Set `trainer.knowledge_update.multi_knowledge_strategy=serial_groups` or "
            "`serial_waves`/`parallel_waves` once grouped sub-batch execution is enabled."
        )

    def _build_rollout_execution_schedule(self, execution_plan: list[dict[str, Any]]) -> dict[str, Any]:
        execution_strategy = self._resolve_rollout_execution_strategy(execution_plan)
        if execution_strategy == "single_group":
            active_plan_item = execution_plan[0] if execution_plan else {"knowledge_id": None, "lora_path": None}
            return {
                "strategy": "single_group",
                "wave_count": 1 if execution_plan else 0,
                "max_active_loras": 1,
                "waves": [
                    {
                        "wave_index": 0,
                        "plan_items": [active_plan_item],
                    }
                ]
                if execution_plan
                else [],
            }

        if execution_strategy == "serial_groups":
            return {
                "strategy": "serial_groups",
                "wave_count": len(execution_plan),
                "max_active_loras": 1,
                "waves": [
                    {
                        "wave_index": wave_index,
                        "plan_items": [plan_item],
                    }
                    for wave_index, plan_item in enumerate(execution_plan)
                ],
            }
        if execution_strategy == "serial_waves":
            wave_capacity = max(1, self.max_simultaneous_ephemeral_loras)
            waves: list[dict[str, Any]] = []
            for wave_index, wave_start in enumerate(range(0, len(execution_plan), wave_capacity)):
                waves.append(
                    {
                        "wave_index": wave_index,
                        "plan_items": execution_plan[wave_start : wave_start + wave_capacity],
                    }
                )
            return {
                "strategy": "serial_waves",
                "wave_count": len(waves),
                "max_active_loras": 1,
                "max_planned_loras_per_wave": wave_capacity,
                "waves": waves,
            }
        if execution_strategy == "parallel_waves":
            server_ids = self._get_async_rollout_server_ids()
            if not server_ids:
                raise RuntimeError(
                    "KnowledgeUpdatePPOTrainer requires async rollout server ids for `parallel_waves`."
                )
            max_loras_per_server = max(1, self.max_ephemeral_loras_per_async_server)
            server_slots = [
                (replica_index, server_ids[replica_index])
                for slot_round in range(max_loras_per_server)
                for replica_index in range(len(server_ids))
                if slot_round >= 0
            ]
            wave_capacity = min(max(1, self.max_simultaneous_ephemeral_loras), len(server_slots))
            waves = []
            for wave_index, wave_start in enumerate(range(0, len(execution_plan), wave_capacity)):
                raw_plan_items = execution_plan[wave_start : wave_start + wave_capacity]
                plan_items = []
                for slot_index, plan_item in enumerate(raw_plan_items):
                    assigned_plan_item = dict(plan_item)
                    replica_index, preferred_server_id = server_slots[slot_index]
                    assigned_plan_item["replica_index"] = replica_index
                    assigned_plan_item["preferred_server_id"] = preferred_server_id
                    plan_items.append(assigned_plan_item)
                waves.append(
                    {
                        "wave_index": wave_index,
                        "plan_items": plan_items,
                    }
                )
            return {
                "strategy": "parallel_waves",
                "wave_count": len(waves),
                "max_active_loras": wave_capacity,
                "max_planned_loras_per_wave": wave_capacity,
                "max_planned_loras_per_server": max_loras_per_server,
                "server_ids": list(server_ids),
                "waves": waves,
            }

        raise ValueError(f"Unsupported rollout execution strategy: {execution_strategy}")

    def _execute_rollout_execution_schedule(
        self,
        gen_batch,
        execution_schedule: dict[str, Any],
        original_generate_sequences,
        *args,
        **kwargs,
    ) -> DataProto:
        strategy = execution_schedule.get("strategy", "single_group")
        if strategy == "single_group":
            waves = list(execution_schedule.get("waves", []))
            if not waves:
                return original_generate_sequences(gen_batch, *args, **kwargs)
            return self._execute_rollout_execution_wave(
                gen_batch=gen_batch,
                execution_wave=waves[0],
                original_generate_sequences=original_generate_sequences,
                *args,
                **kwargs,
            )

        if strategy == "serial_groups":
            return self._generate_sequences_by_schedule_waves(
                gen_batch=gen_batch,
                execution_schedule=execution_schedule,
                original_generate_sequences=original_generate_sequences,
                *args,
                **kwargs,
            )
        if strategy == "serial_waves":
            return self._generate_sequences_by_schedule_waves(
                gen_batch=gen_batch,
                execution_schedule=execution_schedule,
                original_generate_sequences=original_generate_sequences,
                *args,
                **kwargs,
            )
        if strategy == "parallel_waves":
            return self._generate_sequences_by_parallel_waves(
                gen_batch=gen_batch,
                execution_schedule=execution_schedule,
                original_generate_sequences=original_generate_sequences,
                *args,
                **kwargs,
            )

        raise ValueError(f"Unsupported rollout execution schedule strategy: {strategy}")

    def _execute_rollout_execution_wave(
        self,
        gen_batch,
        execution_wave: dict[str, Any],
        original_generate_sequences,
        *args,
        **kwargs,
    ) -> DataProto:
        plan_items = list(execution_wave.get("plan_items", []))
        if not plan_items:
            return original_generate_sequences(gen_batch, *args, **kwargs)
        if len(plan_items) != 1:
            raise ValueError(
                "KnowledgeUpdatePPOTrainer staged-v1 only supports one active ephemeral LoRA per execution wave. "
                f"Received {len(plan_items)} plan items in wave {execution_wave.get('wave_index')}."
            )

        return self._generate_sequences_for_plan_item(
            gen_batch=gen_batch,
            plan_item=plan_items[0],
            original_generate_sequences=original_generate_sequences,
            *args,
            **kwargs,
        )

    def _generate_sequences_for_plan_item(
        self,
        gen_batch,
        plan_item: dict[str, Any],
        original_generate_sequences,
        staged_knowledge_ids: list[str | None] | None = None,
        *args,
        **kwargs,
    ) -> DataProto:
        knowledge_id = plan_item["knowledge_id"]
        if self.enable_ephemeral_lora:
            self._materialize_ephemeral_lora_requests_for_plan_items([plan_item])
        if self.enable_ephemeral_lora and staged_knowledge_ids is None:
            self._load_ephemeral_lora_for_batch(
                gen_batch=gen_batch,
                knowledge_id=knowledge_id,
                lora_path=plan_item.get("lora_path"),
                eager_load=True,
                expected_staged_knowledge_ids=None,
            )

        try:
            if self.enable_ephemeral_lora and staged_knowledge_ids is not None:
                self._activate_staged_ephemeral_lora(
                    knowledge_id=knowledge_id,
                    expected_staged_knowledge_ids=staged_knowledge_ids,
                )
            output = original_generate_sequences(gen_batch, *args, **kwargs)
            output = self._annotate_rollout_output_with_lora_loss_paths(
                output=output,
                row_indices=list(plan_item.get("row_indices", [])),
                plan_items=[plan_item],
            )
            if self.enable_ephemeral_lora and self.verify_ephemeral_lora_state:
                self._verify_ephemeral_lora_state(
                    expected_staged=staged_knowledge_ids or knowledge_id,
                    expected_loaded=knowledge_id,
                    phase="after_generate",
                )
            return output
        finally:
            if self.enable_ephemeral_lora:
                remaining_staged_knowledge_ids = None
                if staged_knowledge_ids is not None:
                    remaining_staged_knowledge_ids = [
                        staged_id for staged_id in staged_knowledge_ids if staged_id != knowledge_id
                    ]
                self._clear_ephemeral_lora(
                    knowledge_id=knowledge_id,
                    expected_staged_knowledge_ids=remaining_staged_knowledge_ids,
                )
                self._delete_ephemeral_lora_path_after_use(
                    lora_path=plan_item.get("lora_path"),
                    knowledge_id=knowledge_id,
                    lora_variant=plan_item.get("lora_variant"),
                )

    def _generate_sequences_by_schedule_waves(
        self,
        gen_batch,
        execution_schedule: dict[str, Any],
        original_generate_sequences,
        *args,
        **kwargs,
    ) -> DataProto:
        grouped_outputs: list[DataProto] = []
        output_row_indices: list[int] = []

        for execution_wave in execution_schedule.get("waves", []):
            plan_items = list(execution_wave.get("plan_items", []))
            staged_knowledge_ids = [self._normalize_optional_id(item.get("knowledge_id")) for item in plan_items]
            active_staged_knowledge_ids = list(staged_knowledge_ids)
            wave_row_indices = [row_index for item in plan_items for row_index in item.get("row_indices", [])]
            self._record_ephemeral_lora_event(
                event="schedule_wave",
                knowledge_id=None,
                wave_index=execution_wave.get("wave_index"),
                row_indices=wave_row_indices,
                row_count=len(wave_row_indices),
                knowledge_ids=[item.get("knowledge_id") for item in plan_items],
                lora_paths=[item.get("lora_path") for item in plan_items],
                planned_item_count=len(plan_items),
                max_planned_loras_per_wave=execution_schedule.get("max_planned_loras_per_wave", 1),
            )
            if self.enable_ephemeral_lora and execution_schedule.get("strategy") == "serial_waves":
                self._materialize_ephemeral_lora_requests_for_plan_items(plan_items)
                self._stage_ephemeral_lora_wave(plan_items=plan_items, staged_knowledge_ids=staged_knowledge_ids)
            for plan_item_index, plan_item in enumerate(plan_items):
                row_indices = list(plan_item.get("row_indices", []))
                if not row_indices:
                    continue
                knowledge_id = plan_item["knowledge_id"]
                group_batch = gen_batch.select_idxs(row_indices)
                self._record_ephemeral_lora_event(
                    event="serial_group",
                    knowledge_id=knowledge_id,
                    wave_index=execution_wave.get("wave_index"),
                    wave_item_index=plan_item_index,
                    row_indices=row_indices,
                    row_count=len(row_indices),
                    lora_path=plan_item.get("lora_path"),
                )
                group_output = self._generate_sequences_for_plan_item(
                    gen_batch=group_batch,
                    plan_item=plan_item,
                    original_generate_sequences=original_generate_sequences,
                    staged_knowledge_ids=(
                        active_staged_knowledge_ids if execution_schedule.get("strategy") == "serial_waves" else None
                    ),
                    *args,
                    **kwargs,
                )
                grouped_outputs.append(group_output)
                output_row_indices.extend(row_indices)
                if self.enable_ephemeral_lora and execution_schedule.get("strategy") == "serial_waves":
                    active_staged_knowledge_ids = [
                        staged_id for staged_id in active_staged_knowledge_ids if staged_id != knowledge_id
                    ]

        if not grouped_outputs:
            return original_generate_sequences(gen_batch, *args, **kwargs)

        return self._concat_group_outputs_in_input_order(grouped_outputs, output_row_indices)

    def _generate_sequences_by_parallel_waves(
        self,
        gen_batch,
        execution_schedule: dict[str, Any],
        original_generate_sequences,
        *args,
        **kwargs,
    ) -> DataProto:
        grouped_outputs: list[DataProto] = []
        output_row_indices: list[int] = []

        for execution_wave in execution_schedule.get("waves", []):
            plan_items = list(execution_wave.get("plan_items", []))
            wave_row_indices = sorted(
                {int(row_index) for item in plan_items for row_index in item.get("row_indices", [])},
                key=int,
            )
            if not wave_row_indices:
                continue

            self._record_ephemeral_lora_event(
                event="schedule_wave",
                knowledge_id=None,
                wave_index=execution_wave.get("wave_index"),
                row_indices=wave_row_indices,
                row_count=len(wave_row_indices),
                knowledge_ids=[item.get("knowledge_id") for item in plan_items],
                lora_paths=[item.get("lora_path") for item in plan_items],
                preferred_server_ids=[item.get("preferred_server_id") for item in plan_items],
                replica_indices=[item.get("replica_index") for item in plan_items],
                planned_item_count=len(plan_items),
                max_planned_loras_per_wave=execution_schedule.get("max_planned_loras_per_wave", 1),
            )

            self._log_memory_diagnostics(
                phase="before_parallel_wave_stage",
                wave_index=execution_wave.get("wave_index"),
                plan_items=plan_items,
            )
            if self.enable_ephemeral_lora:
                self._materialize_ephemeral_lora_requests_for_plan_items(plan_items)
            self._stage_parallel_ephemeral_lora_wave(plan_items=plan_items)
            wave_batch = gen_batch.select_idxs(wave_row_indices)
            preferred_server_ids_by_row = self._build_preferred_server_ids_for_wave(
                row_indices=wave_row_indices,
                plan_items=plan_items,
            )
            wave_batch.non_tensor_batch["preferred_server_id"] = np.array(preferred_server_ids_by_row, dtype=object)
            preferred_ephemeral_lora_knowledge_ids_by_row = self._build_ephemeral_lora_knowledge_ids_for_wave(
                row_indices=wave_row_indices,
                plan_items=plan_items,
            )
            wave_batch.non_tensor_batch["preferred_ephemeral_lora_knowledge_id"] = np.array(
                preferred_ephemeral_lora_knowledge_ids_by_row,
                dtype=object,
            )

            try:
                wave_output = original_generate_sequences(wave_batch, *args, **kwargs)
                wave_output = self._annotate_rollout_output_with_lora_loss_paths(
                    output=wave_output,
                    row_indices=wave_row_indices,
                    plan_items=plan_items,
                )
                if self.enable_ephemeral_lora and self.verify_ephemeral_lora_state:
                    self._verify_parallel_ephemeral_lora_wave_state(
                        plan_items=plan_items,
                        phase="after_generate",
                        expected_loaded=True,
                    )
                self._log_memory_diagnostics(
                    phase="after_parallel_wave_generate",
                    wave_index=execution_wave.get("wave_index"),
                    plan_items=plan_items,
                )
                grouped_outputs.append(wave_output)
                output_row_indices.extend(wave_row_indices)
            finally:
                self._clear_parallel_ephemeral_lora_wave(plan_items=plan_items)
                self._log_memory_diagnostics(
                    phase="after_parallel_wave_clear",
                    wave_index=execution_wave.get("wave_index"),
                    plan_items=plan_items,
                )

        if not grouped_outputs:
            return original_generate_sequences(gen_batch, *args, **kwargs)

        return self._concat_group_outputs_in_input_order(grouped_outputs, output_row_indices)

    def _extract_knowledge_ids(self, gen_batch) -> set[str]:
        return {
            plan_item["knowledge_id"]
            for plan_item in self._build_ephemeral_lora_execution_plan(gen_batch)
            if plan_item["knowledge_id"] is not None
        }

    @staticmethod
    def _to_python_list(value: Any) -> list[Any]:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _as_object_vector(values: Sequence[Any]) -> np.ndarray:
        object_values = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            object_values[index] = value
        return object_values

    def _load_ephemeral_lora_for_batch(
        self,
        gen_batch,
        knowledge_id: str | None,
        lora_path: str | None = None,
        eager_load: bool = True,
        expected_staged_knowledge_ids: list[str | None] | None = None,
        verify_state: bool = True,
        replica_indices: list[int] | None = None,
    ) -> None:
        lora_path = lora_path or self._resolve_ephemeral_lora_path(gen_batch=gen_batch, knowledge_id=knowledge_id)
        if lora_path is None:
            if self.allow_missing_ephemeral_lora:
                return
            raise FileNotFoundError(
                "KnowledgeUpdatePPOTrainer could not resolve an ephemeral LoRA path. "
                f"Checked batch field `{self.ephemeral_lora_path_field}` and "
                "`trainer.knowledge_update.ephemeral_lora_dir`."
            )
        lora_path = self._validate_ephemeral_lora_path(lora_path)

        controller = self._get_ephemeral_lora_controller()
        stage_kwargs = {
            "lora_path": lora_path,
            "knowledge_id": knowledge_id,
        }
        if controller is getattr(self, "async_rollout_manager", None) or hasattr(controller, "load_staged_ephemeral_lora"):
            stage_kwargs["eager_load"] = eager_load
        if replica_indices is not None and controller is getattr(self, "async_rollout_manager", None):
            stage_kwargs["replica_indices"] = list(replica_indices)
        worker_result = controller.stage_ephemeral_lora(**stage_kwargs)
        self._record_ephemeral_lora_event(
            event="stage",
            knowledge_id=knowledge_id,
            lora_path=lora_path,
            eager_load=eager_load,
            replica_indices=replica_indices,
            worker_result=worker_result,
        )
        if self.verify_ephemeral_lora_state and verify_state:
            expected_loaded = knowledge_id if eager_load else None
            if replica_indices is None:
                self._verify_ephemeral_lora_state(
                    expected_staged=expected_staged_knowledge_ids or knowledge_id,
                    expected_loaded=expected_loaded,
                    phase="after_stage",
                )
            else:
                self._verify_ephemeral_lora_state_for_replicas(
                    replica_indices=replica_indices,
                    expected_staged=expected_staged_knowledge_ids or knowledge_id,
                    expected_loaded=expected_loaded,
                    phase="after_stage",
                )

    def _activate_staged_ephemeral_lora(
        self,
        *,
        knowledge_id: str | None,
        expected_staged_knowledge_ids: list[str | None] | None = None,
        replica_indices: list[int] | None = None,
    ) -> None:
        controller = self._get_ephemeral_lora_controller()
        if not hasattr(controller, "load_staged_ephemeral_lora"):
            return
        load_kwargs = {"knowledge_id": knowledge_id}
        if replica_indices is not None and controller is getattr(self, "async_rollout_manager", None):
            load_kwargs["replica_indices"] = list(replica_indices)
        worker_result = controller.load_staged_ephemeral_lora(**load_kwargs)
        self._record_ephemeral_lora_event(
            event="load",
            knowledge_id=knowledge_id,
            replica_indices=replica_indices,
            worker_result=worker_result,
        )
        if self.verify_ephemeral_lora_state:
            if replica_indices is None:
                self._verify_ephemeral_lora_state(
                    expected_staged=expected_staged_knowledge_ids or knowledge_id,
                    expected_loaded=knowledge_id,
                    phase="after_load",
                )
            else:
                self._verify_ephemeral_lora_state_for_replicas(
                    replica_indices=replica_indices,
                    expected_staged=expected_staged_knowledge_ids or knowledge_id,
                    expected_loaded=knowledge_id,
                    phase="after_load",
                )

    def _stage_ephemeral_lora_wave(
        self,
        *,
        plan_items: list[dict[str, Any]],
        staged_knowledge_ids: list[str | None],
    ) -> None:
        for plan_item in plan_items:
            self._load_ephemeral_lora_for_batch(
                gen_batch=None,
                knowledge_id=plan_item.get("knowledge_id"),
                lora_path=plan_item.get("lora_path"),
                eager_load=False,
                expected_staged_knowledge_ids=None,
                verify_state=False,
                replica_indices=None,
            )
        self._record_ephemeral_lora_event(
            event="wave_stage",
            knowledge_id=None,
            knowledge_ids=staged_knowledge_ids,
            planned_item_count=len(plan_items),
        )
        if self.verify_ephemeral_lora_state:
            self._verify_ephemeral_lora_state(
                expected_staged=staged_knowledge_ids,
                expected_loaded=None,
                phase="after_wave_stage",
            )

    def _clear_ephemeral_lora(
        self,
        knowledge_id: str | None,
        expected_staged_knowledge_ids: list[str | None] | None = None,
        expected_loaded_knowledge_ids: list[str | None] | None = None,
        replica_indices: list[int] | None = None,
    ) -> None:
        controller = self._get_ephemeral_lora_controller()
        clear_kwargs = {"knowledge_id": knowledge_id}
        if replica_indices is not None and controller is getattr(self, "async_rollout_manager", None):
            clear_kwargs["replica_indices"] = list(replica_indices)
        worker_result = controller.clear_ephemeral_lora(**clear_kwargs)
        self._record_ephemeral_lora_event(
            event="clear",
            knowledge_id=knowledge_id,
            replica_indices=replica_indices,
            worker_result=worker_result,
        )
        if self.verify_ephemeral_lora_state:
            if replica_indices is None:
                self._verify_ephemeral_lora_state(
                    expected_staged=expected_staged_knowledge_ids,
                    expected_loaded=expected_loaded_knowledge_ids,
                    phase="after_clear",
                )
            else:
                self._verify_ephemeral_lora_state_for_replicas(
                    replica_indices=replica_indices,
                    expected_staged=expected_staged_knowledge_ids,
                    expected_loaded=expected_loaded_knowledge_ids,
                    phase="after_clear",
                )

    def _stage_parallel_ephemeral_lora_wave(self, *, plan_items: list[dict[str, Any]]) -> None:
        for plan_item in plan_items:
            replica_index = plan_item.get("replica_index")
            self._load_ephemeral_lora_for_batch(
                gen_batch=None,
                knowledge_id=plan_item.get("knowledge_id"),
                lora_path=plan_item.get("lora_path"),
                eager_load=True,
                expected_staged_knowledge_ids=None,
                verify_state=False,
                replica_indices=[replica_index] if replica_index is not None else None,
            )
        self._record_ephemeral_lora_event(
            event="wave_stage",
            knowledge_id=None,
            knowledge_ids=[plan_item.get("knowledge_id") for plan_item in plan_items],
            preferred_server_ids=[plan_item.get("preferred_server_id") for plan_item in plan_items],
            replica_indices=[plan_item.get("replica_index") for plan_item in plan_items],
            planned_item_count=len(plan_items),
        )
        if self.verify_ephemeral_lora_state:
            self._verify_parallel_ephemeral_lora_wave_state(
                plan_items=plan_items,
                phase="after_stage",
                expected_loaded=True,
            )

    def _clear_parallel_ephemeral_lora_wave(self, *, plan_items: list[dict[str, Any]]) -> None:
        remaining_plan_items = list(plan_items)
        for plan_item in plan_items:
            knowledge_id = plan_item.get("knowledge_id")
            replica_index = plan_item.get("replica_index")
            remaining_plan_items = [
                candidate
                for candidate in remaining_plan_items
                if candidate.get("knowledge_id") != knowledge_id or candidate.get("replica_index") != replica_index
            ]
            remaining_staged_knowledge_ids = [
                candidate.get("knowledge_id") for candidate in remaining_plan_items if candidate.get("replica_index") == replica_index
            ]
            self._clear_ephemeral_lora(
                knowledge_id=knowledge_id,
                expected_staged_knowledge_ids=remaining_staged_knowledge_ids,
                expected_loaded_knowledge_ids=remaining_staged_knowledge_ids,
                replica_indices=[replica_index] if replica_index is not None else None,
            )
            self._delete_ephemeral_lora_path_after_use(
                lora_path=plan_item.get("lora_path"),
                knowledge_id=knowledge_id,
                lora_variant=plan_item.get("lora_variant"),
            )

    def _build_preferred_server_ids_for_wave(
        self,
        *,
        row_indices: list[int],
        plan_items: list[dict[str, Any]],
    ) -> list[str | None]:
        preferred_server_id_by_row_index: dict[int, str | None] = {}
        for plan_item in plan_items:
            preferred_server_id = self._normalize_optional_id(plan_item.get("preferred_server_id"))
            for row_index in plan_item.get("row_indices", []):
                existing_server_id = preferred_server_id_by_row_index.get(row_index)
                if existing_server_id is not None and existing_server_id != preferred_server_id:
                    raise ValueError(
                        "KnowledgeUpdatePPOTrainer parallel wave row routing conflict: "
                        f"row_index={row_index} existing_server_id={existing_server_id!r} "
                        f"new_server_id={preferred_server_id!r}."
                    )
                preferred_server_id_by_row_index[row_index] = preferred_server_id
        return [preferred_server_id_by_row_index.get(row_index) for row_index in row_indices]

    def _build_ephemeral_lora_knowledge_ids_for_wave(
        self,
        *,
        row_indices: list[int],
        plan_items: list[dict[str, Any]],
    ) -> list[str | None]:
        knowledge_id_by_row_index: dict[int, str | None] = {}
        for plan_item in plan_items:
            knowledge_id = self._normalize_optional_id(plan_item.get("knowledge_id"))
            for row_index in plan_item.get("row_indices", []):
                existing_knowledge_id = knowledge_id_by_row_index.get(row_index)
                if existing_knowledge_id is not None and existing_knowledge_id != knowledge_id:
                    raise ValueError(
                        "KnowledgeUpdatePPOTrainer parallel wave knowledge routing conflict: "
                        f"row_index={row_index} existing_knowledge_id={existing_knowledge_id!r} "
                        f"new_knowledge_id={knowledge_id!r}."
                    )
                knowledge_id_by_row_index[row_index] = knowledge_id
        return [knowledge_id_by_row_index.get(row_index) for row_index in row_indices]

    def _validate_rollout_execution_schedule(
        self,
        *,
        execution_schedule: dict[str, Any],
        batch_size: int | None,
    ) -> None:
        waves = list(execution_schedule.get("waves", []))
        if not waves:
            return

        scheduled_row_indices: list[int] = []
        for execution_wave in waves:
            plan_items = list(execution_wave.get("plan_items", []))
            wave_index = execution_wave.get("wave_index")
            if not plan_items:
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer produced an empty execution wave. "
                    f"wave_index={wave_index}."
                )

            wave_row_indices: list[int] = []
            for plan_item in plan_items:
                plan_row_indices = [int(row_index) for row_index in plan_item.get("row_indices", [])]
                if not plan_row_indices:
                    raise ValueError(
                        "KnowledgeUpdatePPOTrainer execution plan item is missing row indices. "
                        f"wave_index={wave_index} knowledge_id={plan_item.get('knowledge_id')!r}."
                    )
                wave_row_indices.extend(plan_row_indices)
                scheduled_row_indices.extend(plan_row_indices)

            if len(set(wave_row_indices)) != len(wave_row_indices):
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer execution wave contains duplicate row indices. "
                    f"wave_index={wave_index} row_indices={wave_row_indices}."
                )

            if execution_schedule.get("strategy") == "parallel_waves":
                self._validate_parallel_wave_assignments(
                    plan_items=plan_items,
                    wave_index=wave_index,
                )

        if len(set(scheduled_row_indices)) != len(scheduled_row_indices):
            raise ValueError(
                "KnowledgeUpdatePPOTrainer execution schedule contains duplicate row indices across waves. "
                f"row_indices={scheduled_row_indices}."
            )

        if batch_size is not None:
            expected_row_indices = list(range(batch_size))
            if sorted(scheduled_row_indices) != expected_row_indices:
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer execution schedule does not cover the batch exactly once. "
                    f"expected_row_indices={expected_row_indices} scheduled_row_indices={sorted(scheduled_row_indices)}."
                )

    def _validate_parallel_wave_assignments(
        self,
        *,
        plan_items: list[dict[str, Any]],
        wave_index: int | None,
    ) -> None:
        replica_counts: dict[int, int] = {}
        preferred_server_id_counts: dict[str, int] = {}
        for plan_item in plan_items:
            knowledge_id = plan_item.get("knowledge_id")
            replica_index = plan_item.get("replica_index")
            preferred_server_id = self._normalize_optional_id(plan_item.get("preferred_server_id"))
            if replica_index is None:
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer parallel wave item is missing replica_index. "
                    f"wave_index={wave_index} knowledge_id={knowledge_id!r}."
                )
            if preferred_server_id is None:
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer parallel wave item is missing preferred_server_id. "
                    f"wave_index={wave_index} knowledge_id={knowledge_id!r} replica_index={replica_index}."
                )
            replica_index = int(replica_index)
            replica_counts[replica_index] = replica_counts.get(replica_index, 0) + 1
            preferred_server_id_counts[preferred_server_id] = preferred_server_id_counts.get(preferred_server_id, 0) + 1

        max_loras_per_server = max(1, self.max_ephemeral_loras_per_async_server)
        overfull_replicas = {
            replica_index: count
            for replica_index, count in replica_counts.items()
            if count > max_loras_per_server
        }
        if overfull_replicas:
            raise ValueError(
                "KnowledgeUpdatePPOTrainer parallel wave exceeds per-replica LoRA capacity. "
                f"wave_index={wave_index} replica_counts={replica_counts} "
                f"max_ephemeral_loras_per_async_server={max_loras_per_server}."
            )
        overfull_servers = {
            preferred_server_id: count
            for preferred_server_id, count in preferred_server_id_counts.items()
            if count > max_loras_per_server
        }
        if overfull_servers:
            raise ValueError(
                "KnowledgeUpdatePPOTrainer parallel wave exceeds per-server LoRA capacity. "
                f"wave_index={wave_index} preferred_server_id_counts={preferred_server_id_counts} "
                f"max_ephemeral_loras_per_async_server={max_loras_per_server}."
            )

    def _verify_parallel_ephemeral_lora_wave_state(
        self,
        *,
        plan_items: list[dict[str, Any]],
        phase: str,
        expected_loaded: bool,
    ) -> None:
        expected_states_by_replica: dict[int, dict[str, Any]] = {}
        for plan_item in plan_items:
            replica_index = plan_item.get("replica_index")
            if replica_index is None:
                continue
            knowledge_id = self._normalize_optional_id(plan_item.get("knowledge_id"))
            replica_index = int(replica_index)
            replica_state = expected_states_by_replica.setdefault(
                replica_index,
                {
                    "expected_staged": [],
                    "expected_loaded": [],
                },
            )
            if knowledge_id is not None:
                replica_state["expected_staged"].append(knowledge_id)
                if expected_loaded:
                    replica_state["expected_loaded"].append(knowledge_id)
        if expected_states_by_replica:
            self._verify_ephemeral_lora_states_by_replica(
                expected_states_by_replica=expected_states_by_replica,
                phase=phase,
            )

    def _resolve_ephemeral_lora_path(
        self,
        gen_batch,
        knowledge_id: str | None,
        lora_variant: str | None = None,
    ) -> str | None:
        execution_plan = self._build_ephemeral_lora_execution_plan(gen_batch)
        for plan_item in execution_plan:
            if plan_item["knowledge_id"] == self._normalize_optional_id(knowledge_id):
                plan_item_variant = self._normalize_lora_variant(plan_item.get("lora_variant", self.default_lora_variant))
                if lora_variant is not None and plan_item_variant != self._normalize_lora_variant(lora_variant):
                    continue
                return plan_item.get("lora_path")
        return self._build_ephemeral_lora_dir_from_knowledge_id(knowledge_id, lora_variant=lora_variant)

    def _build_ephemeral_lora_dir_from_knowledge_id(
        self,
        knowledge_id: str | None,
        *,
        lora_variant: str | None = None,
    ) -> str | None:
        if not self.ephemeral_lora_dir or knowledge_id in (None, "", "None"):
            return None
        normalized_variant = self._normalize_lora_variant(lora_variant or self.default_lora_variant)
        knowledge_segment = self._sanitize_ephemeral_lora_build_filename(str(knowledge_id))
        if normalized_variant == KNOWLEDGE_LORA_VARIANT:
            return os.path.abspath(os.path.join(os.fspath(self.ephemeral_lora_dir), knowledge_segment))
        return os.path.abspath(
            os.path.join(os.fspath(self.ephemeral_lora_dir), normalized_variant, knowledge_segment)
        )

    def _build_ephemeral_lora_execution_plan(self, gen_batch) -> list[dict[str, Any]]:
        knowledge_groups = self._collect_batch_knowledge_groups(gen_batch)
        execution_plan: list[dict[str, Any]] = []
        for group_index, group in enumerate(knowledge_groups):
            lora_variant, lora_request = self._select_lora_variant_for_group(group=group)
            lora_path = self._normalize_optional_path((lora_request or {}).get("lora_path"))
            lora_path, build_request = self._resolve_or_defer_ephemeral_lora_path_for_group(
                group=group,
                knowledge_id=group["knowledge_id"],
                lora_path=lora_path,
                lora_variant=lora_variant,
            )
            execution_plan.append(
                {
                    "group_index": group_index,
                    "knowledge_id": group["knowledge_id"],
                    "lora_variant": lora_variant,
                    "lora_path": lora_path,
                    "build_request": dict(build_request) if build_request is not None else None,
                    "lora_request": {
                        **(lora_request or {}),
                        "knowledge_id": group["knowledge_id"],
                        "lora_variant": lora_variant,
                        "lora_path": lora_path,
                    },
                    "row_indices": list(group["row_indices"]),
                    "row_count": len(group["row_indices"]),
                    "request_count": len(group["request_items"]),
                    "max_simultaneous_ephemeral_loras": self.max_simultaneous_ephemeral_loras,
                }
            )

        if self.log_variant_lora_norm:
            for plan_item in execution_plan:
                if plan_item.get("build_request") is None:
                    self._cache_ephemeral_lora_norm(plan_item.get("lora_path"))

        non_null_knowledge_ids = [
            plan_item["knowledge_id"] for plan_item in execution_plan if plan_item["knowledge_id"] is not None
        ]
        if self.require_single_knowledge_id and len(non_null_knowledge_ids) > 1:
            raise ValueError(
                "KnowledgeUpdatePPOTrainer requires a single knowledge_id per rollout batch, "
                f"but received {sorted(non_null_knowledge_ids)}."
            )
        return execution_plan

    def _update_plan_item_ephemeral_lora_path(
        self,
        *,
        plan_item: dict[str, Any],
        resolved_path: str,
    ) -> str:
        normalized_path = self._validate_ephemeral_lora_path(resolved_path)
        plan_item["lora_path"] = normalized_path
        if isinstance(plan_item.get("lora_request"), dict):
            plan_item["lora_request"]["lora_path"] = normalized_path
        plan_item["build_request"] = None
        return normalized_path

    def _materialize_ephemeral_lora_requests_for_plan_items(
        self,
        plan_items: list[dict[str, Any]],
    ) -> None:
        if not plan_items:
            return

        build_requests: list[dict[str, Any]] = []
        request_keys_to_items: dict[str, list[dict[str, Any]]] = {}
        for plan_item in plan_items:
            existing_path = self._normalize_optional_path(plan_item.get("lora_path"))
            build_request = plan_item.get("build_request") if isinstance(plan_item.get("build_request"), dict) else None
            if build_request is None:
                if existing_path is not None and self._ephemeral_lora_path_has_required_files(existing_path):
                    self._update_plan_item_ephemeral_lora_path(plan_item=plan_item, resolved_path=existing_path)
                continue

            request = dict(build_request)
            request_key = os.path.abspath(os.fspath(request["request_key"]))
            output_dir = os.path.abspath(os.fspath(request["output_dir"]))
            request["request_key"] = request_key
            request["output_dir"] = output_dir
            plan_item["build_request"] = request

            if self._ephemeral_lora_path_has_required_files(output_dir):
                self._update_plan_item_ephemeral_lora_path(plan_item=plan_item, resolved_path=output_dir)
                continue

            build_requests.append(request)
            request_keys_to_items.setdefault(request_key, []).append(plan_item)

        if not build_requests:
            return

        built_paths = self._build_missing_ephemeral_lora_requests(build_requests)
        for request_key, target_items in request_keys_to_items.items():
            resolved_path = built_paths[request_key]
            for plan_item in target_items:
                self._update_plan_item_ephemeral_lora_path(plan_item=plan_item, resolved_path=resolved_path)

    def _collect_batch_knowledge_groups(self, gen_batch) -> list[dict[str, Any]]:
        non_tensor_batch = getattr(gen_batch, "non_tensor_batch", None)
        if non_tensor_batch is None:
            return []

        knowledge_values = self._expand_batch_field(non_tensor_batch.get("knowledge_id", None))
        extra_infos = self._expand_batch_field(non_tensor_batch.get("extra_info", None))
        lora_paths = self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_path_field, None))
        request_items = self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_request_field, None))
        variant_items = self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_variants_field, None))
        selected_variants = self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_variant_field, None))

        batch_size = max(
            len(knowledge_values),
            len(extra_infos),
            len(lora_paths),
            len(request_items),
            len(variant_items),
            len(selected_variants),
            1,
        )
        knowledge_values = self._pad_optional_values(knowledge_values, batch_size)
        extra_infos = self._pad_optional_values(extra_infos, batch_size)
        lora_paths = self._pad_optional_values(lora_paths, batch_size)
        request_items = self._pad_optional_values(request_items, batch_size)
        variant_items = self._pad_optional_values(variant_items, batch_size)
        selected_variants = self._pad_optional_values(selected_variants, batch_size)

        grouped_rows: dict[str, dict[str, Any]] = {}
        row_order: list[str] = []
        for row_index in range(batch_size):
            extra_info = extra_infos[row_index] if isinstance(extra_infos[row_index], dict) else {}
            request_item = request_items[row_index] if isinstance(request_items[row_index], dict) else {}
            variant_item = variant_items[row_index] if isinstance(variant_items[row_index], dict) else {}
            mount_ephemeral_lora = extra_info.get("mount_ephemeral_lora", True)
            if not self._normalize_bool(mount_ephemeral_lora):
                continue
            selected_variant = self._normalize_lora_variant(
                selected_variants[row_index]
                or extra_info.get(self.ephemeral_lora_variant_field)
                or extra_info.get("lora_variant")
                or request_item.get("lora_variant")
                or self.default_lora_variant
            )
            knowledge_id = self._normalize_optional_id(knowledge_values[row_index])
            if knowledge_id is None:
                knowledge_id = self._normalize_optional_id(extra_info.get("knowledge_id"))
            if knowledge_id is None:
                knowledge_id = self._normalize_optional_id(request_item.get("knowledge_id"))

            explicit_variant = self._resolve_explicit_group_variant(
                selected_variant_item=selected_variants[row_index],
                extra_info=extra_info,
                request_item=request_item,
            )
            group_key = self._build_knowledge_group_key(
                knowledge_id=knowledge_id,
                explicit_variant=explicit_variant,
                row_index=row_index,
            )
            if group_key not in grouped_rows:
                grouped_rows[group_key] = {
                    "knowledge_id": knowledge_id,
                    "row_indices": [],
                    "candidate_paths": [],
                    "extra_infos": [],
                    "request_items": [],
                    "variant_requests": {},
                    "forced_variants": [],
                }
                row_order.append(group_key)

            group = grouped_rows[group_key]
            group["row_indices"].append(row_index)
            if extra_info:
                group["extra_infos"].append(extra_info)
            if request_item:
                group["request_items"].append(request_item)

            if explicit_variant is not None:
                group["forced_variants"].append(explicit_variant)

            group["candidate_paths"].extend(
                [
                    request_item.get("lora_path"),
                    request_item.get(self.ephemeral_lora_path_field),
                    extra_info.get(self.ephemeral_lora_path_field),
                    lora_paths[row_index],
                ]
            )
            self._merge_group_variant_request(
                group=group,
                request=self._coerce_lora_request(
                    request_item,
                    knowledge_id=knowledge_id,
                    fallback_variant=selected_variant,
                ),
            )
            self._merge_group_variant_request(
                group=group,
                request=self._coerce_lora_request(
                    {
                        "knowledge_id": knowledge_id,
                        "lora_variant": selected_variant,
                        "lora_path": request_item.get(self.ephemeral_lora_path_field)
                        or extra_info.get(self.ephemeral_lora_path_field)
                        or lora_paths[row_index],
                    },
                    knowledge_id=knowledge_id,
                    fallback_variant=selected_variant,
                ),
            )
            for variant_name, variant_request in variant_item.items():
                self._merge_group_variant_request(
                    group=group,
                    request=self._coerce_lora_request(
                        variant_request,
                        knowledge_id=knowledge_id,
                        fallback_variant=variant_name,
                    ),
                )
            for variant_name, variant_request in (extra_info.get(self.ephemeral_lora_variants_field, {}) or {}).items():
                self._merge_group_variant_request(
                    group=group,
                    request=self._coerce_lora_request(
                        variant_request,
                        knowledge_id=knowledge_id,
                        fallback_variant=variant_name,
                    ),
                )

        return [grouped_rows[group_key] for group_key in row_order]

    def _build_knowledge_group_key(
        self,
        *,
        knowledge_id: str | None,
        explicit_variant: str | None,
        row_index: int,
    ) -> str:
        if knowledge_id is None:
            return f"__row_{row_index}"
        if explicit_variant is None:
            return knowledge_id
        return f"{knowledge_id}::{explicit_variant}"

    def _resolve_or_defer_ephemeral_lora_path_for_group(
        self,
        *,
        group: dict[str, Any],
        knowledge_id: str | None,
        lora_path: str | None,
        lora_variant: str,
    ) -> tuple[str | None, dict[str, Any] | None]:
        normalized_path = self._normalize_optional_path(lora_path)
        if normalized_path is not None and self._ephemeral_lora_path_has_required_files(normalized_path):
            return normalized_path, None

        if not self.build_missing_ephemeral_lora:
            return normalized_path, None

        if knowledge_id is None:
            raise ValueError(
                "KnowledgeUpdatePPOTrainer cannot build a missing ephemeral LoRA without a knowledge_id."
            )

        build_path = normalized_path or self._build_ephemeral_lora_dir_from_knowledge_id(
            knowledge_id,
            lora_variant=lora_variant,
        )
        if build_path is None:
            raise ValueError(
                "KnowledgeUpdatePPOTrainer cannot materialize a missing ephemeral LoRA because "
                "`trainer.knowledge_update.ephemeral_lora_dir` is not configured."
            )
        if self._ephemeral_lora_path_has_required_files(build_path):
            return build_path, None

        sample = self._extract_build_sample_from_group(group=group, knowledge_id=knowledge_id)
        build_path = os.path.abspath(os.fspath(build_path))
        return build_path, {
            "request_key": build_path,
            "knowledge_id": knowledge_id,
            "lora_variant": lora_variant,
            "sample": sample,
            "output_dir": build_path,
        }

    def _ensure_ephemeral_lora_path_for_group(
        self,
        *,
        group: dict[str, Any],
        knowledge_id: str | None,
        lora_path: str | None,
        lora_variant: str,
    ) -> str | None:
        resolved_path, build_request = self._resolve_or_defer_ephemeral_lora_path_for_group(
            group=group,
            knowledge_id=knowledge_id,
            lora_path=lora_path,
            lora_variant=lora_variant,
        )
        if build_request is None:
            return resolved_path
        return self._build_missing_ephemeral_lora_requests([build_request])[build_request["request_key"]]

    def _select_lora_variant_for_group(self, *, group: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        knowledge_id = group.get("knowledge_id")
        available_variant_requests = {
            variant_name: dict(request)
            for variant_name, request in group.get("variant_requests", {}).items()
            if isinstance(request, dict)
        }
        forced_variants = self._normalize_group_forced_variants(group.get("forced_variants", []))
        if forced_variants:
            chosen_variant = forced_variants[0]
            if (
                chosen_variant not in available_variant_requests
                and chosen_variant not in self.enabled_lora_variants
                and not self.build_missing_ephemeral_lora
            ):
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer received an explicit LoRA variant selection "
                    f"{chosen_variant!r} for knowledge_id={knowledge_id}, but no matching adapter "
                    "request was found and build_missing_ephemeral_lora is disabled."
                )
            request = dict(available_variant_requests.get(chosen_variant, {}))
            request.setdefault("knowledge_id", knowledge_id)
            request["lora_variant"] = chosen_variant
            return chosen_variant, request

        candidate_variants: list[str] = []
        for variant_name in self.enabled_lora_variants:
            if variant_name == NO_OP_LORA_VARIANT and not self._no_op_reward_gate_open:
                continue
            if variant_name in available_variant_requests or self.build_missing_ephemeral_lora:
                candidate_variants.append(variant_name)

        if not candidate_variants:
            candidate_variants = list(available_variant_requests) or [self.default_lora_variant]

        chosen_variant = self._choose_weighted_lora_variant(
            knowledge_id=knowledge_id,
            candidate_variants=candidate_variants,
        )
        request = dict(available_variant_requests.get(chosen_variant, {}))
        request.setdefault("knowledge_id", knowledge_id)
        request["lora_variant"] = chosen_variant
        return chosen_variant, request

    def _resolve_explicit_group_variant(
        self,
        *,
        selected_variant_item: Any,
        extra_info: dict[str, Any],
        request_item: dict[str, Any],
    ) -> str | None:
        del request_item
        raw_value = (
            selected_variant_item
            or extra_info.get(self.ephemeral_lora_variant_field)
            or extra_info.get("lora_variant")
        )
        if raw_value in (None, "", "None"):
            return None
        return self._normalize_lora_variant(raw_value)

    def _normalize_group_forced_variants(self, values: Any) -> list[str]:
        normalized_variants: list[str] = []
        for value in self._expand_batch_field(values):
            if value in (None, "", "None"):
                continue
            normalized_variants.append(self._normalize_lora_variant(value))
        normalized_variants = list(dict.fromkeys(normalized_variants))
        if len(normalized_variants) > 1:
            raise ValueError(
                "KnowledgeUpdatePPOTrainer does not support multiple explicit LoRA variants "
                f"for the same knowledge group: {normalized_variants}."
            )
        return normalized_variants

    def _choose_weighted_lora_variant(
        self,
        *,
        knowledge_id: str | None,
        candidate_variants: list[str],
    ) -> str:
        normalized_candidates = [self._normalize_lora_variant(variant_name) for variant_name in candidate_variants]
        normalized_candidates = list(dict.fromkeys(normalized_candidates))
        if len(normalized_candidates) == 1:
            return normalized_candidates[0]

        weight_items = []
        total_weight = 0.0
        for variant_name in normalized_candidates:
            variant_weight = float(self.lora_variant_weights.get(variant_name, 0.0))
            if variant_weight <= 0:
                continue
            weight_items.append((variant_name, variant_weight))
            total_weight += variant_weight
        if not weight_items or total_weight <= 0:
            return normalized_candidates[0]

        import hashlib
        import random

        sample_key = f"{knowledge_id}:{getattr(self, 'global_steps', 0)}:{self.lora_variant_selection_seed}"
        digest = hashlib.sha256(sample_key.encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        threshold = rng.random() * total_weight
        cumulative = 0.0
        for variant_name, variant_weight in weight_items:
            cumulative += variant_weight
            if threshold <= cumulative:
                return variant_name
        return weight_items[-1][0]

    def _coerce_lora_request(
        self,
        request: Any,
        *,
        knowledge_id: str | None,
        fallback_variant: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(request, dict):
            return None
        request_knowledge_id = self._normalize_optional_id(request.get("knowledge_id")) or knowledge_id
        if request_knowledge_id is None:
            return None
        lora_path = self._normalize_optional_path(
            request.get("lora_path") or request.get(self.ephemeral_lora_path_field)
        )
        normalized_request = {
            key: value
            for key, value in request.items()
            if key not in {"knowledge_id", "lora_path", self.ephemeral_lora_path_field, "lora_variant"}
        }
        normalized_request["knowledge_id"] = request_knowledge_id
        normalized_request["lora_variant"] = self._normalize_lora_variant(
            request.get("lora_variant", fallback_variant or self.default_lora_variant)
        )
        if lora_path is not None:
            normalized_request["lora_path"] = lora_path
        return normalized_request

    def _merge_group_variant_request(self, *, group: dict[str, Any], request: dict[str, Any] | None) -> None:
        if request is None:
            return
        variant_name = request["lora_variant"]
        existing_request = group["variant_requests"].get(variant_name)
        if existing_request is not None:
            existing_path = self._normalize_optional_path(existing_request.get("lora_path"))
            new_path = self._normalize_optional_path(request.get("lora_path"))
            if existing_path is not None and new_path is not None and existing_path != new_path:
                raise ValueError(
                    "KnowledgeUpdatePPOTrainer received conflicting ephemeral LoRA paths for one knowledge variant: "
                    f"knowledge_id={group.get('knowledge_id')} lora_variant={variant_name!r} "
                    f"existing_path={existing_path!r} new_path={new_path!r}"
                )
            merged_request = dict(existing_request)
            merged_request.update({key: value for key, value in request.items() if value not in (None, "", "None")})
            group["variant_requests"][variant_name] = merged_request
            return
        group["variant_requests"][variant_name] = dict(request)

    def _annotate_gen_batch_with_execution_plan(self, *, gen_batch, execution_plan: list[dict[str, Any]]) -> None:
        non_tensor_batch = getattr(gen_batch, "non_tensor_batch", None)
        batch_size = self._infer_batch_size(gen_batch)
        if non_tensor_batch is None or batch_size in (None, 0):
            return

        existing_variants = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_variant_field, None)),
            batch_size,
        )
        existing_paths = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_path_field, None)),
            batch_size,
        )
        existing_requests = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get(self.ephemeral_lora_request_field, None)),
            batch_size,
        )
        extra_infos = self._pad_optional_values(
            self._expand_batch_field(non_tensor_batch.get("extra_info", None)),
            batch_size,
        )
        selected_variants = []
        selected_paths = []
        selected_requests = []
        for row_index in range(batch_size):
            extra_info = extra_infos[row_index] if isinstance(extra_infos[row_index], dict) else {}
            selected_variants.append(
                self._normalize_lora_variant(
                    existing_variants[row_index]
                    or extra_info.get(self.ephemeral_lora_variant_field)
                    or extra_info.get("lora_variant")
                    or self.default_lora_variant
                )
            )
            selected_paths.append(existing_paths[row_index])
            existing_request = existing_requests[row_index]
            selected_requests.append(dict(existing_request) if isinstance(existing_request, dict) else None)
        for plan_item in execution_plan:
            for row_index in plan_item.get("row_indices", []):
                selected_variants[row_index] = plan_item.get("lora_variant", self.default_lora_variant)
                selected_paths[row_index] = plan_item.get("lora_path")
                selected_requests[row_index] = dict(plan_item.get("lora_request", {}))

        non_tensor_batch[self.ephemeral_lora_variant_field] = np.array(selected_variants, dtype=object)
        non_tensor_batch[self.ephemeral_lora_path_field] = np.array(selected_paths, dtype=object)
        non_tensor_batch[self.ephemeral_lora_request_field] = np.array(selected_requests, dtype=object)

        patched_extra_infos: list[dict[str, Any]] = []
        for row_index in range(batch_size):
            extra_info = dict(extra_infos[row_index]) if isinstance(extra_infos[row_index], dict) else {}
            extra_info["lora_variant"] = selected_variants[row_index]
            extra_info[self.ephemeral_lora_variant_field] = selected_variants[row_index]
            if selected_paths[row_index] is not None:
                extra_info[self.ephemeral_lora_path_field] = selected_paths[row_index]
            if selected_requests[row_index] is not None:
                extra_info[self.ephemeral_lora_request_field] = selected_requests[row_index]
            patched_extra_infos.append(extra_info)
        non_tensor_batch["extra_info"] = np.array(patched_extra_infos, dtype=object)

    def _build_missing_ephemeral_lora_requests(self, build_requests: list[dict[str, Any]]) -> dict[str, str]:
        if not build_requests:
            return {}

        deduped_requests: dict[str, dict[str, Any]] = {}
        for request in build_requests:
            request_key = os.path.abspath(os.fspath(request["request_key"]))
            request = dict(request)
            request["request_key"] = request_key
            request["output_dir"] = os.path.abspath(os.fspath(request["output_dir"]))
            deduped_requests.setdefault(request_key, request)

        resolved_paths: dict[str, str] = {}
        pending_requests: list[dict[str, Any]] = []
        for request_key, request in deduped_requests.items():
            output_dir = request["output_dir"]
            if self._ephemeral_lora_path_has_required_files(output_dir):
                resolved_paths[request_key] = self._validate_ephemeral_lora_path(output_dir)
                self._cache_ephemeral_lora_norm(output_dir)
            else:
                pending_requests.append(request)

        if not pending_requests:
            return resolved_paths

        effective_pool_size, device_tokens = self._get_ephemeral_lora_build_pool_spec()
        if self.ephemeral_lora_build_backend == "persistent":
            return self._build_missing_ephemeral_lora_requests_persistent(
                pending_requests=pending_requests,
                resolved_paths=resolved_paths,
                effective_pool_size=effective_pool_size,
                device_tokens=device_tokens,
            )
        if effective_pool_size <= 1 or len(pending_requests) <= 1:
            for request in pending_requests:
                resolved_paths[request["request_key"]] = self._build_missing_ephemeral_lora(
                    knowledge_id=request["knowledge_id"],
                    lora_variant=request["lora_variant"],
                    sample=request["sample"],
                    output_dir=request["output_dir"],
                )
            return resolved_paths

        self._record_ephemeral_lora_event(
            event="build_pool_start",
            knowledge_id=None,
            requested_pool_size=self.ephemeral_lora_build_pool_size,
            effective_pool_size=effective_pool_size,
            pending_count=len(pending_requests),
            cuda_devices=device_tokens,
        )

        with tempfile.TemporaryDirectory(prefix="knowledge_update_ephemeral_lora_build_") as temp_dir:
            future_to_request: dict[Any, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=effective_pool_size) as executor:
                for request_index, request in enumerate(pending_requests):
                    device_token = device_tokens[request_index % len(device_tokens)]
                    future = executor.submit(
                        self._build_missing_ephemeral_lora_subprocess,
                        request=request,
                        temp_dir=temp_dir,
                        device_token=device_token,
                    )
                    future_to_request[future] = request

                for future in as_completed(future_to_request):
                    request = future_to_request[future]
                    result = future.result()
                    self._record_ephemeral_lora_event(
                        event="build",
                        knowledge_id=result["knowledge_id"],
                        lora_variant=result.get("lora_variant"),
                        output_dir=result["output_dir"],
                        build_metadata=result["build_metadata"],
                        build_mode="parallel_pool",
                        build_device=result["device_token"],
                    )
                    self._cache_ephemeral_lora_norm(
                        result["output_dir"],
                        metadata=result.get("build_metadata"),
                    )
                    resolved_paths[request["request_key"]] = self._validate_ephemeral_lora_path(result["output_dir"])

        return resolved_paths

    def _build_missing_ephemeral_lora_requests_persistent(
        self,
        *,
        pending_requests: list[dict[str, Any]],
        resolved_paths: dict[str, str],
        effective_pool_size: int,
        device_tokens: list[str | None],
    ) -> dict[str, str]:
        workers = self._ensure_ephemeral_lora_build_workers(
            effective_pool_size=effective_pool_size,
            device_tokens=device_tokens,
        )
        if not workers:
            raise RuntimeError("Persistent ephemeral LoRA build backend did not create any workers.")

        self._record_ephemeral_lora_event(
            event="build_pool_start",
            knowledge_id=None,
            requested_pool_size=self.ephemeral_lora_build_pool_size,
            effective_pool_size=len(workers),
            pending_count=len(pending_requests),
            cuda_devices=[worker.device_token for worker in workers],
            build_backend="persistent",
        )

        future_to_request: dict[Any, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            for request_index, request in enumerate(pending_requests):
                worker = workers[request_index % len(workers)]
                future = executor.submit(
                    worker.build,
                    self._build_persistent_lora_worker_request(request),
                )
                future_to_request[future] = request

            for future in as_completed(future_to_request):
                request = future_to_request[future]
                result = future.result()
                output_dir = os.path.abspath(os.fspath(result["output_dir"]))
                metadata = result.get("build_metadata", {})
                self._record_ephemeral_lora_event(
                    event="build",
                    knowledge_id=result["knowledge_id"],
                    lora_variant=result.get("lora_variant"),
                    output_dir=output_dir,
                    build_metadata=metadata,
                    build_mode="persistent_pool",
                    build_device=result.get("device_token"),
                )
                self._cache_ephemeral_lora_norm(output_dir, metadata=metadata)
                resolved_paths[request["request_key"]] = self._validate_ephemeral_lora_path(output_dir)

        return resolved_paths

    def _ensure_ephemeral_lora_build_workers(
        self,
        *,
        effective_pool_size: int,
        device_tokens: list[str | None],
    ) -> list[_PersistentEphemeralLoraBuildWorker]:
        existing_workers = self._ephemeral_lora_build_workers
        if existing_workers is not None:
            alive_workers = [worker for worker in existing_workers if worker.process.poll() is None]
            if len(alive_workers) == len(existing_workers):
                return existing_workers
            for worker in existing_workers:
                worker.close()
            self._ephemeral_lora_build_workers = None

        repo_root = Path(__file__).resolve().parents[3]
        script_path = repo_root / "scripts" / "build_ephemeral_lora_adapters.py"
        base_lora_adapter_path = self._resolve_ephemeral_lora_build_base_lora_adapter_path()
        fallback_learning_rate = self.ephemeral_lora_build_fallback_learning_rate
        if fallback_learning_rate is None:
            fallback_learning_rate = max(self.ephemeral_lora_build_learning_rate / 5, 1e-6)

        command = [
            sys.executable,
            os.fspath(script_path),
            "--worker-jsonl",
            "--model-path",
            self.ephemeral_lora_build_model_path,
            "--steps",
            str(self.ephemeral_lora_build_steps),
            "--learning-rate",
            str(self.ephemeral_lora_build_learning_rate),
            "--fallback-learning-rate",
            str(float(fallback_learning_rate)),
            "--max-length",
            str(self.ephemeral_lora_build_max_length),
            "--lora-rank",
            str(self.ephemeral_lora_build_lora_rank),
            "--lora-alpha",
            str(self.ephemeral_lora_build_lora_alpha),
            "--lora-dropout",
            str(self.ephemeral_lora_build_lora_dropout),
            "--train-prompt-mode",
            self.ephemeral_lora_build_train_prompt_mode,
            "--chat-target-mode",
            self.ephemeral_lora_build_chat_target_mode,
            "--answer-field",
            self.ephemeral_lora_build_answer_field,
            "--target-modules",
            ",".join(self.ephemeral_lora_build_target_modules),
            "--dtype",
            self.ephemeral_lora_build_dtype,
            "--fallback-dtype",
            self.ephemeral_lora_build_fallback_dtype,
            "--lock-timeout-seconds",
            str(self.ephemeral_lora_build_timeout_seconds),
            "--lock-poll-interval-seconds",
            str(self.ephemeral_lora_build_poll_interval_seconds),
        ]
        if self.ephemeral_lora_build_use_rslora:
            command.append("--use-rslora")
        if self.ephemeral_lora_build_quality_max_last_loss is not None:
            command.extend(
                ["--quality-max-last-loss", str(self.ephemeral_lora_build_quality_max_last_loss)]
            )
        if self.ephemeral_lora_build_quality_min_l2_norm is not None:
            command.extend(["--quality-min-l2-norm", str(self.ephemeral_lora_build_quality_min_l2_norm)])
        if self.ephemeral_lora_build_quality_max_l2_norm is not None:
            command.extend(["--quality-max-l2-norm", str(self.ephemeral_lora_build_quality_max_l2_norm)])
        command.extend(
            [
                "--quality-retry-step-multiplier",
                str(self.ephemeral_lora_build_quality_retry_step_multiplier),
                "--quality-safe-lora-rank",
                str(self.ephemeral_lora_build_quality_safe_lora_rank),
                "--quality-safe-lora-alpha",
                str(self.ephemeral_lora_build_quality_safe_lora_alpha),
            ]
        )
        if self.ephemeral_lora_build_telemetry_csv:
            command.extend(["--telemetry-csv", self.ephemeral_lora_build_telemetry_csv])
        if self.ephemeral_lora_build_early_stop_loss is not None:
            command.extend(
                [
                    "--build-early-stop-loss",
                    str(self.ephemeral_lora_build_early_stop_loss),
                    "--build-early-stop-min-steps",
                    str(self.ephemeral_lora_build_early_stop_min_steps),
                ]
            )
        if self.ephemeral_lora_build_lora_recipe_pool:
            command.extend(["--lora-recipe-pool", self.ephemeral_lora_build_lora_recipe_pool])
        if base_lora_adapter_path:
            command.extend(["--base-lora-adapter-path", base_lora_adapter_path])
        if self.ephemeral_lora_build_gradient_clip_norm is not None:
            command.extend(["--gradient-clip-norm", str(self.ephemeral_lora_build_gradient_clip_norm)])

        base_env = os.environ.copy()
        existing_pythonpath = base_env.get("PYTHONPATH", "")
        base_env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else os.fspath(repo_root)
        )
        base_env["TOKENIZERS_PARALLELISM"] = "false"

        workers: list[_PersistentEphemeralLoraBuildWorker] = []
        for worker_index in range(max(1, effective_pool_size)):
            device_token = device_tokens[worker_index % len(device_tokens)]
            env = dict(base_env)
            if device_token is None:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(device_token)
            workers.append(
                _PersistentEphemeralLoraBuildWorker(
                    command=command,
                    cwd=os.fspath(repo_root),
                    env=env,
                    device_token=device_token,
                    timeout_seconds=self.ephemeral_lora_build_timeout_seconds,
                )
            )

        self._ephemeral_lora_build_workers = workers
        return workers

    def _build_persistent_lora_worker_request(self, request: dict[str, Any]) -> dict[str, Any]:
        fallback_learning_rate = self.ephemeral_lora_build_fallback_learning_rate
        if fallback_learning_rate is None:
            fallback_learning_rate = max(self.ephemeral_lora_build_learning_rate / 5, 1e-6)

        build_spec = self._resolve_ephemeral_lora_build_spec(
            knowledge_id=str(request["knowledge_id"]),
            lora_variant=str(request.get("lora_variant", KNOWLEDGE_LORA_VARIANT)),
        )
        return {
            "sample": request["sample"],
            "knowledge_id": str(request["knowledge_id"]),
            "lora_variant": str(request.get("lora_variant", KNOWLEDGE_LORA_VARIANT)),
            "output_dir": os.path.abspath(os.fspath(request["output_dir"])),
            "steps": int(build_spec.get("build_steps", self.ephemeral_lora_build_steps)),
            "learning_rate": self.ephemeral_lora_build_learning_rate,
            "fallback_learning_rate": float(fallback_learning_rate),
            "max_length": self.ephemeral_lora_build_max_length,
            "lora_rank": build_spec["lora_rank"],
            "lora_alpha": build_spec["lora_alpha"],
            "lora_dropout": build_spec["lora_dropout"],
            "use_rslora": bool(build_spec.get("use_rslora", self.ephemeral_lora_build_use_rslora)),
            "train_prompt_mode": self.ephemeral_lora_build_train_prompt_mode,
            "chat_target_mode": self.ephemeral_lora_build_chat_target_mode,
            "answer_field": self.ephemeral_lora_build_answer_field,
            "target_modules": list(build_spec["target_modules"]),
            "seed": build_spec["seed"],
            "gradient_clip_norm": self.ephemeral_lora_build_gradient_clip_norm,
            "lock_timeout_seconds": self.ephemeral_lora_build_timeout_seconds,
            "lock_poll_interval_seconds": self.ephemeral_lora_build_poll_interval_seconds,
            "quality_max_last_loss": self.ephemeral_lora_build_quality_max_last_loss,
            "quality_min_l2_norm": self.ephemeral_lora_build_quality_min_l2_norm,
            "quality_max_l2_norm": self.ephemeral_lora_build_quality_max_l2_norm,
            "quality_retry_step_multiplier": self.ephemeral_lora_build_quality_retry_step_multiplier,
            "quality_safe_lora_rank": self.ephemeral_lora_build_quality_safe_lora_rank,
            "quality_safe_lora_alpha": self.ephemeral_lora_build_quality_safe_lora_alpha,
            "telemetry_csv": self.ephemeral_lora_build_telemetry_csv or None,
            "build_spec_metadata": {
                "randomized_lora_config": build_spec.get("randomized_lora_config"),
                "randomization_mode": build_spec.get("randomization_mode"),
                "randomization_base_seed": build_spec.get("randomization_base_seed"),
                "lora_rank_candidates": build_spec.get("lora_rank_candidates"),
                "lora_alpha_candidates": build_spec.get("lora_alpha_candidates"),
                "lora_dropout_candidates": build_spec.get("lora_dropout_candidates"),
                "lora_recipe_pool_size": build_spec.get("lora_recipe_pool_size"),
                "lora_recipe_id": build_spec.get("lora_recipe_id"),
                "build_steps": build_spec.get("build_steps"),
            },
        }

    def _resolve_ephemeral_lora_build_spec(self, *, knowledge_id: str, lora_variant: str) -> dict[str, Any]:
        try:
            from scripts.build_ephemeral_lora_adapters import resolve_lora_build_spec
        except ImportError as exc:
            raise RuntimeError("Failed to import the ephemeral LoRA builder module.") from exc

        return resolve_lora_build_spec(
            knowledge_id=f"{knowledge_id}:{lora_variant}",
            seed=self.ephemeral_lora_build_random_base_seed,
            randomize_lora_config=self.ephemeral_lora_build_randomize_config,
            randomization_base_seed=self.ephemeral_lora_build_random_base_seed,
            randomization_mode=self.ephemeral_lora_build_randomization_mode,
            randomization_seed=self.ephemeral_lora_build_randomization_seed,
            lora_rank=self.ephemeral_lora_build_lora_rank,
            lora_alpha=self.ephemeral_lora_build_lora_alpha,
            lora_dropout=self.ephemeral_lora_build_lora_dropout,
            use_rslora=self.ephemeral_lora_build_use_rslora,
            lora_rank_candidates=self.ephemeral_lora_build_lora_rank_candidates,
            lora_alpha_candidates=self.ephemeral_lora_build_lora_alpha_candidates,
            lora_dropout_candidates=self.ephemeral_lora_build_lora_dropout_candidates,
            target_modules=list(self.ephemeral_lora_build_target_modules),
            lora_recipe_pool=self._load_ephemeral_lora_recipe_pool(),
        )

    def _load_ephemeral_lora_recipe_pool(self) -> list[dict[str, Any]] | None:
        if not self.ephemeral_lora_build_lora_recipe_pool:
            return None
        try:
            from scripts.build_ephemeral_lora_adapters import _load_lora_recipe_pool
        except ImportError as exc:
            raise RuntimeError("Failed to import the ephemeral LoRA builder module.") from exc
        return _load_lora_recipe_pool(self.ephemeral_lora_build_lora_recipe_pool)

    def _get_ephemeral_lora_build_pool_spec(self) -> tuple[int, list[str | None]]:
        device_tokens = list(self.ephemeral_lora_build_cuda_devices)
        if not device_tokens:
            device_tokens = [None]

        effective_pool_size = max(1, self.ephemeral_lora_build_pool_size)
        if any(device_token is not None for device_token in device_tokens):
            effective_pool_size = min(effective_pool_size, len(device_tokens))
        else:
            effective_pool_size = 1
        return effective_pool_size, device_tokens

    def _build_missing_ephemeral_lora_subprocess(
        self,
        *,
        request: dict[str, Any],
        temp_dir: str,
        device_token: str | None,
    ) -> dict[str, Any]:
        repo_root = Path(__file__).resolve().parents[3]
        script_path = repo_root / "scripts" / "build_ephemeral_lora_adapters.py"
        base_lora_adapter_path = self._resolve_ephemeral_lora_build_base_lora_adapter_path()
        sample_file = Path(temp_dir) / (
            f"{self._sanitize_ephemeral_lora_build_filename(request['knowledge_id'])}_{uuid4().hex}.json"
        )
        sample_file.write_text(json.dumps(request["sample"], ensure_ascii=False), encoding="utf-8")

        fallback_learning_rate = self.ephemeral_lora_build_fallback_learning_rate
        if fallback_learning_rate is None:
            fallback_learning_rate = max(self.ephemeral_lora_build_learning_rate / 5, 1e-6)

        command = [
            sys.executable,
            os.fspath(script_path),
            "--model-path",
            self.ephemeral_lora_build_model_path,
            "--sample-file",
            os.fspath(sample_file),
            "--knowledge-id",
            str(request["knowledge_id"]),
            "--lora-variant",
            str(request.get("lora_variant", KNOWLEDGE_LORA_VARIANT)),
            "--output-dir",
            os.fspath(request["output_dir"]),
            "--steps",
            str(self.ephemeral_lora_build_steps),
            "--learning-rate",
            str(self.ephemeral_lora_build_learning_rate),
            "--fallback-learning-rate",
            str(float(fallback_learning_rate)),
            "--max-length",
            str(self.ephemeral_lora_build_max_length),
            "--lora-rank",
            str(self.ephemeral_lora_build_lora_rank),
            "--lora-alpha",
            str(self.ephemeral_lora_build_lora_alpha),
            "--lora-dropout",
            str(self.ephemeral_lora_build_lora_dropout),
            "--train-prompt-mode",
            self.ephemeral_lora_build_train_prompt_mode,
            "--chat-target-mode",
            self.ephemeral_lora_build_chat_target_mode,
            "--answer-field",
            self.ephemeral_lora_build_answer_field,
            "--target-modules",
            ",".join(self.ephemeral_lora_build_target_modules),
            "--dtype",
            self.ephemeral_lora_build_dtype,
            "--fallback-dtype",
            self.ephemeral_lora_build_fallback_dtype,
            "--lock-timeout-seconds",
            str(self.ephemeral_lora_build_timeout_seconds),
            "--lock-poll-interval-seconds",
            str(self.ephemeral_lora_build_poll_interval_seconds),
        ]
        if self.ephemeral_lora_build_use_rslora:
            command.append("--use-rslora")
        if self.ephemeral_lora_build_quality_max_last_loss is not None:
            command.extend(["--quality-max-last-loss", str(self.ephemeral_lora_build_quality_max_last_loss)])
        if self.ephemeral_lora_build_quality_min_l2_norm is not None:
            command.extend(["--quality-min-l2-norm", str(self.ephemeral_lora_build_quality_min_l2_norm)])
        if self.ephemeral_lora_build_quality_max_l2_norm is not None:
            command.extend(["--quality-max-l2-norm", str(self.ephemeral_lora_build_quality_max_l2_norm)])
        command.extend(
            [
                "--quality-retry-step-multiplier",
                str(self.ephemeral_lora_build_quality_retry_step_multiplier),
                "--quality-safe-lora-rank",
                str(self.ephemeral_lora_build_quality_safe_lora_rank),
                "--quality-safe-lora-alpha",
                str(self.ephemeral_lora_build_quality_safe_lora_alpha),
            ]
        )
        if self.ephemeral_lora_build_telemetry_csv:
            command.extend(["--telemetry-csv", self.ephemeral_lora_build_telemetry_csv])
        if self.ephemeral_lora_build_early_stop_loss is not None:
            command.extend(
                [
                    "--build-early-stop-loss",
                    str(self.ephemeral_lora_build_early_stop_loss),
                    "--build-early-stop-min-steps",
                    str(self.ephemeral_lora_build_early_stop_min_steps),
                ]
            )
        if self.ephemeral_lora_build_lora_recipe_pool:
            command.extend(["--lora-recipe-pool", self.ephemeral_lora_build_lora_recipe_pool])
        if base_lora_adapter_path:
            command.extend(["--base-lora-adapter-path", base_lora_adapter_path])
        if self.ephemeral_lora_build_gradient_clip_norm is not None:
            command.extend(
                [
                    "--gradient-clip-norm",
                    str(self.ephemeral_lora_build_gradient_clip_norm),
                ]
            )
        if self.ephemeral_lora_build_randomize_config:
            command.extend(
                [
                    "--randomize-lora-config",
                    "--randomization-mode",
                    self.ephemeral_lora_build_randomization_mode,
                    "--randomization-seed",
                    str(self.ephemeral_lora_build_randomization_seed),
                ]
            )
            if self.ephemeral_lora_build_randomization_mode == "deterministic":
                command.extend(["--randomization-base-seed", str(self.ephemeral_lora_build_random_base_seed)])
            if self.ephemeral_lora_build_lora_rank_candidates:
                command.extend(
                    [
                        "--lora-rank-candidates",
                        ",".join(str(value) for value in self.ephemeral_lora_build_lora_rank_candidates),
                    ]
                )
            if self.ephemeral_lora_build_lora_alpha_candidates:
                command.extend(
                    [
                        "--lora-alpha-candidates",
                        ",".join(str(value) for value in self.ephemeral_lora_build_lora_alpha_candidates),
                    ]
                )
            if self.ephemeral_lora_build_lora_dropout_candidates:
                command.extend(
                    [
                        "--lora-dropout-candidates",
                        ",".join(str(value) for value in self.ephemeral_lora_build_lora_dropout_candidates),
                    ]
                )

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else os.fspath(repo_root)
        )
        env["TOKENIZERS_PARALLELISM"] = "false"
        if device_token is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(device_token)

        completed = subprocess.run(
            command,
            cwd=os.fspath(repo_root),
            env=env,
            text=True,
            capture_output=True,
            timeout=max(300.0, self.ephemeral_lora_build_timeout_seconds),
            check=False,
        )
        if completed.returncode != 0:
            stdout_tail = completed.stdout[-4000:] if completed.stdout else ""
            stderr_tail = completed.stderr[-4000:] if completed.stderr else ""
            raise RuntimeError(
                "Ephemeral LoRA build subprocess failed: "
                f"knowledge_id={request['knowledge_id']!r} device={device_token!r} "
                f"returncode={completed.returncode} stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
            )

        output_dir = os.path.abspath(os.fspath(request["output_dir"]))
        metadata = self._load_ephemeral_lora_build_metadata(output_dir)
        return {
            "request_key": request["request_key"],
            "knowledge_id": request["knowledge_id"],
            "lora_variant": request.get("lora_variant"),
            "output_dir": output_dir,
            "build_metadata": metadata,
            "device_token": device_token,
        }

    @staticmethod
    def _load_ephemeral_lora_build_metadata(output_dir: str) -> dict[str, Any]:
        metadata_path = Path(output_dir) / "knowledge_metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @staticmethod
    def _sanitize_ephemeral_lora_build_filename(value: str) -> str:
        sanitized = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
        sanitized = sanitized.strip("_")
        return sanitized[:120] or "knowledge_update_sample"

    @classmethod
    def _normalize_group_output_non_tensors_for_concat(
        cls,
        output: DataProto,
        *,
        all_non_tensor_keys: set[str],
    ) -> DataProto:
        batch_size = len(output)
        normalized = dict(getattr(output, "non_tensor_batch", None) or {})
        for key in all_non_tensor_keys:
            if key not in normalized:
                missing_values = np.empty(batch_size, dtype=object)
                missing_values[:] = None
                normalized[key] = missing_values
                continue
            values = np.asarray(normalized[key], dtype=object)
            value_count = int(values.shape[0]) if values.ndim > 0 else 1
            if value_count == batch_size:
                normalized[key] = values
            elif value_count == 1:
                normalized[key] = np.repeat(values.reshape(1), batch_size)
            elif value_count > 0 and batch_size % value_count == 0:
                normalized[key] = np.repeat(values, batch_size // value_count)
            elif value_count > batch_size:
                normalized[key] = values[:batch_size]
            else:
                padded_values = np.empty(batch_size, dtype=object)
                padded_values[:] = None
                padded_values[:value_count] = values
                normalized[key] = padded_values
        output.non_tensor_batch = normalized
        return output

    def _annotate_rollout_output_with_lora_loss_paths(
        self,
        *,
        output: DataProto,
        row_indices: list[int],
        plan_items: list[dict[str, Any]],
    ) -> DataProto:
        if not (self.enable_ephemeral_lora and self.use_ephemeral_lora_for_loss):
            return output
        output_batch_size = self._infer_batch_size(output)
        if output_batch_size in (None, 0):
            return output

        path_by_row_index: dict[int, str | None] = {}
        for plan_item in plan_items:
            lora_path = plan_item.get("lora_path")
            for row_index in plan_item.get("row_indices", []):
                path_by_row_index[int(row_index)] = lora_path

        normalized_row_indices = list(row_indices) if row_indices else list(range(int(output_batch_size)))
        paths = [path_by_row_index.get(int(row_index)) for row_index in normalized_row_indices]
        paths = self._pad_optional_values(paths, int(output_batch_size))

        non_tensor_batch = getattr(output, "non_tensor_batch", None)
        if non_tensor_batch is None:
            output.non_tensor_batch = {}
            non_tensor_batch = output.non_tensor_batch
        non_tensor_batch["ephemeral_lora_loss_path"] = np.array(paths, dtype=object)
        return output

    @classmethod
    def _concat_group_outputs_in_input_order(
        cls,
        grouped_outputs: list[DataProto],
        output_row_indices: list[int],
    ) -> DataProto:
        prepared_outputs: list[DataProto] = []
        merged_timing: dict[str, Any] = {}
        all_non_tensor_keys = {
            key
            for output in grouped_outputs
            for key in (getattr(output, "non_tensor_batch", None) or {}).keys()
        }
        for output in grouped_outputs:
            output.meta_info = dict(output.meta_info)
            timing = output.meta_info.pop("timing", None)
            if isinstance(timing, dict):
                merged_timing = cls._merge_timing_info(merged_timing, timing)
            prepared_outputs.append(
                cls._normalize_group_output_non_tensors_for_concat(
                    output,
                    all_non_tensor_keys=all_non_tensor_keys,
                )
            )

        merged_output = DataProto.concat(prepared_outputs)
        if merged_timing:
            merged_output.meta_info["timing"] = merged_timing

        if output_row_indices:
            reorder_indices = torch.tensor(
                np.argsort(np.asarray(output_row_indices, dtype=np.int64), kind="stable"),
                dtype=torch.long,
            )
            merged_output.reorder(reorder_indices)
        return merged_output

    @classmethod
    def _merge_timing_info(cls, merged_timing: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
        merged = dict(merged_timing)
        for key, value in timing.items():
            if key not in merged:
                merged[key] = value
                continue
            existing_value = merged[key]
            if isinstance(existing_value, dict) and isinstance(value, dict):
                merged[key] = cls._merge_timing_info(existing_value, value)
            elif isinstance(existing_value, (int, float)) and isinstance(value, (int, float)):
                merged[key] = existing_value + value
            else:
                merged[key] = value
        return merged

    def _validate_ephemeral_lora_path(self, lora_path: str) -> str:
        resolved_path = os.path.abspath(os.fspath(lora_path))
        missing_files = self._missing_ephemeral_lora_files(resolved_path)
        if missing_files:
            raise FileNotFoundError(
                "Ephemeral LoRA directory is missing required files: "
                f"path={resolved_path} missing={missing_files}"
            )
        return resolved_path

    @staticmethod
    def _missing_ephemeral_lora_files(lora_path: str) -> list[str]:
        required_files = ("adapter_config.json", "adapter_model.safetensors")
        return [required_file for required_file in required_files if not os.path.isfile(os.path.join(lora_path, required_file))]

    @classmethod
    def _ephemeral_lora_path_has_required_files(cls, lora_path: str | None) -> bool:
        if lora_path in (None, "", "None"):
            return False
        resolved_path = os.path.abspath(os.fspath(lora_path))
        return not cls._missing_ephemeral_lora_files(resolved_path)

    def _build_missing_ephemeral_lora(
        self,
        *,
        knowledge_id: str,
        lora_variant: str,
        sample: dict[str, Any],
        output_dir: str,
    ) -> str:
        try:
            from scripts.build_ephemeral_lora_adapters import (
                build_single_sample_with_lock,
                _write_build_metadata,
                resolve_dtype,
                resolve_lora_build_spec,
            )
        except ImportError as exc:
            raise RuntimeError("Failed to import the ephemeral LoRA builder module.") from exc

        output_dir_path = Path(output_dir).expanduser().resolve()
        output_dir_path.parent.mkdir(parents=True, exist_ok=True)
        learning_rate = self.ephemeral_lora_build_learning_rate
        fallback_learning_rate = self.ephemeral_lora_build_fallback_learning_rate
        if fallback_learning_rate is None:
            fallback_learning_rate = max(learning_rate / 5, 1e-6)
        fallback_dtype_name = self.ephemeral_lora_build_fallback_dtype.strip().lower()
        fallback_dtype = None if fallback_dtype_name == "none" else resolve_dtype(fallback_dtype_name)
        build_spec = resolve_lora_build_spec(
            knowledge_id=f"{knowledge_id}:{lora_variant}",
            seed=self.ephemeral_lora_build_random_base_seed,
            randomize_lora_config=self.ephemeral_lora_build_randomize_config,
            randomization_base_seed=self.ephemeral_lora_build_random_base_seed,
            randomization_mode=self.ephemeral_lora_build_randomization_mode,
            randomization_seed=self.ephemeral_lora_build_randomization_seed,
            lora_rank=self.ephemeral_lora_build_lora_rank,
            lora_alpha=self.ephemeral_lora_build_lora_alpha,
            lora_dropout=self.ephemeral_lora_build_lora_dropout,
            use_rslora=self.ephemeral_lora_build_use_rslora,
            lora_rank_candidates=self.ephemeral_lora_build_lora_rank_candidates,
            lora_alpha_candidates=self.ephemeral_lora_build_lora_alpha_candidates,
            lora_dropout_candidates=self.ephemeral_lora_build_lora_dropout_candidates,
            target_modules=list(self.ephemeral_lora_build_target_modules),
            lora_recipe_pool=self._load_ephemeral_lora_recipe_pool(),
        )
        base_lora_adapter_path = self._resolve_ephemeral_lora_build_base_lora_adapter_path()

        build_metadata = build_single_sample_with_lock(
            model_path=self.ephemeral_lora_build_model_path,
            base_lora_adapter_path=base_lora_adapter_path,
            sample={**sample, "_ephemeral_lora_answer_field": self.ephemeral_lora_build_answer_field},
            knowledge_id=knowledge_id,
            output_dir=output_dir_path,
            steps=int(build_spec.get("build_steps", self.ephemeral_lora_build_steps)),
            learning_rate=learning_rate,
            fallback_learning_rate=float(fallback_learning_rate),
            max_length=self.ephemeral_lora_build_max_length,
            lora_rank=build_spec["lora_rank"],
            lora_alpha=build_spec["lora_alpha"],
            lora_dropout=build_spec["lora_dropout"],
            use_rslora=bool(build_spec.get("use_rslora", self.ephemeral_lora_build_use_rslora)),
            target_modules=list(build_spec["target_modules"]),
            seed=build_spec["seed"],
            preferred_dtype=resolve_dtype(self.ephemeral_lora_build_dtype),
            fallback_dtype=fallback_dtype,
            gradient_clip_norm=self.ephemeral_lora_build_gradient_clip_norm,
            lora_variant=lora_variant,
            train_prompt_mode=self.ephemeral_lora_build_train_prompt_mode,
            chat_target_mode=self.ephemeral_lora_build_chat_target_mode,
            lock_timeout_seconds=self.ephemeral_lora_build_timeout_seconds,
            lock_poll_interval_seconds=self.ephemeral_lora_build_poll_interval_seconds,
            quality_max_last_loss=self.ephemeral_lora_build_quality_max_last_loss,
            quality_min_l2_norm=self.ephemeral_lora_build_quality_min_l2_norm,
            quality_max_l2_norm=self.ephemeral_lora_build_quality_max_l2_norm,
            quality_retry_step_multiplier=self.ephemeral_lora_build_quality_retry_step_multiplier,
            quality_safe_lora_rank=self.ephemeral_lora_build_quality_safe_lora_rank,
            quality_safe_lora_alpha=self.ephemeral_lora_build_quality_safe_lora_alpha,
            telemetry_csv=self.ephemeral_lora_build_telemetry_csv or None,
        )
        build_metadata.update(
            {
                "lora_variant": lora_variant,
                "randomized_lora_config": build_spec["randomized_lora_config"],
                "randomization_mode": build_spec.get("randomization_mode"),
                "randomization_base_seed": build_spec.get("randomization_base_seed"),
                "lora_rank_candidates": build_spec.get("lora_rank_candidates"),
                "lora_alpha_candidates": build_spec.get("lora_alpha_candidates"),
                "lora_dropout_candidates": build_spec.get("lora_dropout_candidates"),
                "use_rslora": build_spec.get("use_rslora"),
                "lora_scale": build_spec.get("lora_scale"),
                "lora_recipe_pool_size": build_spec.get("lora_recipe_pool_size"),
                "lora_recipe_id": build_spec.get("lora_recipe_id"),
                "build_steps": build_spec.get("build_steps"),
            }
        )
        _write_build_metadata(output_dir_path, build_metadata)
        self._record_ephemeral_lora_event(
            event="build",
            knowledge_id=knowledge_id,
            lora_variant=lora_variant,
            output_dir=str(output_dir_path),
            build_metadata=build_metadata,
        )
        return self._validate_ephemeral_lora_path(str(output_dir_path))

    def _extract_build_sample_from_group(self, *, group: dict[str, Any], knowledge_id: str) -> dict[str, Any]:
        variants: list[dict[str, Any]] = []
        seen_variant_keys: set[tuple[str, str]] = set()
        base_sample: dict[str, Any] | None = None

        for extra_info in group.get("extra_infos", []):
            if not isinstance(extra_info, dict):
                continue
            sample = {
                "knowledge_id": knowledge_id,
                "title": extra_info.get("title", ""),
                "category": extra_info.get("category", ""),
                "subcategory": extra_info.get("subcategory", ""),
                "context": extra_info.get("context", ""),
                "question": extra_info.get("question", ""),
                "answer": extra_info.get("answer", ""),
            }
            if sample["question"] and sample["answer"]:
                if base_sample is None:
                    base_sample = dict(sample)
                variant_key = (str(sample["question"]), str(sample["answer"]))
                if variant_key not in seen_variant_keys:
                    seen_variant_keys.add(variant_key)
                    variant = dict(sample)
                    for optional_key in ("rewrite_variant_id", "rewrite_style", "sample_hash", "rewrite_sample_hash"):
                        if optional_key in extra_info:
                            variant[optional_key] = extra_info[optional_key]
                    variants.append(variant)

            extra_variants = extra_info.get("variants")
            if isinstance(extra_variants, list):
                for extra_variant in extra_variants:
                    if not isinstance(extra_variant, dict):
                        continue
                    question = str(extra_variant.get("question", "")).strip()
                    answer = str(extra_variant.get("answer", "")).strip()
                    if not question or not answer:
                        continue
                    if base_sample is None:
                        base_sample = {
                            "knowledge_id": knowledge_id,
                            "title": extra_variant.get("title", extra_info.get("title", "")),
                            "category": extra_variant.get("category", extra_info.get("category", "")),
                            "subcategory": extra_variant.get("subcategory", extra_info.get("subcategory", "")),
                            "context": extra_variant.get("context", extra_info.get("context", "")),
                            "question": question,
                            "answer": answer,
                        }
                    variant_key = (question, answer)
                    if variant_key in seen_variant_keys:
                        continue
                    seen_variant_keys.add(variant_key)
                    variant = dict(extra_variant)
                    variant.setdefault("knowledge_id", knowledge_id)
                    variants.append(variant)

        if base_sample is not None:
            if len(variants) > 1:
                base_sample["variants"] = variants
            return base_sample

        raise ValueError(
            "KnowledgeUpdatePPOTrainer cannot build a missing ephemeral LoRA because the batch "
            f"does not include enough QA metadata for knowledge_id={knowledge_id}."
        )

    def _verify_ephemeral_lora_state(
        self,
        *,
        expected_staged: Any,
        expected_loaded: Any,
        phase: str,
    ) -> None:
        controller = self._get_ephemeral_lora_controller()
        if not hasattr(controller, "get_ephemeral_lora_state"):
            return

        worker_states = self._normalize_worker_results(controller.get_ephemeral_lora_state())
        if not worker_states:
            return

        mismatches: list[dict[str, Any]] = []
        expected_staged_ids = self._normalize_optional_id_set(expected_staged)
        expected_loaded_ids = self._normalize_optional_id_set(expected_loaded)
        for worker_state in worker_states:
            actual_staged = self._normalize_optional_id_set(
                worker_state.get("staged_knowledge_ids", worker_state.get("staged_knowledge_id"))
            )
            actual_loaded_ids = self._normalize_optional_id_set(
                worker_state.get("loaded_knowledge_ids", worker_state.get("loaded_knowledge_id"))
            )
            if actual_staged != expected_staged_ids or actual_loaded_ids != expected_loaded_ids:
                mismatches.append(
                    {
                        "worker_rank": worker_state.get("worker_rank"),
                        "expected_staged": sorted(expected_staged_ids),
                        "actual_staged": sorted(actual_staged),
                        "expected_loaded": sorted(expected_loaded_ids),
                        "actual_loaded": sorted(actual_loaded_ids),
                    }
                )

        self._record_ephemeral_lora_event(
            event="verify",
            knowledge_id=self._first_optional_id(expected_loaded_ids) or self._first_optional_id(expected_staged_ids),
            phase=phase,
            worker_states=worker_states,
        )
        if mismatches:
            raise RuntimeError(
                "Ephemeral LoRA state verification failed "
                f"during `{phase}` with mismatches: {json.dumps(mismatches, ensure_ascii=False)}"
            )

    def _verify_ephemeral_lora_state_for_replicas(
        self,
        *,
        replica_indices: list[int],
        expected_staged: Any,
        expected_loaded: Any,
        phase: str,
    ) -> None:
        controller = self._get_ephemeral_lora_controller()
        if not hasattr(controller, "get_ephemeral_lora_state"):
            return
        if controller is not getattr(self, "async_rollout_manager", None):
            self._verify_ephemeral_lora_state(
                expected_staged=expected_staged,
                expected_loaded=expected_loaded,
                phase=phase,
            )
            return

        worker_states = self._normalize_worker_results(controller.get_ephemeral_lora_state(replica_indices=replica_indices))
        if not worker_states:
            return

        expected_staged_ids = self._normalize_optional_id_set(expected_staged)
        expected_loaded_ids = self._normalize_optional_id_set(expected_loaded)
        mismatches: list[dict[str, Any]] = []
        for worker_state in worker_states:
            actual_staged = self._normalize_optional_id_set(
                worker_state.get("staged_knowledge_ids", worker_state.get("staged_knowledge_id"))
            )
            actual_loaded_ids = self._normalize_optional_id_set(
                worker_state.get("loaded_knowledge_ids", worker_state.get("loaded_knowledge_id"))
            )
            if actual_staged != expected_staged_ids or actual_loaded_ids != expected_loaded_ids:
                mismatches.append(
                    {
                        "worker_rank": worker_state.get("worker_rank"),
                        "expected_staged": sorted(expected_staged_ids),
                        "actual_staged": sorted(actual_staged),
                        "expected_loaded": sorted(expected_loaded_ids),
                        "actual_loaded": sorted(actual_loaded_ids),
                    }
                )

        self._record_ephemeral_lora_event(
            event="verify",
            knowledge_id=self._first_optional_id(expected_loaded_ids) or self._first_optional_id(expected_staged_ids),
            phase=phase,
            replica_indices=replica_indices,
            worker_states=worker_states,
        )
        if mismatches:
            raise RuntimeError(
                "Ephemeral LoRA state verification failed "
                f"during `{phase}` with mismatches: {json.dumps(mismatches, ensure_ascii=False)}"
            )

    def _verify_ephemeral_lora_states_by_replica(
        self,
        *,
        expected_states_by_replica: dict[int, dict[str, Any]],
        phase: str,
    ) -> None:
        controller = self._get_ephemeral_lora_controller()
        if not hasattr(controller, "get_ephemeral_lora_state"):
            return
        if controller is not getattr(self, "async_rollout_manager", None):
            raise RuntimeError(
                "KnowledgeUpdatePPOTrainer per-replica verification requires async_rollout_manager targeting support."
            )

        replica_indices = sorted(expected_states_by_replica)
        worker_states = self._normalize_worker_results(controller.get_ephemeral_lora_state(replica_indices=replica_indices))
        if not worker_states:
            return

        worker_states_by_rank: dict[int, dict[str, Any]] = {}
        unresolved_worker_states: list[dict[str, Any]] = []
        for worker_state in worker_states:
            worker_rank = worker_state.get("worker_rank", worker_state.get("replica_index"))
            if worker_rank in (None, "", "None"):
                unresolved_worker_states.append(worker_state)
                continue
            worker_states_by_rank[int(worker_rank)] = worker_state

        for replica_index, worker_state in zip(replica_indices, unresolved_worker_states, strict=False):
            worker_state = dict(worker_state)
            worker_state.setdefault("worker_rank", replica_index)
            worker_states_by_rank[int(replica_index)] = worker_state

        mismatches: list[dict[str, Any]] = []
        for replica_index, expected_state in expected_states_by_replica.items():
            worker_state = worker_states_by_rank.get(replica_index)
            if worker_state is None:
                mismatches.append(
                    {
                        "worker_rank": replica_index,
                        "expected_staged": sorted(
                            self._normalize_optional_id_set(expected_state.get("expected_staged"))
                        ),
                        "actual_staged": [],
                        "expected_loaded": sorted(
                            self._normalize_optional_id_set(expected_state.get("expected_loaded"))
                        ),
                        "actual_loaded": [],
                    }
                )
                continue
            actual_staged = self._normalize_optional_id_set(
                worker_state.get("staged_knowledge_ids", worker_state.get("staged_knowledge_id"))
            )
            actual_loaded = self._normalize_optional_id_set(
                worker_state.get("loaded_knowledge_ids", worker_state.get("loaded_knowledge_id"))
            )
            expected_staged_ids = self._normalize_optional_id_set(expected_state.get("expected_staged"))
            expected_loaded = self._normalize_optional_id_set(expected_state.get("expected_loaded"))
            if actual_staged != expected_staged_ids or actual_loaded != expected_loaded:
                mismatches.append(
                    {
                        "worker_rank": replica_index,
                        "expected_staged": sorted(expected_staged_ids),
                        "actual_staged": sorted(actual_staged),
                        "expected_loaded": sorted(expected_loaded),
                        "actual_loaded": sorted(actual_loaded),
                    }
                )

        self._record_ephemeral_lora_event(
            event="verify",
            knowledge_id=None,
            phase=phase,
            replica_indices=replica_indices,
            worker_states=worker_states,
        )
        if mismatches:
            raise RuntimeError(
                "Ephemeral LoRA state verification failed "
                f"during `{phase}` with mismatches: {json.dumps(mismatches, ensure_ascii=False)}"
            )

    def _require_async_rollout_targeting(self, strategy_name: str):
        manager = getattr(self, "async_rollout_manager", None)
        if manager is None:
            raise RuntimeError(
                f"KnowledgeUpdatePPOTrainer `{strategy_name}` requires async_rollout_manager."
            )
        required_methods = (
            "stage_ephemeral_lora",
            "load_staged_ephemeral_lora",
            "clear_ephemeral_lora",
            "get_ephemeral_lora_state",
        )
        missing_methods = [method_name for method_name in required_methods if not hasattr(manager, method_name)]
        if missing_methods:
            raise RuntimeError(
                f"KnowledgeUpdatePPOTrainer `{strategy_name}` requires async_rollout_manager methods: {missing_methods}"
            )
        return manager

    def _get_async_rollout_server_ids(self) -> list[str]:
        manager = self._require_async_rollout_targeting("parallel_waves")
        server_ids = getattr(manager, "server_addresses", None)
        if server_ids is None:
            return []
        return [self._normalize_optional_id(server_id) for server_id in server_ids if self._normalize_optional_id(server_id) is not None]

    def _get_ephemeral_lora_controller(self):
        if hasattr(self, "async_rollout_manager") and hasattr(self.async_rollout_manager, "stage_ephemeral_lora"):
            return self.async_rollout_manager
        return self.actor_rollout_wg

    def _record_ephemeral_lora_event(self, *, event: str, knowledge_id: str | None, **payload: Any) -> None:
        record = {
            "event": event,
            "knowledge_id": self._normalize_optional_id(knowledge_id),
            **payload,
        }
        self._ephemeral_lora_events.append(record)
        if len(self._ephemeral_lora_events) > 128:
            self._ephemeral_lora_events.pop(0)
        if self.log_ephemeral_lora_events:
            logger.info("knowledge_update_ephemeral_lora: %s", json.dumps(record, ensure_ascii=False, default=str))

    def _log_memory_diagnostics(
        self,
        *,
        phase: str,
        wave_index: int | None = None,
        plan_items: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.memory_diagnostics:
            return

        worker_diagnostics = None
        controller = getattr(self, "async_rollout_manager", None)
        if controller is not None and hasattr(controller, "get_memory_diagnostics"):
            try:
                worker_diagnostics = self._normalize_worker_results(controller.get_memory_diagnostics())
            except Exception as exc:  # pragma: no cover - best effort only
                worker_diagnostics = [{"error": str(exc)}]

        payload = {
            "phase": phase,
            "wave_index": wave_index,
            "trainer_pid": os.getpid(),
            "trainer_rss_gb": self._read_current_rss_gb(),
            "trainer_gc_counts": list(gc.get_count()),
            "trainer_cuda_allocated_gb": (
                torch.cuda.memory_allocated() / (1024.0**3) if torch.cuda.is_available() else None
            ),
            "trainer_cuda_reserved_gb": (
                torch.cuda.memory_reserved() / (1024.0**3) if torch.cuda.is_available() else None
            ),
            "ephemeral_event_cache_size": len(self._ephemeral_lora_events),
            "planned_item_count": len(plan_items or []),
            "knowledge_ids": [item.get("knowledge_id") for item in (plan_items or [])],
            "replica_indices": [item.get("replica_index") for item in (plan_items or [])],
            "preferred_server_ids": [item.get("preferred_server_id") for item in (plan_items or [])],
            "worker_diagnostics": worker_diagnostics,
        }
        logger.info("knowledge_update_memory_diagnostics: %s", json.dumps(payload, ensure_ascii=False, default=str))

    @classmethod
    def _normalize_worker_results(cls, worker_result: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                if (
                    "worker_rank" in item
                    or "staged_knowledge_id" in item
                    or "staged_knowledge_ids" in item
                    or "loaded_knowledge_id" in item
                    or "loaded_knowledge_ids" in item
                ):
                    normalized.append(item)
                    return
                for nested_key in ("server_states", "server_results", "state", "result"):
                    if nested_key in item:
                        visit(item[nested_key])
                return
            if isinstance(item, np.ndarray):
                for nested_item in item.tolist():
                    visit(nested_item)
                return
            if isinstance(item, (list, tuple)):
                for nested_item in item:
                    visit(nested_item)

        visit(worker_result)
        return normalized

    @staticmethod
    def _infer_batch_size(gen_batch) -> int | None:
        if gen_batch is None:
            return None
        non_tensor_batch = getattr(gen_batch, "non_tensor_batch", None)
        if non_tensor_batch:
            for value in non_tensor_batch.values():
                if isinstance(value, np.ndarray):
                    return int(len(value))
                if isinstance(value, (list, tuple)):
                    return int(len(value))
        batch = getattr(gen_batch, "batch", None)
        if batch:
            for value in batch.values():
                if hasattr(value, "shape") and len(value.shape) > 0:
                    return int(value.shape[0])
        return None

    @staticmethod
    def _normalize_lora_variant(value: Any) -> str:
        normalized = str(value or KNOWLEDGE_LORA_VARIANT).strip().lower().replace("-", "_")
        if normalized not in SUPPORTED_LORA_VARIANTS:
            return KNOWLEDGE_LORA_VARIANT
        return normalized

    @classmethod
    def _normalize_lora_variants(cls, value: Any) -> list[str]:
        if value is None:
            return [KNOWLEDGE_LORA_VARIANT]
        if isinstance(value, str):
            raw_items = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, Sequence):
            raw_items = [str(item).strip() for item in value if str(item).strip()]
        else:
            raw_items = [str(value).strip()]
        normalized = [cls._normalize_lora_variant(item) for item in raw_items]
        if KNOWLEDGE_LORA_VARIANT not in normalized:
            normalized.insert(0, KNOWLEDGE_LORA_VARIANT)
        return list(dict.fromkeys(normalized))

    @classmethod
    def _normalize_lora_variant_weights(
        cls,
        value: Any,
        *,
        enabled_variants: list[str],
    ) -> dict[str, float]:
        weights = {variant_name: 1.0 for variant_name in enabled_variants}
        if value is None:
            return weights
        if isinstance(value, Mapping):
            iterator = value.items()
        elif isinstance(value, str):
            iterator = []
            for item in value.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" not in item:
                    iterator.append((item, 1.0))
                    continue
                variant_name, raw_weight = item.split(":", 1)
                iterator.append((variant_name, raw_weight))
        else:
            raise TypeError(
                "trainer.knowledge_update.lora_variant_weights must be a mapping or comma-separated string."
            )
        for variant_name, raw_weight in iterator:
            normalized_variant = cls._normalize_lora_variant(variant_name)
            try:
                weights[normalized_variant] = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid lora variant weight for {variant_name!r}: {raw_weight!r}") from exc
        return weights

    @staticmethod
    def _normalize_optional_id(value: Any) -> str | None:
        if value in (None, "", "None"):
            return None
        return str(value)

    @classmethod
    def _normalize_optional_id_set(cls, value: Any) -> set[str | None]:
        if value is None:
            return set()
        if isinstance(value, np.ndarray):
            items = value.tolist()
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]
        normalized = {cls._normalize_optional_id(item) for item in items}
        normalized.discard(None)
        return normalized

    @staticmethod
    def _first_optional_id(values: set[str | None]) -> str | None:
        if not values:
            return None
        return sorted(value for value in values if value is not None)[0]

    @staticmethod
    def _expand_batch_field(value: Any) -> list[Any]:
        if value is None:
            return []
        return KnowledgeUpdatePPOTrainer._to_python_list(value)

    @staticmethod
    def _pad_optional_values(values: list[Any], target_length: int, fill_value: Any = None) -> list[Any]:
        padded_values = list(values)
        if len(padded_values) < target_length:
            padded_values.extend([fill_value] * (target_length - len(padded_values)))
        return padded_values

    @staticmethod
    def _normalize_optional_path(value: Any) -> str | None:
        if value in (None, "", "None"):
            return None
        return os.path.abspath(os.fspath(value))

    def _actor_model_uses_lora(self) -> bool:
        actor_model_config = getattr(getattr(self.config, "actor_rollout_ref", None), "model", None)
        if actor_model_config is None:
            return False

        lora_rank = getattr(actor_model_config, "lora_rank", 0) or 0
        lora_config = getattr(actor_model_config, "lora", None)
        if lora_rank <= 0 and lora_config is not None and hasattr(lora_config, "get"):
            lora_rank = lora_config.get("rank", 0) or 0

        lora_adapter_path = self._normalize_optional_path(getattr(actor_model_config, "lora_adapter_path", None))
        return int(lora_rank) > 0 or lora_adapter_path is not None

    def _resolve_latest_actor_lora_adapter_path(self) -> str | None:
        explicit_policy_path = self.ephemeral_lora_build_policy_lora_adapter_path
        if explicit_policy_path and self._ephemeral_lora_path_has_required_files(explicit_policy_path):
            return explicit_policy_path

        trainer_default_local_dir = getattr(getattr(self.config, "trainer", None), "default_local_dir", None)
        checkpoint_root = self._normalize_optional_path(trainer_default_local_dir)
        if checkpoint_root is not None:
            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

            latest_checkpoint = find_latest_ckpt_path(checkpoint_root)
            if latest_checkpoint:
                latest_adapter_path = os.path.join(latest_checkpoint, "actor", "lora_adapter")
                if self._ephemeral_lora_path_has_required_files(latest_adapter_path):
                    return os.path.abspath(os.fspath(latest_adapter_path))

        actor_model_config = getattr(getattr(self.config, "actor_rollout_ref", None), "model", None)
        configured_lora_adapter_path = self._normalize_optional_path(
            getattr(actor_model_config, "lora_adapter_path", None)
        )
        if configured_lora_adapter_path and self._ephemeral_lora_path_has_required_files(configured_lora_adapter_path):
            return configured_lora_adapter_path
        return explicit_policy_path

    def _resolve_ephemeral_lora_build_base_lora_adapter_path(self) -> str | None:
        if self.ephemeral_lora_build_source == "base":
            return None
        if self.ephemeral_lora_build_source != "policy":
            raise ValueError(
                "Unsupported `trainer.knowledge_update.ephemeral_lora_build_source`: "
                f"{self.ephemeral_lora_build_source!r}. Expected 'base' or 'policy'."
            )
        if not self._actor_model_uses_lora():
            raise ValueError(
                "`trainer.knowledge_update.ephemeral_lora_build_source=policy` currently requires the actor policy "
                "itself to be LoRA-based. Full-parameter actor checkpoints are not yet exported in a HuggingFace "
                "adapter format that the ephemeral LoRA builder can consume directly."
            )

        policy_lora_adapter_path = self._resolve_latest_actor_lora_adapter_path()
        if policy_lora_adapter_path and self._ephemeral_lora_path_has_required_files(policy_lora_adapter_path):
            return os.path.abspath(os.fspath(policy_lora_adapter_path))

        raise FileNotFoundError(
            "Failed to resolve a policy LoRA adapter for ephemeral LoRA builds. "
            "Set `trainer.knowledge_update.ephemeral_lora_build_policy_lora_adapter_path`, "
            "or enable periodic checkpoint saving so the trainer can reuse the latest `actor/lora_adapter`."
        )

    def _resolve_ephemeral_lora_build_model_path(self) -> str:
        explicit_model_path = self.knowledge_update_config.get("ephemeral_lora_build_model_path", None)
        if explicit_model_path:
            return os.path.abspath(os.fspath(explicit_model_path))

        actor_model_config = getattr(getattr(self.config, "actor_rollout_ref", None), "model", None)
        actor_model_path = getattr(actor_model_config, "path", None)
        if actor_model_path:
            return os.path.abspath(os.fspath(actor_model_path))
        return ""

    @staticmethod
    def _resolve_ephemeral_lora_build_cuda_devices(value: Any) -> list[str | None]:
        if value is None:
            if torch.cuda.is_available():
                return [str(device_index) for device_index in range(torch.cuda.device_count())]
            return [None]

        if isinstance(value, str):
            raw_devices = [device.strip() for device in value.split(",")]
        elif isinstance(value, Sequence):
            raw_devices = [str(device).strip() if device is not None else "" for device in value]
        else:
            raise TypeError(
                "`trainer.knowledge_update.ephemeral_lora_build_cuda_devices` must be a string or list of strings."
            )

        normalized_devices: list[str | None] = []
        for raw_device in raw_devices:
            if raw_device in ("", "none", "None", "cpu", "CPU"):
                normalized_devices.append(None)
                continue
            normalized_devices.append(raw_device)

        # Preserve repeated device tokens intentionally: users may request multiple build
        # workers per GPU, e.g. ["0", ..., "7", "0", ..., "7"] for 16 workers on 8 GPUs.
        return normalized_devices or ([None] if not torch.cuda.is_available() else [str(device_index) for device_index in range(torch.cuda.device_count())])

    @staticmethod
    def _normalize_target_modules(value: Any) -> list[str]:
        if value is None:
            return list(DEFAULT_EPHEMERAL_LORA_TARGET_MODULES)
        if isinstance(value, str):
            modules = [module.strip() for module in value.split(",") if module.strip()]
            return modules or list(DEFAULT_EPHEMERAL_LORA_TARGET_MODULES)
        if isinstance(value, Sequence) and not isinstance(value, str):
            modules = [str(module).strip() for module in value if str(module).strip()]
            return modules or list(DEFAULT_EPHEMERAL_LORA_TARGET_MODULES)
        raise TypeError(
            "`trainer.knowledge_update.ephemeral_lora_build_target_modules` must be a string or list of strings."
        )

    @staticmethod
    def _read_current_rss_gb() -> float | None:
        try:
            with open("/proc/self/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        rss_kb = float(line.split()[1])
                        return rss_kb / (1024.0 * 1024.0)
        except OSError:
            return None
        return None

    @staticmethod
    def _normalize_int_candidates(value: Any) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, Sequence) and not isinstance(value, str):
            items = [item for item in value]
        else:
            raise TypeError("LoRA integer candidate config must be a string or list.")
        if not items:
            return None
        return [int(item) for item in items]

    @staticmethod
    def _normalize_float_candidates(value: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, Sequence) and not isinstance(value, str):
            items = [item for item in value]
        else:
            raise TypeError("LoRA float candidate config must be a string or list.")
        if not items:
            return None
        return [float(item) for item in items]

    @staticmethod
    def _normalize_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {"none", "null"}:
                return None
            return float(stripped)
        return float(value)

    @staticmethod
    def _normalize_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"1", "true", "yes", "y", "on"}:
                return True
            if stripped in {"0", "false", "no", "n", "off", "", "none", "null"}:
                return False
            raise ValueError(f"Cannot parse boolean config value: {value!r}")
        return bool(value)
