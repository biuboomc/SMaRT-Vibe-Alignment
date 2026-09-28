from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from verl.protocol import DataProto
from verl.trainer.ppo.knowledge_update_trainer import (
    DEFAULT_EPHEMERAL_LORA_TARGET_MODULES,
    KnowledgeUpdatePPOTrainer,
    KNOWLEDGE_LORA_VARIANT,
    NO_OP_LORA_VARIANT,
)


class DummyWorkerGroup:
    def __init__(self):
        self.stage_calls = []
        self.load_calls = []
        self.clear_calls = []
        self._staged_knowledge_ids = []
        self._loaded_knowledge_id = None
        self._refresh_state_response()

    def _refresh_state_response(self):
        staged_knowledge_ids = list(self._staged_knowledge_ids)
        self.state_response = [
            {
                "worker_rank": 0,
                "staged_knowledge_id": staged_knowledge_ids[0] if len(staged_knowledge_ids) == 1 else None,
                "staged_knowledge_ids": staged_knowledge_ids,
                "loaded_knowledge_id": self._loaded_knowledge_id,
            }
        ]

    def stage_ephemeral_lora(self, **kwargs):
        self.stage_calls.append(kwargs)
        knowledge_id = kwargs.get("knowledge_id")
        eager_load = kwargs.get("eager_load", False)
        if knowledge_id is not None and knowledge_id not in self._staged_knowledge_ids:
            self._staged_knowledge_ids.append(knowledge_id)
        self._loaded_knowledge_id = knowledge_id if eager_load else self._loaded_knowledge_id
        self._refresh_state_response()
        return {"ok": True}

    def load_staged_ephemeral_lora(self, knowledge_id=None):
        self.load_calls.append({"knowledge_id": knowledge_id})
        if knowledge_id is None:
            self._loaded_knowledge_id = self._staged_knowledge_ids[0] if self._staged_knowledge_ids else None
        elif knowledge_id in self._staged_knowledge_ids:
            self._loaded_knowledge_id = knowledge_id
        self._refresh_state_response()
        return {"ok": True}

    def clear_ephemeral_lora(self, **kwargs):
        self.clear_calls.append(kwargs)
        knowledge_id = kwargs.get("knowledge_id")
        if knowledge_id in (None, "", "None"):
            self._staged_knowledge_ids = []
            self._loaded_knowledge_id = None
        else:
            self._staged_knowledge_ids = [item for item in self._staged_knowledge_ids if item != knowledge_id]
            if self._loaded_knowledge_id == knowledge_id:
                self._loaded_knowledge_id = None
        self._refresh_state_response()
        return {"ok": True}

    def get_ephemeral_lora_state(self):
        return self.state_response


class DummyAsyncRolloutManager(DummyWorkerGroup):
    def __init__(self, replica_count: int = 1):
        self.server_addresses = [f"server-{replica_index}" for replica_index in range(replica_count)]
        self._worker_states = {
            replica_index: {
                "worker_rank": replica_index,
                "staged_knowledge_ids": [],
                "loaded_knowledge_id": None,
            }
            for replica_index in range(replica_count)
        }
        self.stage_calls = []
        self.load_calls = []
        self.clear_calls = []
        self.generate_calls = []
        self._refresh_state_response()

    def _refresh_state_response(self):
        self.state_response = [
            {
                "worker_rank": replica_index,
                "staged_knowledge_id": state["staged_knowledge_ids"][0] if len(state["staged_knowledge_ids"]) == 1 else None,
                "staged_knowledge_ids": list(state["staged_knowledge_ids"]),
                "loaded_knowledge_id": state["loaded_knowledge_id"],
            }
            for replica_index, state in sorted(self._worker_states.items())
        ]

    def stage_ephemeral_lora(self, **kwargs):
        self.stage_calls.append(kwargs)
        knowledge_id = kwargs.get("knowledge_id")
        eager_load = kwargs.get("eager_load", False)
        replica_indices = kwargs.get("replica_indices")
        if replica_indices is None:
            replica_indices = list(self._worker_states)
        for replica_index in replica_indices:
            state = self._worker_states[replica_index]
            if knowledge_id is not None and knowledge_id not in state["staged_knowledge_ids"]:
                state["staged_knowledge_ids"].append(knowledge_id)
            if eager_load:
                state["loaded_knowledge_id"] = knowledge_id
        self._refresh_state_response()
        return {"ok": True}

    def load_staged_ephemeral_lora(self, knowledge_id=None, replica_indices=None):
        call = {"knowledge_id": knowledge_id}
        if replica_indices is not None:
            call["replica_indices"] = replica_indices
        self.load_calls.append(call)
        if replica_indices is None:
            replica_indices = list(self._worker_states)
        for replica_index in replica_indices:
            state = self._worker_states[replica_index]
            if knowledge_id is None:
                state["loaded_knowledge_id"] = state["staged_knowledge_ids"][0] if state["staged_knowledge_ids"] else None
            elif knowledge_id in state["staged_knowledge_ids"]:
                state["loaded_knowledge_id"] = knowledge_id
        self._refresh_state_response()
        return {"ok": True}

    def clear_ephemeral_lora(self, **kwargs):
        self.clear_calls.append(kwargs)
        knowledge_id = kwargs.get("knowledge_id")
        replica_indices = kwargs.get("replica_indices")
        if replica_indices is None:
            replica_indices = list(self._worker_states)
        for replica_index in replica_indices:
            state = self._worker_states[replica_index]
            if knowledge_id in (None, "", "None"):
                state["staged_knowledge_ids"] = []
                state["loaded_knowledge_id"] = None
            else:
                state["staged_knowledge_ids"] = [
                    item for item in state["staged_knowledge_ids"] if item != knowledge_id
                ]
                if state["loaded_knowledge_id"] == knowledge_id:
                    state["loaded_knowledge_id"] = None
        self._refresh_state_response()
        return {"ok": True}

    def get_ephemeral_lora_state(self, replica_indices=None):
        if replica_indices is None:
            return self.state_response
        return [self.state_response[replica_index] for replica_index in replica_indices]

    def generate_sequences(self, gen_batch):
        row_ids = [int(row_id) for row_id in gen_batch.non_tensor_batch["row_id"].tolist()]
        knowledge_ids = [str(knowledge_id) for knowledge_id in gen_batch.non_tensor_batch["knowledge_id"].tolist()]
        preferred_server_ids_field = gen_batch.non_tensor_batch.get("preferred_server_id", None)
        preferred_server_ids = None
        if preferred_server_ids_field is not None:
            preferred_server_ids = [None if value in (None, "", "None") else str(value) for value in preferred_server_ids_field.tolist()]
        generate_call = {
            "row_ids": row_ids,
            "knowledge_ids": knowledge_ids,
        }
        if preferred_server_ids is not None:
            generate_call["preferred_server_ids"] = preferred_server_ids
        self.generate_calls.append(generate_call)
        return DataProto.from_dict(
            tensors={
                "responses": torch.tensor([[row_id] for row_id in row_ids], dtype=torch.long),
            },
            non_tensors={
                "row_id": np.array(row_ids, dtype=np.int64),
                "knowledge_id": np.array(knowledge_ids, dtype=object),
                "emitted_text": np.array([f"answer-{row_id}" for row_id in row_ids], dtype=object),
                "preferred_server_id": np.array(preferred_server_ids, dtype=object)
                if preferred_server_ids is not None
                else np.array([None for _ in row_ids], dtype=object),
            },
            meta_info={
                "timing": {"generate": float(len(row_ids))},
                "source": "dummy_async_manager",
            },
        )


def _build_trainer(tmp_path: Path) -> KnowledgeUpdatePPOTrainer:
    trainer = KnowledgeUpdatePPOTrainer.__new__(KnowledgeUpdatePPOTrainer)
    trainer.knowledge_update_config = OmegaConf.create(
        {
            "enable_ephemeral_lora": True,
            "ephemeral_lora_path_field": "ephemeral_lora_path",
            "ephemeral_lora_request_field": "ephemeral_lora_request",
            "ephemeral_lora_dir": str(tmp_path),
            "allow_missing_ephemeral_lora": False,
            "multi_knowledge_strategy": "error",
            "max_simultaneous_ephemeral_loras": 1,
        }
    )
    trainer.enable_ephemeral_lora = True
    trainer.require_single_knowledge_id = True
    trainer.ephemeral_lora_path_field = "ephemeral_lora_path"
    trainer.ephemeral_lora_request_field = "ephemeral_lora_request"
    trainer.ephemeral_lora_variants_field = "ephemeral_lora_variants"
    trainer.ephemeral_lora_variant_field = "ephemeral_lora_variant"
    trainer.ephemeral_lora_dir = str(tmp_path)
    trainer.allow_missing_ephemeral_lora = False
    trainer.multi_knowledge_strategy = "error"
    trainer.max_simultaneous_ephemeral_loras = 1
    trainer.max_ephemeral_loras_per_async_server = 1
    trainer.verify_ephemeral_lora_state = False
    trainer.log_ephemeral_lora_events = False
    trainer.delete_ephemeral_lora_after_use = False
    trainer.memory_diagnostics = False
    trainer.build_missing_ephemeral_lora = False
    trainer.ephemeral_lora_build_model_path = str((tmp_path / "dummy-model").resolve())
    trainer.ephemeral_lora_build_steps = 2
    trainer.ephemeral_lora_build_learning_rate = 5e-4
    trainer.ephemeral_lora_build_fallback_learning_rate = None
    trainer.ephemeral_lora_build_max_length = 512
    trainer.ephemeral_lora_build_lora_rank = 8
    trainer.ephemeral_lora_build_lora_alpha = 16
    trainer.ephemeral_lora_build_lora_dropout = 0.0
    trainer.ephemeral_lora_build_randomize_config = False
    trainer.ephemeral_lora_build_random_base_seed = 42
    trainer.ephemeral_lora_build_lora_rank_candidates = None
    trainer.ephemeral_lora_build_lora_alpha_candidates = None
    trainer.ephemeral_lora_build_lora_dropout_candidates = None
    trainer.ephemeral_lora_build_gradient_clip_norm = 1.0
    trainer.ephemeral_lora_build_dtype = "auto"
    trainer.ephemeral_lora_build_fallback_dtype = "fp32"
    trainer.ephemeral_lora_build_target_modules = list(DEFAULT_EPHEMERAL_LORA_TARGET_MODULES)
    trainer.ephemeral_lora_build_backend = "subprocess"
    trainer._ephemeral_lora_build_workers = None
    trainer.ephemeral_lora_build_pool_size = 1
    trainer.ephemeral_lora_build_timeout_seconds = 7200
    trainer.ephemeral_lora_build_poll_interval_seconds = 1.0
    trainer.ephemeral_lora_build_cuda_devices = [None]
    trainer.default_lora_variant = KNOWLEDGE_LORA_VARIANT
    trainer.enabled_lora_variants = [KNOWLEDGE_LORA_VARIANT]
    trainer.lora_variant_weights = {KNOWLEDGE_LORA_VARIANT: 1.0}
    trainer.lora_variant_selection_seed = 0
    trainer.no_op_reward_gate_threshold = 0.95
    trainer.no_op_reward_gate_window = 32
    trainer.process_reward_positive_sample_limit = 10
    trainer.process_reward_positive_samples_path = None
    trainer._process_reward_positive_samples = []
    trainer._process_reward_positive_sample_keys = set()
    trainer.variant_reward_keys = ("existence_reward", "process_reward", "evaluation_reward")
    trainer._no_op_reward_gate_open = True
    trainer._knowledge_reward_history = {
        "existence_reward": deque(maxlen=32),
        "process_reward": deque(maxlen=32),
        "evaluation_reward": deque(maxlen=32),
    }
    trainer._ephemeral_lora_events = []
    trainer._ephemeral_lora_norm_cache = {}
    trainer.log_variant_lora_norm = False
    trainer.luffy_config = OmegaConf.create({})
    trainer.enable_luffy_teacher_loss = False
    trainer.luffy_teacher_dataset = None
    trainer.luffy_teacher_rng = np.random.default_rng(0)
    trainer._luffy_teacher_indices_by_knowledge = {}
    trainer._luffy_teacher_indices_by_pair = {}
    trainer._luffy_teacher_indices_by_sample_hash_pair = {}
    trainer.actor_rollout_wg = DummyWorkerGroup()
    return trainer


def test_luffy_teacher_only_warmup_skips_on_policy_reward(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.enable_luffy_teacher_loss = True
    trainer.luffy_teacher_dataset = object()
    trainer.luffy_config = OmegaConf.create(
        {
            "warmup_steps": 10,
            "warmup_teacher_only": True,
            "warmup_skip_on_policy_reward": True,
        }
    )
    trainer.global_steps = 3
    batch = DataProto.from_dict(
        tensors={"responses": torch.ones((2, 4), dtype=torch.long)},
        non_tensors={},
        meta_info={},
    )

    assert trainer._should_skip_on_policy_reward_for_luffy_warmup(batch=batch)
    reward_tensor, reward_extra_infos = trainer._build_skipped_luffy_warmup_reward(batch=batch)

    assert reward_tensor.shape == (2, 4)
    assert reward_tensor.dtype == torch.float32
    assert float(reward_tensor.sum().item()) == 0.0
    assert reward_extra_infos["judge_used"] == [0.0, 0.0]
    assert reward_extra_infos["luffy_warmup_reward_skipped"] == [1.0, 1.0]


def _write_adapter(adapter_dir: Path):
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "dummy",
                "bias": "none",
                "inference_mode": True,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "peft_type": "LORA",
                "r": 8,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    save_file({"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 2)}, adapter_dir / "adapter_model.safetensors")


def test_resolve_ephemeral_lora_path_from_extra_info(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    adapter_dir = tmp_path / "from-extra-info"
    _write_adapter(adapter_dir)

    gen_batch = SimpleNamespace(
        non_tensor_batch={"extra_info": [{"knowledge_id": "k1", "ephemeral_lora_path": str(adapter_dir)}]}
    )

    resolved = trainer._resolve_ephemeral_lora_path(gen_batch=gen_batch, knowledge_id="k1")

    assert resolved == str(adapter_dir.resolve())


def test_load_and_clear_ephemeral_lora_dispatch_uses_path_based_staging(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    adapter_dir = tmp_path / "k-001"
    _write_adapter(adapter_dir)

    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "knowledge_id": ["k-001"],
            "ephemeral_lora_request": [{"knowledge_id": "k-001", "lora_path": str(adapter_dir)}],
        }
    )

    trainer._load_ephemeral_lora_for_batch(gen_batch=gen_batch, knowledge_id="k-001")
    trainer._clear_ephemeral_lora(knowledge_id="k-001")

    assert len(trainer.actor_rollout_wg.stage_calls) == 1
    stage_call = trainer.actor_rollout_wg.stage_calls[0]
    assert stage_call == {
        "knowledge_id": "k-001",
        "lora_path": str(adapter_dir.resolve()),
        "eager_load": True,
    }
    assert trainer.actor_rollout_wg.load_calls == []
    assert trainer.actor_rollout_wg.clear_calls == [{"knowledge_id": "k-001"}]


def test_build_ephemeral_lora_execution_plan_groups_rows_by_knowledge_id(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    adapter_dir = tmp_path / "k-grouped"
    _write_adapter(adapter_dir)

    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "knowledge_id": ["k-grouped", "k-grouped"],
            "extra_info": [
                {"knowledge_id": "k-grouped", "ephemeral_lora_path": str(adapter_dir)},
                {"knowledge_id": "k-grouped"},
            ],
            "ephemeral_lora_request": [
                None,
                {"knowledge_id": "k-grouped", "lora_path": str(adapter_dir)},
            ],
        }
    )

    execution_plan = trainer._build_ephemeral_lora_execution_plan(gen_batch)

    assert len(execution_plan) == 1
    assert execution_plan[0]["knowledge_id"] == "k-grouped"
    assert execution_plan[0]["row_count"] == 2
    assert execution_plan[0]["row_indices"] == [0, 1]
    assert execution_plan[0]["lora_path"] == str(adapter_dir.resolve())
    assert execution_plan[0]["request_count"] == 1


def test_build_ephemeral_lora_execution_plan_raises_on_conflicting_paths(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    adapter_dir_a = tmp_path / "adapter-a"
    adapter_dir_b = tmp_path / "adapter-b"
    _write_adapter(adapter_dir_a)
    _write_adapter(adapter_dir_b)

    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "knowledge_id": ["k-001", "k-001"],
            "ephemeral_lora_path": [str(adapter_dir_a), str(adapter_dir_b)],
        }
    )

    try:
        trainer._build_ephemeral_lora_execution_plan(gen_batch)
    except ValueError as exc:
        assert "conflicting ephemeral LoRA paths" in str(exc)
        assert "k-001" in str(exc)
    else:
        raise AssertionError("Expected conflicting adapter paths to raise.")


def test_build_ephemeral_lora_execution_plan_materializes_missing_adapter_from_extra_info(
    monkeypatch,
    tmp_path: Path,
):
    trainer = _build_trainer(tmp_path)
    trainer.build_missing_ephemeral_lora = True
    built_adapter_dir = tmp_path / "built" / "k-built"
    build_calls = []

    def fake_build_missing_ephemeral_lora(*, knowledge_id, lora_variant, sample, output_dir):
        build_calls.append(
            {
                "knowledge_id": knowledge_id,
                "lora_variant": lora_variant,
                "sample": sample,
                "output_dir": output_dir,
            }
        )
        _write_adapter(Path(output_dir))
        return str(Path(output_dir).resolve())

    monkeypatch.setattr(trainer, "_build_missing_ephemeral_lora", fake_build_missing_ephemeral_lora)
    monkeypatch.setattr(
        trainer,
        "_build_ephemeral_lora_dir_from_knowledge_id",
        lambda knowledge_id, lora_variant=None: str((tmp_path / "built" / str(knowledge_id)).resolve()),
    )

    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "knowledge_id": ["k-built"],
            "extra_info": [
                {
                    "knowledge_id": "k-built",
                    "title": "Title",
                    "context": "Context",
                    "question": "What changed?",
                    "answer": "New answer",
                }
            ],
        }
    )

    execution_plan = trainer._build_ephemeral_lora_execution_plan(gen_batch)

    assert execution_plan[0]["lora_path"] == str(built_adapter_dir)
    assert build_calls == []
    assert execution_plan[0]["build_request"] == {
        "request_key": str(built_adapter_dir),
        "knowledge_id": "k-built",
        "lora_variant": "knowledge",
        "sample": {
            "knowledge_id": "k-built",
            "title": "Title",
            "category": "",
            "subcategory": "",
            "context": "Context",
            "question": "What changed?",
            "answer": "New answer",
        },
        "output_dir": str(built_adapter_dir),
    }


def test_extract_build_sample_from_group_raises_without_question_and_answer(tmp_path: Path):
    trainer = _build_trainer(tmp_path)

    try:
        trainer._extract_build_sample_from_group(
            group={"extra_infos": [{"knowledge_id": "k-missing", "context": "Only context"}]},
            knowledge_id="k-missing",
        )
    except ValueError as exc:
        assert "k-missing" in str(exc)
        assert "enough QA metadata" in str(exc)
    else:
        raise AssertionError("Expected missing QA metadata to raise.")


def test_persistent_build_pool_routes_requests_through_workers(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.ephemeral_lora_build_backend = "persistent"
    trainer.ephemeral_lora_build_pool_size = 2
    trainer.ephemeral_lora_build_cuda_devices = ["0", "1"]

    class FakeWorker:
        def __init__(self, device_token):
            self.device_token = device_token
            self.requests = []

        def build(self, request):
            self.requests.append(request)
            output_dir = Path(request["output_dir"])
            _write_adapter(output_dir)
            metadata = {"build_backend": "persistent_worker", "lora_weight_l2_norm": 1.25}
            (output_dir / "knowledge_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            return {
                "ok": True,
                "knowledge_id": request["knowledge_id"],
                "lora_variant": request["lora_variant"],
                "output_dir": str(output_dir),
                "build_metadata": metadata,
                "device_token": self.device_token,
            }

    workers = [FakeWorker("0"), FakeWorker("1")]
    monkeypatch.setattr(
        trainer,
        "_ensure_ephemeral_lora_build_workers",
        lambda *, effective_pool_size, device_tokens: workers,
    )

    pending_requests = [
        {
            "request_key": str(tmp_path / "k0"),
            "knowledge_id": "k0",
            "lora_variant": KNOWLEDGE_LORA_VARIANT,
            "sample": {"question": "Q0", "answer": "A0"},
            "output_dir": str(tmp_path / "k0"),
        },
        {
            "request_key": str(tmp_path / "k1"),
            "knowledge_id": "k1",
            "lora_variant": KNOWLEDGE_LORA_VARIANT,
            "sample": {"question": "Q1", "answer": "A1"},
            "output_dir": str(tmp_path / "k1"),
        },
    ]

    resolved = trainer._build_missing_ephemeral_lora_requests_persistent(
        pending_requests=pending_requests,
        resolved_paths={},
        effective_pool_size=2,
        device_tokens=["0", "1"],
    )

    assert set(resolved) == {str(tmp_path / "k0"), str(tmp_path / "k1")}
    assert workers[0].requests[0]["knowledge_id"] == "k0"
    assert workers[1].requests[0]["knowledge_id"] == "k1"


def test_build_ephemeral_lora_execution_plan_mixes_prebuilt_and_materialized_paths(
    monkeypatch,
    tmp_path: Path,
):
    trainer = _build_trainer(tmp_path)
    trainer.build_missing_ephemeral_lora = True
    trainer.require_single_knowledge_id = False
    trainer.multi_knowledge_strategy = "serial_groups"

    prebuilt_dir = tmp_path / "prebuilt" / "k-prebuilt"
    built_dir = tmp_path / "built" / "k-missing"
    _write_adapter(prebuilt_dir)
    build_calls = []

    def fake_build_missing_ephemeral_lora(*, knowledge_id, lora_variant, sample, output_dir):
        build_calls.append(
            {
                "knowledge_id": knowledge_id,
                "lora_variant": lora_variant,
                "sample": sample,
                "output_dir": output_dir,
            }
        )
        _write_adapter(Path(output_dir))
        return str(Path(output_dir).resolve())

    monkeypatch.setattr(trainer, "_build_missing_ephemeral_lora", fake_build_missing_ephemeral_lora)
    monkeypatch.setattr(
        trainer,
        "_build_ephemeral_lora_dir_from_knowledge_id",
        lambda knowledge_id, lora_variant=None: str((tmp_path / ("prebuilt" if knowledge_id == "k-prebuilt" else "built") / str(knowledge_id)).resolve()),
    )

    gen_batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[10], [11]], dtype=torch.long)},
        non_tensors={
            "knowledge_id": np.array(["k-prebuilt", "k-missing"], dtype=object),
            "extra_info": np.array(
                [
                    {
                        "knowledge_id": "k-prebuilt",
                        "question": "What changed for prebuilt?",
                        "answer": "Prebuilt answer",
                    },
                    {
                        "knowledge_id": "k-missing",
                        "title": "Missing Title",
                        "context": "Missing context",
                        "question": "What changed for missing?",
                        "answer": "Missing answer",
                    },
                ],
                dtype=object,
            ),
            "ephemeral_lora_request": np.array(
                [
                    {"knowledge_id": "k-prebuilt", "lora_path": str(prebuilt_dir)},
                    None,
                ],
                dtype=object,
            ),
        },
        meta_info={"do_sample": True},
    )

    execution_plan = trainer._build_ephemeral_lora_execution_plan(gen_batch)

    assert [item["knowledge_id"] for item in execution_plan] == ["k-prebuilt", "k-missing"]
    assert execution_plan[0]["lora_path"] == str(prebuilt_dir.resolve())
    assert execution_plan[1]["lora_path"] == str(built_dir.resolve())
    assert build_calls == []
    assert execution_plan[1]["build_request"] == {
        "request_key": str(built_dir),
        "knowledge_id": "k-missing",
        "lora_variant": "knowledge",
        "sample": {
            "knowledge_id": "k-missing",
            "title": "Missing Title",
            "category": "",
            "subcategory": "",
            "context": "Missing context",
            "question": "What changed for missing?",
            "answer": "Missing answer",
        },
        "output_dir": str(built_dir),
    }


def test_validate_ephemeral_lora_path_requires_adapter_files(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    incomplete_dir = tmp_path / "missing-files"
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    (incomplete_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    try:
        trainer._validate_ephemeral_lora_path(str(incomplete_dir))
    except FileNotFoundError as exc:
        assert "adapter_model.safetensors" in str(exc)
    else:
        raise AssertionError("Expected missing adapter files to raise.")


def test_verify_ephemeral_lora_state_detects_loaded_id_mismatch(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.verify_ephemeral_lora_state = True
    trainer.actor_rollout_wg.state_response = [
        {
            "worker_rank": 0,
            "staged_knowledge_id": "k-001",
            "staged_knowledge_ids": ["k-001"],
            "loaded_knowledge_id": "unexpected",
        }
    ]

    try:
        trainer._verify_ephemeral_lora_state(
            expected_staged="k-001",
            expected_loaded="k-001",
            phase="after_generate",
        )
    except RuntimeError as exc:
        assert "after_generate" in str(exc)
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("Expected _verify_ephemeral_lora_state to raise on mismatched worker state.")


def test_normalize_worker_results_flattens_nested_server_state_payloads(tmp_path: Path):
    trainer = _build_trainer(tmp_path)

    worker_result = [
        [
            {
                "worker_rank": 0,
                "staged_knowledge_id": "k-001",
                "staged_knowledge_ids": ["k-001"],
                "loaded_knowledge_id": "k-001",
            },
            {
                "worker_rank": 1,
                "staged_knowledge_id": "k-001",
                "staged_knowledge_ids": ["k-001"],
                "loaded_knowledge_id": "k-001",
            },
        ],
        {
            "knowledge_id": "k-001",
            "server_states": [
                {
                    "worker_rank": 2,
                    "staged_knowledge_id": None,
                    "staged_knowledge_ids": [],
                    "loaded_knowledge_id": None,
                },
            ],
        },
    ]

    normalized = trainer._normalize_worker_results(worker_result)

    assert normalized == [
        {
            "worker_rank": 0,
            "staged_knowledge_id": "k-001",
            "staged_knowledge_ids": ["k-001"],
            "loaded_knowledge_id": "k-001",
        },
        {
            "worker_rank": 1,
            "staged_knowledge_id": "k-001",
            "staged_knowledge_ids": ["k-001"],
            "loaded_knowledge_id": "k-001",
        },
        {
            "worker_rank": 2,
            "staged_knowledge_id": None,
            "staged_knowledge_ids": [],
            "loaded_knowledge_id": None,
        },
    ]


def test_verify_ephemeral_lora_state_accepts_nested_async_manager_states(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.verify_ephemeral_lora_state = True
    trainer.async_rollout_manager = DummyAsyncRolloutManager()
    trainer.async_rollout_manager.state_response = [
        [
            {
                "worker_rank": 0,
                "staged_knowledge_id": "k-async",
                "staged_knowledge_ids": ["k-async"],
                "loaded_knowledge_id": "k-async",
            },
            {
                "worker_rank": 1,
                "staged_knowledge_id": "k-async",
                "staged_knowledge_ids": ["k-async"],
                "loaded_knowledge_id": "k-async",
            },
        ]
    ]

    trainer._verify_ephemeral_lora_state(
        expected_staged="k-async",
        expected_loaded="k-async",
        phase="after_generate",
    )


def test_verify_ephemeral_lora_states_by_replica_falls_back_to_requested_replica_order(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.verify_ephemeral_lora_state = True
    trainer.async_rollout_manager = DummyAsyncRolloutManager(replica_count=2)
    trainer.async_rollout_manager.state_response = [
        {
            "worker_rank": None,
            "staged_knowledge_id": "k-a",
            "staged_knowledge_ids": ["k-a"],
            "loaded_knowledge_id": "k-a",
        },
        {
            "worker_rank": None,
            "staged_knowledge_id": "k-b",
            "staged_knowledge_ids": ["k-b"],
            "loaded_knowledge_id": "k-b",
        },
    ]

    trainer._verify_ephemeral_lora_states_by_replica(
        expected_states_by_replica={
            0: {"expected_staged": ["k-a"], "expected_loaded": "k-a"},
            1: {"expected_staged": ["k-b"], "expected_loaded": "k-b"},
        },
        phase="after_stage",
    )


def test_load_ephemeral_lora_verifies_loaded_after_stage_for_async_manager(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.verify_ephemeral_lora_state = True
    trainer.async_rollout_manager = DummyAsyncRolloutManager()
    adapter_dir = tmp_path / "k-async"
    _write_adapter(adapter_dir)

    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "knowledge_id": ["k-async"],
            "ephemeral_lora_request": [{"knowledge_id": "k-async", "lora_path": str(adapter_dir)}],
        }
    )

    trainer._load_ephemeral_lora_for_batch(gen_batch=gen_batch, knowledge_id="k-async")

    assert trainer.async_rollout_manager.stage_calls == [
        {"lora_path": str(adapter_dir.resolve()), "knowledge_id": "k-async", "eager_load": True}
    ]


def test_select_lora_variant_for_group_respects_no_op_gate(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.enabled_lora_variants = [KNOWLEDGE_LORA_VARIANT, NO_OP_LORA_VARIANT]
    trainer.lora_variant_weights = {KNOWLEDGE_LORA_VARIANT: 1.0, NO_OP_LORA_VARIANT: 1.0}
    trainer._no_op_reward_gate_open = False

    group = {
        "knowledge_id": "k-001",
        "variant_requests": {
            KNOWLEDGE_LORA_VARIANT: {"knowledge_id": "k-001", "lora_variant": KNOWLEDGE_LORA_VARIANT, "lora_path": "/tmp/knowledge"},
            NO_OP_LORA_VARIANT: {"knowledge_id": "k-001", "lora_variant": NO_OP_LORA_VARIANT, "lora_path": "/tmp/no_op"},
        },
    }

    selected_variant, request = trainer._select_lora_variant_for_group(group=group)

    assert selected_variant == KNOWLEDGE_LORA_VARIANT
    assert request["lora_path"] == "/tmp/knowledge"


def test_update_no_op_reward_gate_opens_once_all_reward_means_clear_threshold(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.enabled_lora_variants = [KNOWLEDGE_LORA_VARIANT, NO_OP_LORA_VARIANT]
    trainer._no_op_reward_gate_open = False
    batch = SimpleNamespace(
        non_tensor_batch={
            trainer.ephemeral_lora_variant_field: np.array([KNOWLEDGE_LORA_VARIANT], dtype=object),
        }
    )

    trainer._update_no_op_reward_gate(
        batch=batch,
        reward_extra_infos_dict={
            "existence_reward": [0.96],
            "process_reward": [0.97],
            "evaluation_reward": [0.98],
        },
    )

    assert trainer._no_op_reward_gate_open is True
    assert trainer._ephemeral_lora_events[-1]["event"] == "no_op_gate_open"


def test_build_ephemeral_lora_execution_plan_can_select_no_op_variant(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.enabled_lora_variants = [NO_OP_LORA_VARIANT]
    trainer.lora_variant_weights = {NO_OP_LORA_VARIANT: 1.0}
    trainer._no_op_reward_gate_open = True

    no_op_dir = tmp_path / "no_op" / "k-no-op"
    _write_adapter(no_op_dir)
    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "knowledge_id": ["k-no-op"],
            "extra_info": [
                {
                    "knowledge_id": "k-no-op",
                    "ephemeral_lora_variants": {
                        "no_op": {
                            "knowledge_id": "k-no-op",
                            "lora_variant": "no_op",
                            "lora_path": str(no_op_dir),
                        }
                    },
                }
            ],
        }
    )

    execution_plan = trainer._build_ephemeral_lora_execution_plan(gen_batch)

    assert execution_plan[0]["lora_variant"] == NO_OP_LORA_VARIANT
    assert execution_plan[0]["lora_path"] == str(no_op_dir.resolve())


def test_load_and_clear_ephemeral_lora_prefers_async_rollout_manager(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.async_rollout_manager = DummyAsyncRolloutManager()
    adapter_dir = tmp_path / "k-async"
    _write_adapter(adapter_dir)

    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "knowledge_id": ["k-async"],
            "ephemeral_lora_request": [{"knowledge_id": "k-async", "lora_path": str(adapter_dir)}],
        }
    )

    trainer._load_ephemeral_lora_for_batch(gen_batch=gen_batch, knowledge_id="k-async")
    trainer._clear_ephemeral_lora(knowledge_id="k-async")

    assert len(trainer.async_rollout_manager.stage_calls) == 1
    assert trainer.async_rollout_manager.stage_calls[0]["eager_load"] is True
    assert trainer.async_rollout_manager.load_calls == []
    assert trainer.async_rollout_manager.clear_calls == [{"knowledge_id": "k-async"}]
    assert trainer.actor_rollout_wg.stage_calls == []
    assert trainer.actor_rollout_wg.clear_calls == []


def test_patch_rollout_generation_serial_groups_executes_per_group_and_restores_row_order(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.async_rollout_manager = DummyAsyncRolloutManager()
    trainer.multi_knowledge_strategy = "serial_groups"
    trainer.require_single_knowledge_id = False

    adapter_dir_a = tmp_path / "k-a"
    adapter_dir_b = tmp_path / "k-b"
    _write_adapter(adapter_dir_a)
    _write_adapter(adapter_dir_b)

    gen_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[10], [11], [12]], dtype=torch.long),
        },
        non_tensors={
            "knowledge_id": np.array(["k-a", "k-b", "k-a"], dtype=object),
            "row_id": np.array([0, 1, 2], dtype=np.int64),
            "ephemeral_lora_request": np.array(
                [
                    {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a)},
                    {"knowledge_id": "k-b", "lora_path": str(adapter_dir_b)},
                    {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a)},
                ],
                dtype=object,
            ),
        },
        meta_info={"do_sample": True},
    )

    with trainer._patch_rollout_generation():
        output = trainer.async_rollout_manager.generate_sequences(gen_batch)

    assert trainer.async_rollout_manager.generate_calls == [
        {"row_ids": [0, 2], "knowledge_ids": ["k-a", "k-a"]},
        {"row_ids": [1], "knowledge_ids": ["k-b"]},
    ]
    assert trainer.async_rollout_manager.stage_calls == [
        {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a.resolve()), "eager_load": True},
        {"knowledge_id": "k-b", "lora_path": str(adapter_dir_b.resolve()), "eager_load": True},
    ]
    assert trainer.async_rollout_manager.load_calls == []
    assert trainer.async_rollout_manager.clear_calls == [{"knowledge_id": "k-a"}, {"knowledge_id": "k-b"}]
    assert output.non_tensor_batch["row_id"].tolist() == [0, 1, 2]
    assert output.non_tensor_batch["knowledge_id"].tolist() == ["k-a", "k-b", "k-a"]
    assert output.non_tensor_batch["emitted_text"].tolist() == ["answer-0", "answer-1", "answer-2"]
    assert output.batch["responses"].squeeze(-1).tolist() == [0, 1, 2]
    assert output.meta_info["timing"] == {"generate": 3.0}
    assert output.meta_info["source"] == "dummy_async_manager"


def test_resolve_rollout_execution_strategy_distinguishes_single_group_and_serial_groups(tmp_path: Path):
    trainer = _build_trainer(tmp_path)

    assert trainer._resolve_rollout_execution_strategy([]) == "single_group"
    assert trainer._resolve_rollout_execution_strategy([{"knowledge_id": "k-a"}]) == "single_group"

    trainer.multi_knowledge_strategy = "serial_groups"
    trainer.max_simultaneous_ephemeral_loras = 1

    assert trainer._resolve_rollout_execution_strategy(
        [{"knowledge_id": "k-a"}, {"knowledge_id": "k-b"}]
    ) == "serial_groups"


def test_resolve_rollout_execution_strategy_distinguishes_serial_waves(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.multi_knowledge_strategy = "serial_waves"
    trainer.max_simultaneous_ephemeral_loras = 2

    assert trainer._resolve_rollout_execution_strategy(
        [{"knowledge_id": "k-a"}, {"knowledge_id": "k-b"}]
    ) == "serial_waves"


def test_resolve_rollout_execution_strategy_distinguishes_parallel_waves(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.async_rollout_manager = DummyAsyncRolloutManager(replica_count=2)
    trainer.multi_knowledge_strategy = "parallel_waves"
    trainer.require_single_knowledge_id = False
    trainer.max_simultaneous_ephemeral_loras = 2

    assert trainer._resolve_rollout_execution_strategy(
        [{"knowledge_id": "k-a"}, {"knowledge_id": "k-b"}]
    ) == "parallel_waves"


def test_build_rollout_execution_schedule_turns_serial_groups_into_single_item_waves(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.multi_knowledge_strategy = "serial_groups"
    trainer.require_single_knowledge_id = False

    execution_schedule = trainer._build_rollout_execution_schedule(
        [
            {"knowledge_id": "k-a", "row_indices": [0, 2], "lora_path": "/tmp/a"},
            {"knowledge_id": "k-b", "row_indices": [1], "lora_path": "/tmp/b"},
        ]
    )

    assert execution_schedule["strategy"] == "serial_groups"
    assert execution_schedule["wave_count"] == 2
    assert execution_schedule["max_active_loras"] == 1
    assert [wave["wave_index"] for wave in execution_schedule["waves"]] == [0, 1]
    assert [wave["plan_items"][0]["knowledge_id"] for wave in execution_schedule["waves"]] == ["k-a", "k-b"]


def test_build_rollout_execution_schedule_batches_serial_waves_by_capacity(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.multi_knowledge_strategy = "serial_waves"
    trainer.require_single_knowledge_id = False
    trainer.max_simultaneous_ephemeral_loras = 2

    execution_schedule = trainer._build_rollout_execution_schedule(
        [
            {"knowledge_id": "k-a", "row_indices": [0, 3], "lora_path": "/tmp/a"},
            {"knowledge_id": "k-b", "row_indices": [1], "lora_path": "/tmp/b"},
            {"knowledge_id": "k-c", "row_indices": [2], "lora_path": "/tmp/c"},
        ]
    )

    assert execution_schedule["strategy"] == "serial_waves"
    assert execution_schedule["wave_count"] == 2
    assert execution_schedule["max_active_loras"] == 1
    assert execution_schedule["max_planned_loras_per_wave"] == 2
    assert [wave["wave_index"] for wave in execution_schedule["waves"]] == [0, 1]
    assert [[item["knowledge_id"] for item in wave["plan_items"]] for wave in execution_schedule["waves"]] == [
        ["k-a", "k-b"],
        ["k-c"],
    ]


def test_build_rollout_execution_schedule_assigns_parallel_waves_to_servers(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.async_rollout_manager = DummyAsyncRolloutManager(replica_count=2)
    trainer.multi_knowledge_strategy = "parallel_waves"
    trainer.require_single_knowledge_id = False
    trainer.max_simultaneous_ephemeral_loras = 2

    execution_schedule = trainer._build_rollout_execution_schedule(
        [
            {"knowledge_id": "k-a", "row_indices": [0, 3], "lora_path": "/tmp/a"},
            {"knowledge_id": "k-b", "row_indices": [1], "lora_path": "/tmp/b"},
            {"knowledge_id": "k-c", "row_indices": [2], "lora_path": "/tmp/c"},
        ]
    )

    assert execution_schedule["strategy"] == "parallel_waves"
    assert execution_schedule["wave_count"] == 2
    assert execution_schedule["max_active_loras"] == 2
    assert execution_schedule["max_planned_loras_per_wave"] == 2
    assert execution_schedule["server_ids"] == ["server-0", "server-1"]
    assert execution_schedule["waves"][0]["plan_items"][0]["replica_index"] == 0
    assert execution_schedule["waves"][0]["plan_items"][0]["preferred_server_id"] == "server-0"
    assert execution_schedule["waves"][0]["plan_items"][1]["replica_index"] == 1
    assert execution_schedule["waves"][0]["plan_items"][1]["preferred_server_id"] == "server-1"


def test_validate_rollout_execution_schedule_rejects_duplicate_rows_across_waves(tmp_path: Path):
    trainer = _build_trainer(tmp_path)

    try:
        trainer._validate_rollout_execution_schedule(
            execution_schedule={
                "strategy": "serial_groups",
                "waves": [
                    {
                        "wave_index": 0,
                        "plan_items": [{"knowledge_id": "k-a", "row_indices": [0, 1]}],
                    },
                    {
                        "wave_index": 1,
                        "plan_items": [{"knowledge_id": "k-b", "row_indices": [1, 2]}],
                    },
                ],
            },
            batch_size=3,
        )
    except ValueError as exc:
        assert "duplicate row indices across waves" in str(exc)
    else:
        raise AssertionError("Expected duplicate rows across waves to fail validation.")


def test_validate_rollout_execution_schedule_rejects_parallel_wave_replica_conflict(tmp_path: Path):
    trainer = _build_trainer(tmp_path)

    try:
        trainer._validate_rollout_execution_schedule(
            execution_schedule={
                "strategy": "parallel_waves",
                "waves": [
                    {
                        "wave_index": 0,
                        "plan_items": [
                            {
                                "knowledge_id": "k-a",
                                "row_indices": [0],
                                "replica_index": 0,
                                "preferred_server_id": "server-0",
                            },
                            {
                                "knowledge_id": "k-b",
                                "row_indices": [1],
                                "replica_index": 0,
                                "preferred_server_id": "server-1",
                            },
                        ],
                    }
                ],
            },
            batch_size=2,
        )
    except ValueError as exc:
        assert "per-replica LoRA capacity" in str(exc)
    else:
        raise AssertionError("Expected duplicate replica assignment to fail validation.")


def test_build_preferred_server_ids_for_wave_rejects_conflicting_row_routing(tmp_path: Path):
    trainer = _build_trainer(tmp_path)

    try:
        trainer._build_preferred_server_ids_for_wave(
            row_indices=[0, 1],
            plan_items=[
                {"knowledge_id": "k-a", "row_indices": [0], "preferred_server_id": "server-0"},
                {"knowledge_id": "k-b", "row_indices": [0, 1], "preferred_server_id": "server-1"},
            ],
        )
    except ValueError as exc:
        assert "row routing conflict" in str(exc)
    else:
        raise AssertionError("Expected conflicting row routing to fail.")


def test_patch_rollout_generation_serial_waves_executes_wave_items_sequentially(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.async_rollout_manager = DummyAsyncRolloutManager()
    trainer.multi_knowledge_strategy = "serial_waves"
    trainer.require_single_knowledge_id = False
    trainer.max_simultaneous_ephemeral_loras = 2

    adapter_dir_a = tmp_path / "k-a"
    adapter_dir_b = tmp_path / "k-b"
    adapter_dir_c = tmp_path / "k-c"
    _write_adapter(adapter_dir_a)
    _write_adapter(adapter_dir_b)
    _write_adapter(adapter_dir_c)

    gen_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[10], [11], [12], [13]], dtype=torch.long),
        },
        non_tensors={
            "knowledge_id": np.array(["k-a", "k-b", "k-c", "k-a"], dtype=object),
            "row_id": np.array([0, 1, 2, 3], dtype=np.int64),
            "ephemeral_lora_request": np.array(
                [
                    {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a)},
                    {"knowledge_id": "k-b", "lora_path": str(adapter_dir_b)},
                    {"knowledge_id": "k-c", "lora_path": str(adapter_dir_c)},
                    {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a)},
                ],
                dtype=object,
            ),
        },
        meta_info={"do_sample": True},
    )

    with trainer._patch_rollout_generation():
        output = trainer.async_rollout_manager.generate_sequences(gen_batch)

    assert trainer.async_rollout_manager.generate_calls == [
        {"row_ids": [0, 3], "knowledge_ids": ["k-a", "k-a"]},
        {"row_ids": [1], "knowledge_ids": ["k-b"]},
        {"row_ids": [2], "knowledge_ids": ["k-c"]},
    ]
    assert trainer.async_rollout_manager.stage_calls == [
        {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a.resolve()), "eager_load": False},
        {"knowledge_id": "k-b", "lora_path": str(adapter_dir_b.resolve()), "eager_load": False},
        {"knowledge_id": "k-c", "lora_path": str(adapter_dir_c.resolve()), "eager_load": False},
    ]
    assert trainer.async_rollout_manager.load_calls == [
        {"knowledge_id": "k-a"},
        {"knowledge_id": "k-b"},
        {"knowledge_id": "k-c"},
    ]
    assert trainer.async_rollout_manager.clear_calls == [
        {"knowledge_id": "k-a"},
        {"knowledge_id": "k-b"},
        {"knowledge_id": "k-c"},
    ]
    assert output.non_tensor_batch["row_id"].tolist() == [0, 1, 2, 3]
    assert output.non_tensor_batch["knowledge_id"].tolist() == ["k-a", "k-b", "k-c", "k-a"]
    assert output.non_tensor_batch["emitted_text"].tolist() == ["answer-0", "answer-1", "answer-2", "answer-3"]
    assert output.batch["responses"].squeeze(-1).tolist() == [0, 1, 2, 3]
    assert output.meta_info["timing"] == {"generate": 4.0}
    assert output.meta_info["source"] == "dummy_async_manager"


def test_patch_rollout_generation_serial_waves_preserves_other_staged_loras_until_wave_clear(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.async_rollout_manager = DummyAsyncRolloutManager()
    trainer.multi_knowledge_strategy = "serial_waves"
    trainer.require_single_knowledge_id = False
    trainer.max_simultaneous_ephemeral_loras = 2
    trainer.verify_ephemeral_lora_state = True

    adapter_dir_a = tmp_path / "k-a"
    adapter_dir_b = tmp_path / "k-b"
    _write_adapter(adapter_dir_a)
    _write_adapter(adapter_dir_b)

    gen_batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[10], [11]], dtype=torch.long)},
        non_tensors={
            "knowledge_id": np.array(["k-a", "k-b"], dtype=object),
            "row_id": np.array([0, 1], dtype=np.int64),
            "ephemeral_lora_request": np.array(
                [
                    {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a)},
                    {"knowledge_id": "k-b", "lora_path": str(adapter_dir_b)},
                ],
                dtype=object,
            ),
        },
        meta_info={"do_sample": True},
    )

    with trainer._patch_rollout_generation():
        trainer.async_rollout_manager.generate_sequences(gen_batch)

    verify_events = [event for event in trainer._ephemeral_lora_events if event["event"] == "verify"]
    after_wave_stage = [event for event in verify_events if event.get("phase") == "after_wave_stage"]
    after_clear = [event for event in verify_events if event.get("phase") == "after_clear"]

    assert after_wave_stage
    assert after_wave_stage[0]["worker_states"][0]["staged_knowledge_ids"] == ["k-a", "k-b"]
    assert [event["worker_states"][0]["staged_knowledge_ids"] for event in after_clear] == [["k-b"], []]


def test_patch_rollout_generation_parallel_waves_routes_rows_to_preferred_servers(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.async_rollout_manager = DummyAsyncRolloutManager(replica_count=2)
    trainer.multi_knowledge_strategy = "parallel_waves"
    trainer.require_single_knowledge_id = False
    trainer.max_simultaneous_ephemeral_loras = 2
    trainer.verify_ephemeral_lora_state = True

    adapter_dir_a = tmp_path / "k-a"
    adapter_dir_b = tmp_path / "k-b"
    adapter_dir_c = tmp_path / "k-c"
    _write_adapter(adapter_dir_a)
    _write_adapter(adapter_dir_b)
    _write_adapter(adapter_dir_c)

    gen_batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[10], [11], [12], [13]], dtype=torch.long)},
        non_tensors={
            "knowledge_id": np.array(["k-a", "k-b", "k-c", "k-a"], dtype=object),
            "row_id": np.array([0, 1, 2, 3], dtype=np.int64),
            "ephemeral_lora_request": np.array(
                [
                    {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a)},
                    {"knowledge_id": "k-b", "lora_path": str(adapter_dir_b)},
                    {"knowledge_id": "k-c", "lora_path": str(adapter_dir_c)},
                    {"knowledge_id": "k-a", "lora_path": str(adapter_dir_a)},
                ],
                dtype=object,
            ),
        },
        meta_info={"do_sample": True},
    )

    with trainer._patch_rollout_generation():
        output = trainer.async_rollout_manager.generate_sequences(gen_batch)

    assert trainer.async_rollout_manager.stage_calls == [
        {
            "knowledge_id": "k-a",
            "lora_path": str(adapter_dir_a.resolve()),
            "eager_load": True,
            "replica_indices": [0],
        },
        {
            "knowledge_id": "k-b",
            "lora_path": str(adapter_dir_b.resolve()),
            "eager_load": True,
            "replica_indices": [1],
        },
        {
            "knowledge_id": "k-c",
            "lora_path": str(adapter_dir_c.resolve()),
            "eager_load": True,
            "replica_indices": [0],
        },
    ]
    assert trainer.async_rollout_manager.generate_calls == [
        {
            "row_ids": [0, 1, 3],
            "knowledge_ids": ["k-a", "k-b", "k-a"],
            "preferred_server_ids": ["server-0", "server-1", "server-0"],
        },
        {
            "row_ids": [2],
            "knowledge_ids": ["k-c"],
            "preferred_server_ids": ["server-0"],
        },
    ]
    assert trainer.async_rollout_manager.clear_calls == [
        {"knowledge_id": "k-a", "replica_indices": [0]},
        {"knowledge_id": "k-b", "replica_indices": [1]},
        {"knowledge_id": "k-c", "replica_indices": [0]},
    ]
    assert output.non_tensor_batch["preferred_server_id"].tolist() == ["server-0", "server-1", "server-0", "server-0"]
    assert output.non_tensor_batch["row_id"].tolist() == [0, 1, 2, 3]


class DummyLuffyTeacherDataset:
    knowledge_id_key = "knowledge_id"

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        return self.rows[item]


def _build_luffy_teacher_row(*, knowledge_id: str, meta_query: str, answer_token: int = 40, sample_hash: str = ""):
    return {
        "input_ids": torch.tensor([11, 12, answer_token, answer_token + 1], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        "loss_mask": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "knowledge_id": knowledge_id,
        "sample_hash": sample_hash,
        "meta_query": meta_query,
        "judge_process_reward": 1.0,
        "judge_existence_reward": 1.0,
        "judge_evaluation_reward": 1.0,
    }


def _build_luffy_sampling_trainer(
    tmp_path: Path,
    *,
    strict_match: bool = False,
    batch_size=0,
    batch_size_source="post_repeat",
):
    trainer = _build_trainer(tmp_path)
    trainer.luffy_config = OmegaConf.create(
        {
            "batch_size": batch_size,
            "batch_size_source": batch_size_source,
            "strict_match": strict_match,
        }
    )
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 4}}})
    trainer.luffy_teacher_rng = np.random.default_rng(0)
    trainer.luffy_teacher_dataset = DummyLuffyTeacherDataset(
        [
            _build_luffy_teacher_row(knowledge_id="k-known", meta_query="What changed?"),
            _build_luffy_teacher_row(knowledge_id="k-other", meta_query="Other query", answer_token=50),
        ]
    )
    trainer._luffy_teacher_indices_by_pair = {("k-known", "What changed?"): [0]}
    trainer._luffy_teacher_indices_by_sample_hash_pair = {}
    trainer._luffy_teacher_indices_by_knowledge = {"k-known": [0], "k-other": [1]}
    trainer.variant_reward_keys = ("process_reward",)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    return trainer


def _build_luffy_template_batch():
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.zeros((2, 6), dtype=torch.long),
            "attention_mask": torch.ones((2, 6), dtype=torch.long),
            "position_ids": torch.arange(6, dtype=torch.long).repeat(2, 1),
            "responses": torch.zeros((2, 3), dtype=torch.long),
            "response_mask": torch.ones((2, 3), dtype=torch.long),
            "token_level_scores": torch.zeros((2, 3), dtype=torch.float32),
            "token_level_rewards": torch.zeros((2, 3), dtype=torch.float32),
        },
        non_tensors={
            "extra_info": np.array(
                [
                    {"knowledge_id": "k-known", "meta_query": "What changed?"},
                    {"knowledge_id": "k-missing", "meta_query": "Missing query"},
                ],
                dtype=object,
            )
        },
        meta_info={},
    )


def test_luffy_teacher_strict_match_skips_unmatched_rows(tmp_path: Path):
    trainer = _build_luffy_sampling_trainer(tmp_path, strict_match=True)

    teacher_batch = trainer._sample_luffy_teacher_batch(_build_luffy_template_batch())

    assert len(teacher_batch) == 1
    assert teacher_batch.non_tensor_batch["luffy_teacher_match_status"].tolist() == ["knowledge_meta_query"]
    assert teacher_batch.non_tensor_batch["extra_info"][0]["source_row_index"] == 0
    assert teacher_batch.non_tensor_batch["extra_info"][0]["teacher_match_status"] == "knowledge_meta_query"
    assert teacher_batch.meta_info["luffy_teacher_sampling_stats"]["match_status_counts"] == {
        "knowledge_meta_query": 1,
        "miss": 1,
    }
    assert teacher_batch.meta_info["luffy_teacher_sampling_stats"]["miss_fraction"] == 0.5


def test_luffy_teacher_non_strict_records_random_fallback_rows(tmp_path: Path):
    trainer = _build_luffy_sampling_trainer(tmp_path, strict_match=False)

    teacher_batch = trainer._sample_luffy_teacher_batch(_build_luffy_template_batch())

    assert len(teacher_batch) == 2
    assert teacher_batch.non_tensor_batch["luffy_teacher_match_status"].tolist() == [
        "knowledge_meta_query",
        "fallback_random",
    ]
    assert teacher_batch.meta_info["luffy_teacher_sampling_stats"]["match_status_counts"] == {
        "knowledge_meta_query": 1,
        "fallback_random": 1,
    }


def test_luffy_teacher_pre_repeat_batch_size_uses_one_teacher_per_prompt(tmp_path: Path):
    trainer = _build_luffy_sampling_trainer(tmp_path, batch_size=0, batch_size_source="pre_repeat")

    assert trainer._resolve_luffy_teacher_sample_size(template_batch_size=8) == 2


def test_luffy_teacher_only_warmup_can_oversample_with_replacement(tmp_path: Path):
    trainer = _build_luffy_sampling_trainer(tmp_path, batch_size=0, batch_size_source="pre_repeat")
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}})
    trainer.global_steps = 0
    trainer.luffy_config = OmegaConf.create(
        {
            "batch_size": 0,
            "batch_size_source": "pre_repeat",
            "warmup_steps": 10,
            "warmup_teacher_only": True,
            "warmup_batch_size": 128,
        }
    )

    assert trainer._luffy_teacher_warmup_allows_replacement()
    assert trainer._resolve_luffy_teacher_sample_size(template_batch_size=8) == 128

    trainer.luffy_config.warmup_sample_with_replacement = False
    assert trainer._resolve_luffy_teacher_sample_size(template_batch_size=8) == 8


def test_luffy_teacher_only_warmup_samples_teacher_rows_with_replacement(tmp_path: Path):
    trainer = _build_luffy_sampling_trainer(tmp_path, strict_match=True, batch_size=0, batch_size_source="pre_repeat")
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}})
    trainer.global_steps = 0
    trainer.luffy_config = OmegaConf.create(
        {
            "batch_size": 0,
            "batch_size_source": "pre_repeat",
            "strict_match": True,
            "warmup_steps": 10,
            "warmup_teacher_only": True,
            "warmup_batch_size": 8,
            "warmup_sample_with_replacement": True,
        }
    )
    trainer._luffy_teacher_indices_by_pair[("k-other", "Other query")] = [1]
    batch = _build_luffy_template_batch()
    batch.non_tensor_batch["extra_info"][1] = {"knowledge_id": "k-other", "meta_query": "Other query"}

    teacher_batch = trainer._sample_luffy_teacher_batch(batch)

    assert len(teacher_batch) == 8
    stats = batch.meta_info["luffy_teacher_sampling_stats"]
    assert stats["sampled_rows"] == 8
    assert stats["template_batch_size"] == 2
    assert stats["sample_with_replacement"] == 1.0
    assert stats["warmup_sample_with_replacement"] == 1.0
    assert stats["miss_rows"] == 0


def test_luffy_teacher_default_batch_size_zero_uses_pre_repeat_ratio(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 4}}})
    trainer.luffy_config = OmegaConf.create({"batch_size": 0})

    assert trainer._resolve_luffy_teacher_sample_size(template_batch_size=8) == 2


def test_luffy_teacher_default_strict_match_skips_unmatched_rows(tmp_path: Path):
    trainer = _build_luffy_sampling_trainer(tmp_path, strict_match=False)
    trainer.luffy_config = OmegaConf.create({"batch_size": 0, "batch_size_source": "post_repeat"})

    teacher_batch = trainer._sample_luffy_teacher_batch(_build_luffy_template_batch())

    assert len(teacher_batch) == 1
    assert teacher_batch.non_tensor_batch["luffy_teacher_match_status"].tolist() == ["knowledge_meta_query"]
    assert teacher_batch.meta_info["luffy_teacher_sampling_stats"]["match_status_counts"] == {
        "knowledge_meta_query": 1,
        "miss": 1,
    }


def test_luffy_teacher_strict_match_rejects_knowledge_id_only_match(tmp_path: Path):
    trainer = _build_luffy_sampling_trainer(tmp_path, strict_match=True)
    batch = _build_luffy_template_batch()
    batch.non_tensor_batch["extra_info"][0]["meta_query"] = "Different prompt"

    teacher_batch = trainer._sample_luffy_teacher_batch(batch)

    assert teacher_batch is None
    assert batch.meta_info["luffy_teacher_sampling_stats"]["match_status_counts"] == {"miss": 2}


def test_luffy_teacher_can_match_by_sample_hash_and_meta_query_when_knowledge_id_drifted(tmp_path: Path):
    trainer = _build_luffy_sampling_trainer(tmp_path, strict_match=True)
    trainer.luffy_teacher_dataset = DummyLuffyTeacherDataset(
        [
            _build_luffy_teacher_row(
                knowledge_id="old-indexed-id",
                sample_hash="stable-hash",
                meta_query="What changed?",
            )
        ]
    )
    trainer._luffy_teacher_indices_by_pair = {}
    trainer._luffy_teacher_indices_by_knowledge = {"old-indexed-id": [0]}
    trainer._luffy_teacher_indices_by_sample_hash_pair = {("stable-hash", "What changed?"): [0]}
    batch = _build_luffy_template_batch()
    batch.non_tensor_batch["extra_info"][0]["knowledge_id"] = "new-indexed-id"
    batch.non_tensor_batch["extra_info"][0]["sample_hash"] = "stable-hash"

    teacher_batch = trainer._sample_luffy_teacher_batch(batch)

    assert len(teacher_batch) == 1
    assert teacher_batch.non_tensor_batch["luffy_teacher_match_status"].tolist() == ["sample_hash_meta_query"]


def test_luffy_teacher_replacement_drops_matched_on_policy_rows(tmp_path: Path):
    trainer = _build_trainer(tmp_path)
    on_policy_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(48, dtype=torch.long).reshape(8, 6),
            "attention_mask": torch.ones((8, 6), dtype=torch.long),
        },
        non_tensors={"uid": np.array([f"on-{index}" for index in range(8)], dtype=object)},
        meta_info={},
    )
    teacher_batch = DataProto.from_dict(
        tensors={"input_ids": torch.zeros((2, 6), dtype=torch.long)},
        non_tensors={
            "extra_info": np.array(
                [
                    {"source_row_index": 1},
                    {"source_row_index": 5},
                ],
                dtype=object,
            )
        },
        meta_info={},
    )

    kept_batch = trainer._drop_luffy_teacher_source_rows(on_policy_batch, teacher_batch)

    assert len(kept_batch) == 6
    assert kept_batch.non_tensor_batch["uid"].tolist() == ["on-0", "on-2", "on-3", "on-4", "on-6", "on-7"]
