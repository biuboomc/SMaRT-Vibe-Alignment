#!/usr/bin/env python3
"""Exercise the balanced24 schedule-aware LUFFY sampler without building LoRAs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.protocol import DataProto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    return parser.parse_args()


def load_trainer_module(patch_root: Path):
    trainer_path = patch_root / "verl" / "trainer" / "ppo" / "knowledge_update_trainer.py"
    spec = importlib.util.spec_from_file_location("reader_balanced24_runtime_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load patched trainer: {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_template_batch(batch_size: int = 64, sequence_length: int = 2048, response_length: int = 256):
    tensors = {
        "input_ids": torch.zeros((batch_size, sequence_length), dtype=torch.long),
        "attention_mask": torch.zeros((batch_size, sequence_length), dtype=torch.long),
        "position_ids": torch.zeros((batch_size, sequence_length), dtype=torch.long),
        "responses": torch.zeros((batch_size, response_length), dtype=torch.long),
        "response_mask": torch.zeros((batch_size, response_length), dtype=torch.long),
        "token_level_scores": torch.zeros((batch_size, response_length), dtype=torch.float32),
        "token_level_rewards": torch.zeros((batch_size, response_length), dtype=torch.float32),
    }
    non_tensors = {
        "knowledge_id": np.array([f"template-{index}" for index in range(batch_size)], dtype=object),
        "extra_info": np.array([{} for _ in range(batch_size)], dtype=object),
        "uid": np.array([f"template-{index}" for index in range(batch_size)], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={})


def main() -> int:
    args = parse_args()
    module = load_trainer_module(args.patch_root)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    balanced_root = args.data_root / "balanced24_4epoch"
    source_path = balanced_root / "reader_source_train_balanced24_17184.jsonl"
    teacher_path = balanced_root / "reader_teacher_train_balanced24_68736.parquet"
    schedule_path = balanced_root / "teacher_schedule_24knowledge_24behavior_8noop_8random_4epoch.jsonl"

    teacher_config = OmegaConf.create(
        {
            "messages_key": "messages",
            "max_length": 2048,
            "truncation": "right",
            "pad_mode": "right",
            "ignore_input_ids_mismatch": True,
            "knowledge_id_key": "knowledge_id",
            "knowledge_lora_path_key": "knowledge_lora_path",
            "knowledge_lora_root_dir": None,
            "knowledge_lora_variant": "knowledge",
            "knowledge_lora_rank_local_root": True,
            "target_probs_key": None,
        }
    )
    teacher_dataset = module.KnowledgeUpdateTeacherTraceDataset(
        parquet_files=[str(teacher_path)],
        tokenizer=tokenizer,
        config=teacher_config,
        max_samples=-1,
    )

    trainer = object.__new__(module.KnowledgeUpdatePPOTrainer)
    trainer.tokenizer = tokenizer
    trainer.processor = None
    trainer.config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 2048},
            "actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )
    trainer.luffy_config = OmegaConf.create(
        {
            "batch_size": 64,
            "batch_size_source": "pre_repeat",
            "warmup_steps": 5728,
            "warmup_batch_size": 64,
            "warmup_teacher_only": True,
            "warmup_sample_with_replacement": False,
        }
    )
    trainer.luffy_teacher_dataset = teacher_dataset
    trainer.luffy_teacher_max_length = 2048
    trainer.luffy_teacher_seed = 0
    trainer.luffy_teacher_rng = np.random.default_rng(0)
    trainer.luffy_negative_sample_seed = 0
    trainer.luffy_negative_rng = np.random.default_rng(0)
    trainer.luffy_no_change_responses = ["No update is present."]
    trainer.luffy_random_change_responses = ["A different update is present."]
    trainer.variant_reward_keys = ("process_reward",)
    trainer.luffy_teacher_advantage_value = 1.0
    trainer.luffy_per_type_schedule_file = str(schedule_path)
    trainer.luffy_per_type_source_file = str(source_path)
    trainer.luffy_per_type_schedule_strict = True
    trainer._luffy_per_type_schedule = []
    trainer._luffy_per_type_source_by_knowledge = {}
    trainer.enable_ephemeral_lora = False
    trainer.use_ephemeral_lora_for_loss = False
    trainer.default_lora_variant = "knowledge"
    trainer.ephemeral_lora_variant_field = "ephemeral_lora_variant"
    trainer.ephemeral_lora_path_field = "ephemeral_lora_path"
    trainer.ephemeral_lora_request_field = "ephemeral_lora_request"
    trainer.ephemeral_lora_variants_field = "ephemeral_lora_variants"
    trainer.global_steps = 1

    trainer._load_luffy_per_type_schedule()
    first_schedule = trainer._current_luffy_per_type_schedule_row()
    trainer.global_steps = 5728
    last_schedule = trainer._current_luffy_per_type_schedule_row()
    trainer.global_steps = 1
    sampled = trainer._sample_luffy_teacher_batch(build_template_batch())
    if sampled is None:
        raise AssertionError("Schedule-aware LUFFY sampler returned no batch")

    non_tensors = sampled.non_tensor_batch
    sample_types = [str(value) for value in non_tensors["luffy_teacher_sample_type"].tolist()]
    update_types = [str(value) for value in non_tensors["update_type"].tolist()]
    variants = [str(value) for value in non_tensors["luffy_teacher_lora_variant"].tolist()]
    mounts = [bool(value) for value in non_tensors["luffy_teacher_mount_lora"].tolist()]
    extra_infos = non_tensors["extra_info"].tolist()
    teacher_indices = [int(value["teacher_dataset_index"]) for value in extra_infos]
    request_items = non_tensors["ephemeral_lora_request"].tolist()

    gates = {
        "schedule_rows_5728": len(trainer._luffy_per_type_schedule) == 5728,
        "first_schedule_step_1": int(first_schedule["global_step"]) == 1,
        "last_schedule_step_5728": int(last_schedule["global_step"]) == 5728,
        "sampled_batch_64": len(sampled) == 64,
        "sample_types_48_8_8": Counter(sample_types) == {
            "changed": 48,
            "no_change": 8,
            "random_change": 8,
        },
        "positive_update_types_24_24": Counter(update_types[:48]) == {
            "knowledge": 24,
            "behavior": 24,
        },
        "no_op_update_types_4_4": Counter(update_types[48:56]) == {
            "knowledge": 4,
            "behavior": 4,
        },
        "random_update_types_4_4": Counter(update_types[56:64]) == {
            "knowledge": 4,
            "behavior": 4,
        },
        "variants_48_8_8": Counter(variants) == {
            "knowledge": 48,
            "no_op": 8,
            "random": 8,
        },
        "mounted_rows_56": sum(mounts) == 56,
        "build_requests_56": sum(isinstance(value, dict) for value in request_items) == 56,
        "teacher_indices_unique_64": len(set(teacher_indices)) == 64,
        "all_source_rows_have_eight_variants": all(len(value.get("variants", [])) == 8 for value in extra_infos),
        "schedule_metric_active": sampled.meta_info["luffy_teacher_sampling_stats"]["per_type_schedule_active"] == 1.0,
    }
    report = {
        "schedule_rows": len(trainer._luffy_per_type_schedule),
        "sample_type_counts": dict(Counter(sample_types)),
        "positive_update_type_counts": dict(Counter(update_types[:48])),
        "no_op_update_type_counts": dict(Counter(update_types[48:56])),
        "random_update_type_counts": dict(Counter(update_types[56:64])),
        "variant_counts": dict(Counter(variants)),
        "mounted_rows": sum(mounts),
        "build_requests": sum(isinstance(value, dict) for value in request_items),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    report_path = args.data_root / "runtime_smoke_luffy_balanced24_schedule.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}))
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
