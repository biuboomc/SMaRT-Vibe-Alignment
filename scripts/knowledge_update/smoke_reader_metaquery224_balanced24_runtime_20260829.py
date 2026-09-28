#!/usr/bin/env python3
"""Smoke the balanced24 Reader source and teacher datasets with Qwen3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.utils.dataset.knowledge_update_dataset import KnowledgeUpdateDataset
from verl.utils.dataset.knowledge_update_teacher_trace_dataset import KnowledgeUpdateTeacherTraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    return parser.parse_args()


def source_dataset(
    source_path: Path,
    teacher_path: Path,
    prompt_path: Path,
    tokenizer,
) -> KnowledgeUpdateDataset:
    config = OmegaConf.create(
        {
            "train_files": [str(source_path)],
            "val_files": [],
            "max_prompt_length": 1024,
            "knowledge_update_query_prompt_file": str(prompt_path),
            "knowledge_update_query_prompt_seed": 0,
            "max_query_prompts": 4,
            "knowledge_update_row_order": "qa_major",
            "knowledge_update_teacher_trace_filter_files": [str(teacher_path)],
            "knowledge_update_teacher_trace_filter_apply_to": "all",
            "knowledge_update_default_lora_variant": "knowledge",
        }
    )
    return KnowledgeUpdateDataset(str(source_path), tokenizer, config)


def main() -> int:
    args = parse_args()
    root = args.root
    balanced_root = root / "balanced24_4epoch"
    prompt_path = root / "query_prompts_grounded_process_plus_metaquery_change_224_20260619.json"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)

    balanced_source = balanced_root / "reader_source_train_balanced24_17184.jsonl"
    balanced_teacher = balanced_root / "reader_teacher_train_balanced24_68736.parquet"
    test_source = root / "reader_source_test_200.jsonl"
    test_teacher = root / "reader_teacher_test_800.parquet"

    # Full pair-filter smoke is tokenizer-free so it checks all 68,736 rows quickly.
    balanced_dataset = source_dataset(balanced_source, balanced_teacher, prompt_path, None)
    test_dataset = source_dataset(test_source, test_teacher, prompt_path, tokenizer)

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
    teacher_dataset = KnowledgeUpdateTeacherTraceDataset(
        parquet_files=[str(balanced_teacher)],
        tokenizer=tokenizer,
        config=teacher_config,
        max_samples=64,
    )
    teacher_samples = [teacher_dataset[index] for index in range(min(64, len(teacher_dataset)))]
    teacher_masks_nonempty = all(int(sample["loss_mask"].sum().item()) > 0 for sample in teacher_samples)
    teacher_ids_nonempty = all(str(sample["knowledge_id"]).strip() for sample in teacher_samples)

    gates = {
        "balanced_source_runtime_rows_68736": len(balanced_dataset) == 68736,
        "balanced_source_filter_drops_zero": balanced_dataset.filtered_missing_teacher_pair_rows == 0,
        "test_source_runtime_rows_800": len(test_dataset) == 800,
        "test_source_filter_drops_zero": test_dataset.filtered_missing_teacher_pair_rows == 0,
        "test_qwen_tokenization_getitem": all(test_dataset[index]["prompt"] for index in range(min(32, len(test_dataset)))),
        "teacher_dataset_64_rows": len(teacher_dataset) == 64,
        "teacher_loss_masks_nonempty": teacher_masks_nonempty,
        "teacher_knowledge_ids_nonempty": teacher_ids_nonempty,
    }
    report = {
        "balanced_source_rows": len(balanced_dataset),
        "balanced_filter_drops": balanced_dataset.filtered_missing_teacher_pair_rows,
        "test_source_rows": len(test_dataset),
        "test_filter_drops": test_dataset.filtered_missing_teacher_pair_rows,
        "teacher_smoke_rows": len(teacher_dataset),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    report_path = root / "runtime_smoke_metaquery224_balanced24.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "report": str(report_path), **report}))
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
