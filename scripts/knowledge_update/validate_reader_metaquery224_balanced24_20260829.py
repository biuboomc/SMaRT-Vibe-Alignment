#!/usr/bin/env python3
"""Validate the Reader v3 metaquery224 balanced24 training snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--original-prompt-file", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_prompts(prompts: list[dict[str, Any]], qa_index: int) -> list[dict[str, Any]]:
    selected = list(prompts)
    random.Random(qa_index).shuffle(selected)
    return selected[:4]


def validate_targets(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row["update_type"] == "knowledge":
            if row["reader_target"] != row["canonical_fact"] or row["reader_target_kind"] != "canonical_fact":
                return False
        elif row["update_type"] == "behavior":
            if row["reader_target"] != row["canonical_behavior"] or row["reader_target_kind"] != "canonical_behavior":
                return False
        else:
            return False
    return True


def validate_source_teacher(
    source_path: Path,
    teacher_path: Path,
    prompts: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    rows = read_jsonl(source_path)
    columns = [
        "knowledge_id",
        "sample_hash",
        "meta_query",
        "query_type",
        "update_type",
        "reader_target",
        "supervised_answer",
        "actor_sft_target",
        "teacher_trace",
    ]
    table = pq.read_table(teacher_path, columns=columns)
    data = table.to_pydict()
    actual_pairs = Counter(zip(data["knowledge_id"], data["sample_hash"], data["meta_query"]))
    expected_pairs: list[tuple[str, str, str]] = []
    expected_types: list[str] = []
    for qa_index, row in enumerate(rows):
        for prompt in selected_prompts(prompts, qa_index):
            expected_pairs.append((str(row["knowledge_id"]), str(row["sample_hash"]), str(prompt["prompt"])))
            expected_types.append(str(prompt["type"]))
    expected_counter = Counter(expected_pairs)
    prompt_usage = Counter(data["query_type"])
    target_columns_exact = all(
        reader_target == supervised == actor_target == teacher_trace
        for reader_target, supervised, actor_target, teacher_trace in zip(
            data["reader_target"],
            data["supervised_answer"],
            data["actor_sft_target"],
            data["teacher_trace"],
        )
    )
    stats = {
        "source_rows": len(rows),
        "teacher_rows": table.num_rows,
        "source_type_counts": dict(sorted(Counter(str(row["update_type"]) for row in rows).items())),
        "teacher_type_counts": dict(sorted(Counter(data["update_type"]).items())),
        "prompt_types_used": len(prompt_usage),
        "prompt_usage_min": min(prompt_usage.values()) if prompt_usage else 0,
        "prompt_usage_max": max(prompt_usage.values()) if prompt_usage else 0,
        "prompt_usage_mean": sum(prompt_usage.values()) / max(len(prompt_usage), 1),
    }
    gates = {
        "four_teacher_rows_per_source": table.num_rows == len(rows) * 4,
        "source_targets_canonical": validate_targets(rows),
        "teacher_pairs_exact": actual_pairs == expected_counter,
        "teacher_query_types_exact": Counter(data["query_type"]) == Counter(expected_types),
        "teacher_targets_canonical": target_columns_exact,
        "teacher_pairs_unique": all(count == 1 for count in actual_pairs.values()),
    }
    return stats, gates


def validate_schedule(schedule_path: Path, teacher_path: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    update_types = pq.read_table(teacher_path, columns=["update_type"]).column("update_type").to_pylist()
    expected_by_type = {
        update_type: {index for index, value in enumerate(update_types) if value == update_type}
        for update_type in ("knowledge", "behavior")
    }
    rows = read_jsonl(schedule_path)
    coverage: dict[int, dict[str, Counter[int]]] = defaultdict(
        lambda: {"knowledge": Counter(), "behavior": Counter()}
    )
    row_shape_ok = True
    row_index_type_ok = True
    negative_type_mix_ok = True
    row_indices_disjoint = True
    sequential_steps_ok = True
    for expected_step, row in enumerate(rows, start=1):
        epoch = int(row["epoch"])
        knowledge_indices = [int(value) for value in row["knowledge_teacher_indices"]]
        behavior_indices = [int(value) for value in row["behavior_teacher_indices"]]
        no_op_indices = [int(value) for value in row["no_op_teacher_indices"]]
        random_indices = [int(value) for value in row["random_teacher_indices"]]
        row_shape_ok &= (
            len(knowledge_indices) == row["knowledge_count"] == 24
            and len(behavior_indices) == row["behavior_count"] == 24
            and len(no_op_indices) == row["no_op_count"] == 8
            and len(random_indices) == row["random_count"] == 8
            and row["batch_size"] == 64
        )
        sequential_steps_ok &= int(row["global_step"]) == expected_step
        row_index_type_ok &= all(update_types[index] == "knowledge" for index in knowledge_indices)
        row_index_type_ok &= all(update_types[index] == "behavior" for index in behavior_indices)
        negative_type_mix_ok &= Counter(update_types[index] for index in no_op_indices) == {
            "knowledge": 4,
            "behavior": 4,
        }
        negative_type_mix_ok &= Counter(update_types[index] for index in random_indices) == {
            "knowledge": 4,
            "behavior": 4,
        }
        row_groups = [set(knowledge_indices), set(behavior_indices), set(no_op_indices), set(random_indices)]
        row_indices_disjoint &= sum(len(group) for group in row_groups) == len(set().union(*row_groups))
        coverage[epoch]["knowledge"].update(knowledge_indices)
        coverage[epoch]["behavior"].update(behavior_indices)
    exact_once_per_epoch = True
    for epoch in range(1, 5):
        for update_type in ("knowledge", "behavior"):
            counter = coverage[epoch][update_type]
            exact_once_per_epoch &= set(counter) == expected_by_type[update_type]
            exact_once_per_epoch &= all(count == 1 for count in counter.values())
    stats = {
        "schedule_rows": len(rows),
        "epochs": sorted(coverage),
        "steps_per_epoch": dict(sorted(Counter(int(row["epoch"]) for row in rows).items())),
        "positive_teacher_rows_per_type": {key: len(value) for key, value in expected_by_type.items()},
    }
    gates = {
        "schedule_rows_5728": len(rows) == 5728,
        "schedule_row_mix_exact": row_shape_ok,
        "schedule_steps_sequential": sequential_steps_ok,
        "schedule_indices_match_update_type": row_index_type_ok,
        "negative_rows_are_4_knowledge_4_behavior_each": negative_type_mix_ok,
        "all_64_teacher_indices_disjoint_within_step": row_indices_disjoint,
        "each_positive_teacher_row_once_per_type_per_epoch": exact_once_per_epoch,
    }
    return stats, gates


def main() -> int:
    args = parse_args()
    root = args.root
    balanced_root = root / "balanced24_4epoch"
    copied_prompt_path = root / args.original_prompt_file.name
    with args.original_prompt_file.open("r", encoding="utf-8") as handle:
        prompt_payload = json.load(handle)
    prompts = prompt_payload["prompts"]

    paths = {
        "train": (
            root / "reader_source_train_17975.jsonl",
            root / "reader_teacher_train_71900.parquet",
        ),
        "val": (
            root / "reader_source_val_946.jsonl",
            root / "reader_teacher_val_3784.parquet",
        ),
        "test": (
            root / "reader_source_test_200.jsonl",
            root / "reader_teacher_test_800.parquet",
        ),
        "balanced": (
            balanced_root / "reader_source_train_balanced24_17184.jsonl",
            balanced_root / "reader_teacher_train_balanced24_68736.parquet",
        ),
        "reserve": (
            balanced_root / "reader_source_train_reserve_791.jsonl",
            balanced_root / "reader_teacher_train_reserve_3164.parquet",
        ),
    }

    split_stats: dict[str, Any] = {}
    split_gates: dict[str, bool] = {}
    loaded_sources: dict[str, list[dict[str, Any]]] = {}
    for name, (source_path, teacher_path) in paths.items():
        loaded_sources[name] = read_jsonl(source_path)
        stats, gates = validate_source_teacher(source_path, teacher_path, prompts)
        split_stats[name] = stats
        split_gates.update({f"{name}_{gate}": value for gate, value in gates.items()})

    train_ids = {str(row["reader_id"]) for row in loaded_sources["train"]}
    val_ids = {str(row["reader_id"]) for row in loaded_sources["val"]}
    test_ids = {str(row["reader_id"]) for row in loaded_sources["test"]}
    train_targets = {str(row["reader_target"]) for row in loaded_sources["train"]}
    val_targets = {str(row["reader_target"]) for row in loaded_sources["val"]}
    test_targets = {str(row["reader_target"]) for row in loaded_sources["test"]}
    balanced_ids = {str(row["reader_id"]) for row in loaded_sources["balanced"]}
    reserve_ids = {str(row["reader_id"]) for row in loaded_sources["reserve"]}
    balanced_targets = {str(row["reader_target"]) for row in loaded_sources["balanced"]}
    reserve_targets = {str(row["reader_target"]) for row in loaded_sources["reserve"]}

    test_counts = Counter(str(row["update_type"]) for row in loaded_sources["test"])
    balanced_counts = Counter(str(row["update_type"]) for row in loaded_sources["balanced"])
    reserve_counts = Counter(str(row["update_type"]) for row in loaded_sources["reserve"])
    structural_gates = {
        "prompt_pool_exactly_224": len(prompts) == 224,
        "prompt_types_unique": len({str(row["type"]) for row in prompts}) == 224,
        "prompt_texts_unique": len({str(row["prompt"]) for row in prompts}) == 224,
        "prompt_copy_sha_exact": sha256(copied_prompt_path) == sha256(args.original_prompt_file),
        "canonical_splits_id_disjoint": not ((train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)),
        "canonical_splits_target_disjoint": not (
            (train_targets & val_targets) | (train_targets & test_targets) | (val_targets & test_targets)
        ),
        "test_exactly_100_per_type": test_counts == {"knowledge": 100, "behavior": 100},
        "balanced_exactly_8592_per_type": balanced_counts == {"knowledge": 8592, "behavior": 8592},
        "reserve_exactly_5_knowledge_786_behavior": reserve_counts == {"knowledge": 5, "behavior": 786},
        "balanced_reserve_partition_train": not (balanced_ids & reserve_ids) and balanced_ids | reserve_ids == train_ids,
        "balanced_reserve_targets_disjoint": not (balanced_targets & reserve_targets),
    }

    schedule_stats, schedule_gates = validate_schedule(
        balanced_root / "teacher_schedule_24knowledge_24behavior_8noop_8random_4epoch.jsonl",
        balanced_root / "reader_teacher_train_balanced24_68736.parquet",
    )
    all_gates = {**split_gates, **structural_gates, **schedule_gates}
    report = {
        "root": str(root),
        "prompt_file": {
            "original": str(args.original_prompt_file),
            "copy": str(copied_prompt_path),
            "sha256": sha256(copied_prompt_path),
            "count": len(prompts),
            "category_counts": dict(sorted(Counter(str(row.get("category") or row.get("source")) for row in prompts).items())),
        },
        "split_stats": split_stats,
        "schedule_stats": schedule_stats,
        "gates": all_gates,
        "all_gates_pass": all(all_gates.values()),
    }
    report_path = root / "validation_metaquery224_balanced24_4epoch.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "report": str(report_path), **schedule_stats}))
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
