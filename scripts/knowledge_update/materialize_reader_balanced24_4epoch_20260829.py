#!/usr/bin/env python3
"""Build a balanced 24+24 teacher schedule from canonical Reader v3 train data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import materialize_imprint_reader_schema_v3_canonical_only_20260829 as base


MIX_PER_TYPE = 24
NO_OP_PER_BATCH = 8
RANDOM_PER_BATCH = 8
EPOCHS = 4
SELECTION_SEED = "reader-v3-balanced24-selection-20260829"
SCHEDULE_SEED = 20260829


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--expected-prompt-count", type=int, default=base.EXPECTED_PROMPT_COUNT)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def cluster_rank(update_type: str, target: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{update_type}:{target}".encode("utf-8")).hexdigest()


def select_balanced_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    counts = Counter(str(row["update_type"]) for row in rows)
    if set(counts) != {"knowledge", "behavior"}:
        raise ValueError(f"Expected knowledge and behavior rows, got {counts}")
    target_per_type = min(counts.values()) // MIX_PER_TYPE * MIX_PER_TYPE
    if target_per_type <= 0:
        raise ValueError("Not enough rows to form one balanced batch")

    reserve_ids: set[str] = set()
    for update_type in ("knowledge", "behavior"):
        type_rows = [row for row in rows if row["update_type"] == update_type]
        reserve_count = len(type_rows) - target_per_type
        target_clusters: dict[str, list[dict[str, Any]]] = {}
        for row in type_rows:
            target_clusters.setdefault(str(row["reader_target"]), []).append(row)
        ranked = sorted(
            target_clusters.items(),
            key=lambda item: cluster_rank(update_type, item[0]),
            reverse=True,
        )
        remaining = reserve_count
        for _, cluster_rows in ranked:
            if remaining == 0:
                break
            if len(cluster_rows) > remaining:
                continue
            reserve_ids.update(str(row["reader_id"]) for row in cluster_rows)
            remaining -= len(cluster_rows)
        if remaining:
            raise AssertionError(
                f"Could not reserve exactly {reserve_count} {update_type} rows without splitting targets"
            )

    balanced = [row for row in rows if str(row["reader_id"]) not in reserve_ids]
    reserve = [row for row in rows if str(row["reader_id"]) in reserve_ids]
    balanced_counts = Counter(str(row["update_type"]) for row in balanced)
    if balanced_counts != {"knowledge": target_per_type, "behavior": target_per_type}:
        raise AssertionError(f"Balanced counts mismatch: {balanced_counts}")
    return balanced, reserve, target_per_type


def write_schedule(path: Path, teacher_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(teacher_path, columns=["update_type", "knowledge_id", "query_type"])
    update_types = table.column("update_type").to_pylist()
    knowledge_indices = [index for index, value in enumerate(update_types) if value == "knowledge"]
    behavior_indices = [index for index, value in enumerate(update_types) if value == "behavior"]
    if len(knowledge_indices) != len(behavior_indices):
        raise AssertionError("Knowledge and behavior teacher row counts differ")
    if len(knowledge_indices) % MIX_PER_TYPE:
        raise AssertionError("Teacher rows per type are not divisible by 24")
    steps_per_epoch = len(knowledge_indices) // MIX_PER_TYPE

    temp = path.with_suffix(path.suffix + ".tmp")
    global_step = 0
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for epoch_index in range(EPOCHS):
            knowledge_epoch = list(knowledge_indices)
            behavior_epoch = list(behavior_indices)
            random.Random(SCHEDULE_SEED + epoch_index * 2).shuffle(knowledge_epoch)
            random.Random(SCHEDULE_SEED + epoch_index * 2 + 1).shuffle(behavior_epoch)
            negative_pools = {
                "knowledge": list(knowledge_indices),
                "behavior": list(behavior_indices),
            }
            random.Random(SCHEDULE_SEED + 1000 + epoch_index * 2).shuffle(negative_pools["knowledge"])
            random.Random(SCHEDULE_SEED + 1000 + epoch_index * 2 + 1).shuffle(negative_pools["behavior"])
            negative_positions = {"knowledge": 0, "behavior": 0}

            def take_negative_indices(update_type: str, count: int, blocked: set[int]) -> list[int]:
                pool = negative_pools[update_type]
                selected: list[int] = []
                attempts = 0
                while len(selected) < count:
                    if attempts > len(pool) * 2:
                        raise AssertionError(f"Could not select {count} {update_type} negative indices")
                    position = negative_positions[update_type] % len(pool)
                    negative_positions[update_type] += 1
                    attempts += 1
                    candidate = pool[position]
                    if candidate in blocked:
                        continue
                    blocked.add(candidate)
                    selected.append(candidate)
                return selected

            for step_in_epoch in range(steps_per_epoch):
                global_step += 1
                start = step_in_epoch * MIX_PER_TYPE
                stop = start + MIX_PER_TYPE
                knowledge_positive = knowledge_epoch[start:stop]
                behavior_positive = behavior_epoch[start:stop]
                blocked = set(knowledge_positive) | set(behavior_positive)
                negative_knowledge = take_negative_indices("knowledge", 8, blocked)
                negative_behavior = take_negative_indices("behavior", 8, blocked)
                no_op_indices = negative_knowledge[:4] + negative_behavior[:4]
                random_indices = negative_knowledge[4:] + negative_behavior[4:]
                row = {
                    "schema_version": "reader_balanced24_teacher_schedule_v1",
                    "global_step": global_step,
                    "epoch": epoch_index + 1,
                    "step_in_epoch": step_in_epoch + 1,
                    "knowledge_teacher_indices": knowledge_positive,
                    "behavior_teacher_indices": behavior_positive,
                    "no_op_teacher_indices": no_op_indices,
                    "random_teacher_indices": random_indices,
                    "knowledge_count": MIX_PER_TYPE,
                    "behavior_count": MIX_PER_TYPE,
                    "no_op_count": NO_OP_PER_BATCH,
                    "random_count": RANDOM_PER_BATCH,
                    "no_op_knowledge_count": 4,
                    "no_op_behavior_count": 4,
                    "random_knowledge_count": 4,
                    "random_behavior_count": 4,
                    "batch_size": MIX_PER_TYPE * 2 + NO_OP_PER_BATCH + RANDOM_PER_BATCH,
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temp, path)
    expected_steps = steps_per_epoch * EPOCHS
    if global_step != expected_steps:
        raise AssertionError(f"Schedule steps {global_step} != {expected_steps}")
    return {
        **base.artifact_record(path, rows=global_step),
        "steps_per_epoch": steps_per_epoch,
        "epochs": EPOCHS,
        "total_steps": global_step,
    }


def type_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "update_type_counts": dict(sorted(Counter(str(row["update_type"]) for row in rows).items())),
        "knowledge_benchmark_counts": dict(
            sorted(
                Counter(
                    str(row.get("source_benchmark", "unknown"))
                    for row in rows
                    if row["update_type"] == "knowledge"
                ).items()
            )
        ),
        "behavior_category_counts": dict(
            sorted(
                Counter(
                    str(row.get("category", "unknown"))
                    for row in rows
                    if row["update_type"] == "behavior"
                ).items()
            )
        ),
        "unique_ids": len({str(row["reader_id"]) for row in rows}),
        "unique_targets": len({str(row["reader_target"]) for row in rows}),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = base.configure_prompt_file(args.prompt_file, expected_count=args.expected_prompt_count)
    rows = list(base.read_jsonl(args.train_source))
    balanced, reserve, target_per_type = select_balanced_rows(rows)

    balanced_source = args.output_dir / f"reader_source_train_balanced24_{len(balanced)}.jsonl"
    reserve_source = args.output_dir / f"reader_source_train_reserve_{len(reserve)}.jsonl"
    balanced_teacher = args.output_dir / (
        f"reader_teacher_train_balanced24_{len(balanced) * base.MAX_QUERY_PROMPTS}.parquet"
    )
    reserve_teacher = args.output_dir / (
        f"reader_teacher_train_reserve_{len(reserve) * base.MAX_QUERY_PROMPTS}.parquet"
    )
    schedule_path = args.output_dir / "teacher_schedule_24knowledge_24behavior_8noop_8random_4epoch.jsonl"

    artifacts = {
        "balanced_source": base.write_jsonl(balanced_source, balanced, "train"),
        "reserve_source": base.write_jsonl(reserve_source, reserve, "reserve"),
        "balanced_teacher": base.write_teacher_parquet(balanced_teacher, balanced, "train"),
        "reserve_teacher": base.write_teacher_parquet(reserve_teacher, reserve, "reserve"),
    }
    pair_validation = {
        "balanced": base.validate_pairs(balanced, balanced_teacher),
        "reserve": base.validate_pairs(reserve, reserve_teacher),
    }
    schedule = write_schedule(schedule_path, balanced_teacher)
    artifacts["schedule"] = schedule

    balanced_targets = {str(row["reader_target"]) for row in balanced}
    reserve_targets = {str(row["reader_target"]) for row in reserve}
    counts = Counter(str(row["update_type"]) for row in balanced)
    manifest = {
        "schema_version": "reader_balanced24_4epoch_metaquery224_v1",
        "source_schema_version": base.SCHEMA_VERSION,
        "input_train_source": base.artifact_record(args.train_source, rows=len(rows)),
        "input_prompt_file": base.artifact_record(args.prompt_file, rows=len(prompts)),
        "settings": {
            "knowledge_per_batch": MIX_PER_TYPE,
            "behavior_per_batch": MIX_PER_TYPE,
            "no_op_per_batch": NO_OP_PER_BATCH,
            "random_per_batch": RANDOM_PER_BATCH,
            "batch_size": 64,
            "epochs": EPOCHS,
            "prompt_pool_size": len(prompts),
            "max_query_prompts": base.MAX_QUERY_PROMPTS,
            "prompt_seed": base.PROMPT_SEED,
            "selection_seed": SELECTION_SEED,
            "schedule_seed": SCHEDULE_SEED,
        },
        "stats": {
            "balanced": type_stats(balanced),
            "reserve": type_stats(reserve),
            "target_per_type": target_per_type,
            "teacher_rows_per_type": target_per_type * base.MAX_QUERY_PROMPTS,
            "steps_per_epoch": schedule["steps_per_epoch"],
            "total_steps": schedule["total_steps"],
        },
        "artifacts": artifacts,
        "pair_validation": pair_validation,
        "gates": {
            "equal_type_counts": counts["knowledge"] == counts["behavior"],
            "knowledge_divisible_by_24": counts["knowledge"] % MIX_PER_TYPE == 0,
            "behavior_divisible_by_24": counts["behavior"] % MIX_PER_TYPE == 0,
            "batch_size_64": MIX_PER_TYPE * 2 + NO_OP_PER_BATCH + RANDOM_PER_BATCH == 64,
            "target_disjoint_from_reserve": not (balanced_targets & reserve_targets),
            "source_partition_complete": len(balanced) + len(reserve) == len(rows),
            "teacher_pairs_exact": all(item["mismatches"] == 0 for item in pair_validation.values()),
            "schedule_4_epochs": schedule["total_steps"] == schedule["steps_per_epoch"] * EPOCHS,
            "prompt_pool_expected_count": len(prompts) == args.expected_prompt_count,
            "prompt_types_unique": len({str(prompt["type"]) for prompt in prompts}) == len(prompts),
            "prompt_texts_unique": len({str(prompt["prompt"]) for prompt in prompts}) == len(prompts),
        },
    }
    manifest["all_gates_pass"] = all(manifest["gates"].values())
    manifest_path = args.output_dir / "manifest_balanced24_4epoch.json"
    base.atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "balanced_knowledge": counts["knowledge"],
                "balanced_behavior": counts["behavior"],
                "reserve": len(reserve),
                "teacher_rows": len(balanced) * base.MAX_QUERY_PROMPTS,
                "prompt_pool_size": len(prompts),
                "steps_per_epoch": schedule["steps_per_epoch"],
                "epochs": EPOCHS,
                "total_steps": schedule["total_steps"],
                "all_gates_pass": manifest["all_gates_pass"],
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
