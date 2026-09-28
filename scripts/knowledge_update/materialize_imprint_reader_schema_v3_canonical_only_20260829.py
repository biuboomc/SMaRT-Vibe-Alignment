#!/usr/bin/env python3
"""Materialize canonical knowledge+behavior data with the original 224 meta-query pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "imprint_reader_v3_canonical_only_metaquery224_20260829"
SYSTEM_PROMPT = "You are a helpful and introspective assistant."
SAMPLE_HASH_FIELDS = ("title", "category", "subcategory", "context", "question", "answer")
PROMPT_SEED = 0
MAX_QUERY_PROMPTS = 4
VAL_FRACTION = 0.05
SPLIT_SEED = "imprint-reader-v2-val-20260828"
TEST_SEED = "imprint-reader-v2-test100-20260828"
EXPECTED_PROMPT_COUNT = 224
ACTIVE_PROMPTS: list[dict[str, Any]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-jsonl", required=True, type=Path)
    parser.add_argument("--behavior-jsonl", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-knowledge", type=int, default=9148)
    parser.add_argument("--expected-behavior", type=int, default=9973)
    parser.add_argument("--expected-prompt-count", type=int, default=EXPECTED_PROMPT_COUNT)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--test-per-type", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def build_sample_hash(sample: dict[str, Any]) -> str:
    payload = {field: str(sample.get(field, "")) for field in SAMPLE_HASH_FIELDS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_reader_target(canonical_target: str) -> str:
    canonical_target = str(canonical_target or "").strip()
    if not canonical_target:
        raise ValueError("Reader targets require a non-empty canonical target")
    return canonical_target


def normalize_variants(row: dict[str, Any], knowledge_id: str) -> list[dict[str, Any]]:
    variants = row.get("variants")
    if not isinstance(variants, list) or len(variants) != 8:
        raise ValueError(f"Expected exactly eight variants for {knowledge_id}")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for variant_index, raw_variant in enumerate(variants):
        if not isinstance(raw_variant, dict):
            raise TypeError(f"Variant {variant_index} for {knowledge_id} is not an object")
        variant = dict(raw_variant)
        question = str(variant.get("question", "")).strip()
        answer = str(variant.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"Empty variant QA for {knowledge_id}:{variant_index}")
        key = (question, answer)
        if key in seen:
            raise ValueError(f"Duplicate variant QA within {knowledge_id}")
        seen.add(key)
        variant.setdefault("knowledge_id", knowledge_id)
        variant.setdefault("rewrite_variant_id", variant_index)
        normalized.append(variant)
    return normalized


def normalize_knowledge_rows(path: Path) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in read_jsonl(path):
        if str(raw.get("mix_stage", "")).strip().lower() != "fact":
            continue
        knowledge_id = str(raw.get("knowledge_id", "")).strip()
        if not knowledge_id:
            raise ValueError("Knowledge fact row is missing knowledge_id")
        if knowledge_id in selected:
            raise ValueError(f"Duplicate knowledge fact row: {knowledge_id}")
        row = dict(raw)
        canonical = str(row.get("canonical_fact", "")).strip()
        variants = normalize_variants(row, knowledge_id)
        row.update(
            {
                "reader_schema_version": SCHEMA_VERSION,
                "reader_id": knowledge_id,
                "knowledge_id": knowledge_id,
                "behavior_id": "",
                "update_type": "knowledge",
                "canonical_behavior": "",
                "canonical_fact": canonical,
                "reader_target": canonical_reader_target(canonical),
                "reader_target_kind": "canonical_fact",
                "mix_stage": "fact",
                "lora_variant": "knowledge",
                "default_lora_variant": "knowledge",
                "ephemeral_lora_variant": "knowledge",
                "variants": variants,
            }
        )
        row["sample_hash"] = build_sample_hash(row)
        row["target_field_sources"] = {
            "question": "question",
            "answer": "answer",
            "canonical_fact": "canonical_fact",
            "canonical_behavior": "",
            "reader_target": "canonical_fact",
        }
        selected[knowledge_id] = row
        order.append(knowledge_id)
    return [selected[knowledge_id] for knowledge_id in order]


def normalize_behavior_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in read_jsonl(path):
        behavior_id = str(raw.get("behavior_id") or raw.get("knowledge_id") or "").strip()
        if not behavior_id:
            raise ValueError("Behavior row is missing behavior_id")
        if behavior_id in seen:
            raise ValueError(f"Duplicate behavior row: {behavior_id}")
        seen.add(behavior_id)
        row = dict(raw)
        canonical = str(row.get("canonical_behavior", "")).strip()
        variants = normalize_variants(row, behavior_id)
        first_variant = variants[0]
        row.update(
            {
                "reader_schema_version": SCHEMA_VERSION,
                "reader_id": behavior_id,
                "knowledge_id": behavior_id,
                "behavior_id": behavior_id,
                "update_type": "behavior",
                "canonical_behavior": canonical,
                "canonical_fact": canonical,
                "question": str(first_variant["question"]),
                "answer": str(first_variant["answer"]),
                "knowledge_question": str(first_variant["question"]),
                "knowledge_answer": str(first_variant["answer"]),
                "reader_target": canonical_reader_target(canonical),
                "reader_target_kind": "canonical_behavior",
                "mix_stage": "behavior",
                "lora_variant": "knowledge",
                "default_lora_variant": "knowledge",
                "ephemeral_lora_variant": "knowledge",
                "variants": variants,
            }
        )
        row["sample_hash"] = build_sample_hash(row)
        row["target_field_sources"] = {
            "question": "variants[0].question",
            "answer": "variants[0].answer",
            "canonical_fact": "canonical_behavior_legacy_alias",
            "canonical_behavior": "canonical_behavior",
            "reader_target": "canonical_behavior",
        }
        rows.append(row)
    return rows


def split_stratum(row: dict[str, Any]) -> tuple[str, str]:
    update_type = str(row["update_type"])
    if update_type == "knowledge":
        return update_type, str(row.get("source_benchmark", "unknown"))
    return update_type, str(row.get("category", "unknown"))


def stable_split(rows: list[dict[str, Any]], val_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[split_stratum(row)].append(row)
    val_ids: set[str] = set()
    for stratum_rows in strata.values():
        target_clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in stratum_rows:
            target_clusters[str(row["reader_target"])].append(row)
        ranked_clusters = sorted(
            target_clusters.items(),
            key=lambda item: hashlib.sha256(
                f"{SPLIT_SEED}:{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
        desired_val_count = max(1, round(len(stratum_rows) * val_fraction))
        selected_count = 0
        for _, cluster_rows in ranked_clusters:
            if selected_count >= desired_val_count:
                break
            val_ids.update(str(row["reader_id"]) for row in cluster_rows)
            selected_count += len(cluster_rows)
    train_rows = [row for row in rows if str(row["reader_id"]) not in val_ids]
    val_rows = [row for row in rows if str(row["reader_id"]) in val_ids]
    return train_rows, val_rows


def proportional_quotas(counts: dict[tuple[str, str], int], total: int) -> dict[tuple[str, str], int]:
    if total < len(counts):
        raise ValueError(f"Cannot cover {len(counts)} strata with only {total} test rows")
    population = sum(counts.values())
    raw = {key: counts[key] * total / population for key in counts}
    quotas = {key: max(1, int(raw[key])) for key in counts}
    while sum(quotas.values()) < total:
        key = max(counts, key=lambda item: (raw[item] - quotas[item], counts[item], item))
        quotas[key] += 1
    while sum(quotas.values()) > total:
        candidates = [key for key in counts if quotas[key] > 1]
        if not candidates:
            raise AssertionError("Unable to reduce stratified test quotas")
        key = min(candidates, key=lambda item: (raw[item] - quotas[item], -quotas[item], item))
        quotas[key] -= 1
    return quotas


def select_exact_test_ids(rows: list[dict[str, Any]], test_per_type: int) -> set[str]:
    if test_per_type <= 0:
        raise ValueError("test_per_type must be positive")
    selected_ids: set[str] = set()
    for update_type in ("knowledge", "behavior"):
        type_rows = [row for row in rows if row["update_type"] == update_type]
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in type_rows:
            strata[split_stratum(row)].append(row)
        quotas = proportional_quotas({key: len(value) for key, value in strata.items()}, test_per_type)
        type_selected = 0
        for stratum, stratum_rows in sorted(strata.items()):
            target_clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in stratum_rows:
                target_clusters[str(row["reader_target"])].append(row)
            ranked_clusters = sorted(
                target_clusters.items(),
                key=lambda item: hashlib.sha256(
                    f"{TEST_SEED}:{stratum}:{item[0]}".encode("utf-8")
                ).hexdigest(),
            )
            quota = quotas[stratum]
            stratum_selected = 0
            for _, cluster_rows in ranked_clusters:
                cluster_size = len(cluster_rows)
                if stratum_selected + cluster_size > quota:
                    continue
                selected_ids.update(str(row["reader_id"]) for row in cluster_rows)
                stratum_selected += cluster_size
                if stratum_selected == quota:
                    break
            if stratum_selected != quota:
                raise AssertionError(
                    f"Could not select exactly {quota} test rows from stratum {stratum}; got {stratum_selected}"
                )
            type_selected += stratum_selected
        if type_selected != test_per_type:
            raise AssertionError(f"Expected {test_per_type} {update_type} test rows, got {type_selected}")
    return selected_ids


def write_jsonl(path: Path, rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for source_index, base_row in enumerate(rows):
            row = dict(base_row)
            row["source_split"] = split
            row["reader_source_index"] = source_index
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)
    return artifact_record(path, rows=len(rows))


def load_prompt_file(path: Path, expected_count: int = EXPECTED_PROMPT_COUNT) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(raw_prompts, list):
        raise ValueError(f"Prompt file must contain a prompts list: {path}")
    if len(raw_prompts) != expected_count:
        raise ValueError(f"Prompt count {len(raw_prompts)} != {expected_count}: {path}")

    normalized: list[dict[str, Any]] = []
    for index, raw_prompt in enumerate(raw_prompts):
        if not isinstance(raw_prompt, dict):
            raise TypeError(f"Prompt {index} is not an object: {path}")
        prompt = str(raw_prompt.get("prompt", "")).strip()
        query_type = str(raw_prompt.get("type", "")).strip()
        if not prompt or not query_type:
            raise ValueError(f"Prompt {index} is missing type or prompt: {path}")
        source = str(raw_prompt.get("source", "external_prompt_pool")).strip() or "external_prompt_pool"
        category = str(raw_prompt.get("category", "")).strip() or source
        family = str(raw_prompt.get("family", "")).strip() or category
        normalized_prompt = dict(raw_prompt)
        normalized_prompt.update(
            {
                "type": query_type,
                "prompt": prompt,
                "source": source,
                "category": category,
                "family": family,
            }
        )
        normalized.append(normalized_prompt)

    query_types = [str(prompt["type"]) for prompt in normalized]
    prompt_texts = [str(prompt["prompt"]) for prompt in normalized]
    if len(set(query_types)) != len(query_types):
        raise ValueError(f"Prompt types are not unique: {path}")
    if len(set(prompt_texts)) != len(prompt_texts):
        raise ValueError(f"Prompt texts are not unique: {path}")
    return normalized


def set_active_prompts(prompts: list[dict[str, Any]]) -> None:
    global ACTIVE_PROMPTS
    ACTIVE_PROMPTS = [dict(prompt) for prompt in prompts]


def configure_prompt_file(path: Path, expected_count: int = EXPECTED_PROMPT_COUNT) -> list[dict[str, Any]]:
    prompts = load_prompt_file(path, expected_count=expected_count)
    set_active_prompts(prompts)
    return prompts


def prompt_rows() -> list[dict[str, Any]]:
    if ACTIVE_PROMPTS is None:
        raise RuntimeError("Prompt pool is not configured; call configure_prompt_file first")
    return [dict(prompt) for prompt in ACTIVE_PROMPTS]


def select_prompts(source_index: int, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = list(prompts)
    random.Random(PROMPT_SEED + source_index).shuffle(selected)
    return selected[:MAX_QUERY_PROMPTS]


def teacher_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("reader_schema_version", pa.string()),
            pa.field("reader_id", pa.string()),
            pa.field("knowledge_id", pa.string()),
            pa.field("behavior_id", pa.string()),
            pa.field("update_type", pa.string()),
            pa.field("category", pa.string()),
            pa.field("subcategory", pa.string()),
            pa.field("title", pa.string()),
            pa.field("context", pa.string()),
            pa.field("question", pa.string()),
            pa.field("answer", pa.string()),
            pa.field("canonical_fact", pa.string()),
            pa.field("canonical_behavior", pa.string()),
            pa.field("reader_target", pa.string()),
            pa.field("reader_target_kind", pa.string()),
            pa.field("source_benchmark", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("source_question_key", pa.string()),
            pa.field("rewrite_group_key", pa.string()),
            pa.field("source_split", pa.string()),
            pa.field("lora_variant", pa.string()),
            pa.field("default_lora_variant", pa.string()),
            pa.field("ephemeral_lora_variant", pa.string()),
            pa.field("sample_hash", pa.string()),
            pa.field("meta_query_family", pa.string()),
            pa.field("query_type", pa.string()),
            pa.field("query_index", pa.int64()),
            pa.field("meta_query", pa.string()),
            pa.field("supervised_answer", pa.string()),
            pa.field("actor_sft_target", pa.string()),
            pa.field("actor_target_format", pa.string()),
            pa.field("target_source", pa.string()),
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                        ]
                    )
                ),
            ),
            pa.field("teacher_trace", pa.string()),
            pa.field("teacher_model", pa.string()),
            pa.field("judge_process_reward", pa.float64()),
            pa.field("eval_prompt_count", pa.int64()),
            pa.field("eval_query_types", pa.list_(pa.string())),
            pa.field("mix_stage", pa.string()),
        ]
    )


def build_teacher_row(
    row: dict[str, Any],
    split: str,
    query_index: int,
    prompt: dict[str, Any],
    all_query_types: list[str],
) -> dict[str, Any]:
    target = str(row["reader_target"])
    meta_query = str(prompt["prompt"])
    return {
        "reader_schema_version": SCHEMA_VERSION,
        "reader_id": str(row["reader_id"]),
        "knowledge_id": str(row["knowledge_id"]),
        "behavior_id": str(row.get("behavior_id", "")),
        "update_type": str(row["update_type"]),
        "category": str(row.get("category", "")),
        "subcategory": str(row.get("subcategory", "")),
        "title": str(row.get("title", "")),
        "context": str(row.get("context", "")),
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "canonical_fact": str(row.get("canonical_fact", "")),
        "canonical_behavior": str(row.get("canonical_behavior", "")),
        "reader_target": target,
        "reader_target_kind": str(row["reader_target_kind"]),
        "source_benchmark": str(row.get("source_benchmark", "")),
        "source_id": str(row.get("source_id", "")),
        "source_question_key": str(row.get("source_question_key", "")),
        "rewrite_group_key": str(row.get("rewrite_group_key", "")),
        "source_split": split,
        "lora_variant": "knowledge",
        "default_lora_variant": "knowledge",
        "ephemeral_lora_variant": "knowledge",
        "sample_hash": str(row["sample_hash"]),
        "meta_query_family": str(prompt["family"]),
        "query_type": str(prompt["type"]),
        "query_index": query_index,
        "meta_query": meta_query,
        "supervised_answer": target,
        "actor_sft_target": target,
        "actor_target_format": "reader_v3_canonical_only",
        "target_source": str(row["reader_target_kind"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": meta_query},
            {"role": "assistant", "content": target},
        ],
        "teacher_trace": target,
        "teacher_model": "deterministic_reader_v2_target",
        "judge_process_reward": 1.0,
        "eval_prompt_count": len(all_query_types),
        "eval_query_types": all_query_types,
        "mix_stage": str(row["mix_stage"]),
    }


def write_teacher_parquet(path: Path, rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    prompts = prompt_rows()
    all_query_types = [str(prompt["type"]) for prompt in prompts]
    schema = teacher_schema()
    temp = path.with_suffix(path.suffix + ".tmp")
    writer = pq.ParquetWriter(temp, schema=schema, compression="zstd")
    buffer: list[dict[str, Any]] = []
    teacher_count = 0
    try:
        for source_index, row in enumerate(rows):
            for query_index, prompt in enumerate(select_prompts(source_index, prompts)):
                buffer.append(build_teacher_row(row, split, query_index, prompt, all_query_types))
                teacher_count += 1
                if len(buffer) >= 4096:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    finally:
        writer.close()
    os.replace(temp, path)
    expected = len(rows) * MAX_QUERY_PROMPTS
    if teacher_count != expected:
        raise AssertionError(f"Teacher row mismatch for {path}: {teacher_count} != {expected}")
    return artifact_record(path, rows=teacher_count)


def validate_pairs(source_rows: list[dict[str, Any]], teacher_path: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    table = pq.read_table(teacher_path, columns=["knowledge_id", "sample_hash", "meta_query"])
    teacher_pairs = Counter(
        zip(
            table.column("knowledge_id").to_pylist(),
            table.column("sample_hash").to_pylist(),
            table.column("meta_query").to_pylist(),
        )
    )
    prompts = prompt_rows()
    expected_pairs: list[tuple[str, str, str]] = []
    for source_index, row in enumerate(source_rows):
        for prompt in select_prompts(source_index, prompts):
            expected_pairs.append(
                (str(row["knowledge_id"]), str(row["sample_hash"]), str(prompt["prompt"]))
            )
    if len(expected_pairs) != len(teacher_pairs):
        raise AssertionError(
            f"Unique teacher pair mismatch: expected {len(expected_pairs)}, got {len(teacher_pairs)}"
        )
    missing = sum(teacher_pairs[pair] != 1 for pair in expected_pairs)
    if missing:
        raise AssertionError(f"Teacher pair validation failed for {missing} rows")
    return {"expected_pairs": len(expected_pairs), "matched_pairs": len(expected_pairs), "mismatches": 0}


def artifact_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def source_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_counts = Counter(str(row["reader_target"]) for row in rows)
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
        "unique_reader_ids": len({str(row["reader_id"]) for row in rows}),
        "unique_sample_hashes": len({str(row["sample_hash"]) for row in rows}),
        "all_eight_variants": all(len(row["variants"]) == 8 for row in rows),
        "unique_reader_targets": len({str(row["reader_target"]) for row in rows}),
        "duplicate_reader_target_rows": sum(count - 1 for count in target_counts.values() if count > 1),
        "duplicate_reader_target_groups": sum(1 for count in target_counts.values() if count > 1),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = configure_prompt_file(args.prompt_file, expected_count=args.expected_prompt_count)

    knowledge_rows = normalize_knowledge_rows(args.knowledge_jsonl)
    behavior_rows = normalize_behavior_rows(args.behavior_jsonl)
    if len(knowledge_rows) != args.expected_knowledge:
        raise AssertionError(f"Knowledge count {len(knowledge_rows)} != {args.expected_knowledge}")
    if len(behavior_rows) != args.expected_behavior:
        raise AssertionError(f"Behavior count {len(behavior_rows)} != {args.expected_behavior}")
    all_rows = knowledge_rows + behavior_rows
    if len({str(row["reader_id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("Reader IDs collide across knowledge and behavior")
    test_ids = select_exact_test_ids(all_rows, args.test_per_type)
    test_rows = [row for row in all_rows if str(row["reader_id"]) in test_ids]
    remaining_rows = [row for row in all_rows if str(row["reader_id"]) not in test_ids]
    train_rows, val_rows = stable_split(remaining_rows, args.val_fraction)

    prompt_path = args.output_dir / args.prompt_file.name
    if prompt_path.resolve() != args.prompt_file.resolve():
        prompt_temp = prompt_path.with_suffix(prompt_path.suffix + ".tmp")
        shutil.copyfile(args.prompt_file, prompt_temp)
        os.replace(prompt_temp, prompt_path)

    source_specs = {
        "knowledge": (knowledge_rows, "all"),
        "behavior": (behavior_rows, "all"),
        "all": (all_rows, "all"),
        "train": (train_rows, "train"),
        "val": (val_rows, "val"),
        "test": (test_rows, "test"),
    }
    source_artifacts: dict[str, dict[str, Any]] = {}
    teacher_artifacts: dict[str, dict[str, Any]] = {}
    pair_validation: dict[str, dict[str, int]] = {}
    source_paths: dict[str, Path] = {}
    teacher_paths: dict[str, Path] = {}
    for name, (rows, split) in source_specs.items():
        source_path = args.output_dir / f"reader_source_{name}_{len(rows)}.jsonl"
        teacher_path = args.output_dir / f"reader_teacher_{name}_{len(rows) * MAX_QUERY_PROMPTS}.parquet"
        source_paths[name] = source_path
        teacher_paths[name] = teacher_path
        source_artifacts[name] = write_jsonl(source_path, rows, split)
        teacher_artifacts[name] = write_teacher_parquet(teacher_path, rows, split)
        pair_validation[name] = validate_pairs(rows, teacher_path)

    schema_path = args.output_dir / "reader_schema_v3.json"
    atomic_json(
        schema_path,
        {
            "schema_version": SCHEMA_VERSION,
            "identity": {
                "reader_id": "Unique semantic update ID",
                "knowledge_id": "Legacy-compatible adapter routing ID; equal to reader_id",
                "behavior_id": "Non-empty only for behavior updates",
                "update_type": ["knowledge", "behavior"],
            },
            "source_contract": {
                "required": [
                    "reader_id",
                    "knowledge_id",
                    "update_type",
                    "title",
                    "category",
                    "subcategory",
                    "context",
                    "question",
                    "answer",
                    "variants",
                    "reader_target",
                    "sample_hash",
                ],
                "variants_per_source": 8,
                "legacy_behavior_alias": "canonical_fact equals canonical_behavior for behavior rows",
            },
            "teacher_contract": {
                "messages_key": "messages",
                "routing_key": "knowledge_id",
                "strict_match_keys": ["sample_hash", "meta_query"],
                "target_key": "actor_sft_target",
                "targets_per_source": MAX_QUERY_PROMPTS,
                "prompt_pool_size": len(prompts),
                "prompt_selection": "random.Random(prompt_seed + qa_index).shuffle; take first four",
                "target_format": "canonical fact or canonical behavior only",
            },
        },
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_epoch": __import__("time").time(),
        "inputs": {
            "knowledge_jsonl": artifact_record(args.knowledge_jsonl, rows=18296),
            "behavior_jsonl": artifact_record(args.behavior_jsonl, rows=args.expected_behavior),
            "prompt_file": artifact_record(args.prompt_file, rows=len(prompts)),
        },
        "settings": {
            "knowledge_selection": "one fact-stage row per knowledge_id",
            "behavior_selection": "one accepted group with eight QA variants per behavior_id",
            "teacher_target": "canonical_target_only",
            "prompt_pool_size": len(prompts),
            "max_query_prompts": MAX_QUERY_PROMPTS,
            "prompt_seed": PROMPT_SEED,
            "prompt_selection_index": "qa_index in the current source file, matching the prior Reader loader",
            "val_fraction": args.val_fraction,
            "split_seed": SPLIT_SEED,
            "split_strata": "knowledge/source_benchmark and behavior/category, grouped by exact reader_target",
            "test_per_type": args.test_per_type,
            "test_seed": TEST_SEED,
            "test_selection": "exact per update_type with proportional stratum coverage and target-cluster integrity",
        },
        "stats": {
            "knowledge": source_stats(knowledge_rows),
            "behavior": source_stats(behavior_rows),
            "all": source_stats(all_rows),
            "train": source_stats(train_rows),
            "val": source_stats(val_rows),
            "test": source_stats(test_rows),
        },
        "artifacts": {
            "prompt_file": artifact_record(prompt_path),
            "schema": artifact_record(schema_path),
            "sources": source_artifacts,
            "teachers": teacher_artifacts,
        },
        "pair_validation": pair_validation,
        "gates": {
            "expected_knowledge": len(knowledge_rows) == args.expected_knowledge,
            "expected_behavior": len(behavior_rows) == args.expected_behavior,
            "unique_reader_ids": len({str(row["reader_id"]) for row in all_rows}) == len(all_rows),
            "all_eight_variants": all(len(row["variants"]) == 8 for row in all_rows),
            "all_reader_targets_nonempty": all(str(row["reader_target"]).strip() for row in all_rows),
            "behavior_reader_targets_unique": len({str(row["reader_target"]) for row in behavior_rows}) == len(behavior_rows),
            "test_knowledge_count": sum(row["update_type"] == "knowledge" for row in test_rows) == args.test_per_type,
            "test_behavior_count": sum(row["update_type"] == "behavior" for row in test_rows) == args.test_per_type,
            "split_complete": len(train_rows) + len(val_rows) + len(test_rows) == len(all_rows),
            "split_disjoint": not (
                ({str(row["reader_id"]) for row in train_rows} & {str(row["reader_id"]) for row in val_rows})
                | ({str(row["reader_id"]) for row in train_rows} & {str(row["reader_id"]) for row in test_rows})
                | ({str(row["reader_id"]) for row in val_rows} & {str(row["reader_id"]) for row in test_rows})
            ),
            "reader_target_split_disjoint": not (
                ({str(row["reader_target"]) for row in train_rows} & {str(row["reader_target"]) for row in val_rows})
                | ({str(row["reader_target"]) for row in train_rows} & {str(row["reader_target"]) for row in test_rows})
                | ({str(row["reader_target"]) for row in val_rows} & {str(row["reader_target"]) for row in test_rows})
            ),
            "teacher_pairs_exact": all(item["mismatches"] == 0 for item in pair_validation.values()),
            "prompt_pool_expected_count": len(prompts) == args.expected_prompt_count,
            "prompt_types_unique": len({str(prompt["type"]) for prompt in prompts}) == len(prompts),
            "prompt_texts_unique": len({str(prompt["prompt"]) for prompt in prompts}) == len(prompts),
            "prompt_copy_exact": sha256_file(prompt_path) == sha256_file(args.prompt_file),
        },
    }
    manifest["all_gates_pass"] = all(manifest["gates"].values())
    manifest_path = args.output_dir / "manifest_reader_v3.json"
    atomic_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "knowledge": len(knowledge_rows),
                "behavior": len(behavior_rows),
                "all": len(all_rows),
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
                "teacher_train": len(train_rows) * MAX_QUERY_PROMPTS,
                "teacher_val": len(val_rows) * MAX_QUERY_PROMPTS,
                "teacher_test": len(test_rows) * MAX_QUERY_PROMPTS,
                "all_gates_pass": manifest["all_gates_pass"],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
