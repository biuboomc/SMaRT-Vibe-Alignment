#!/usr/bin/env python3
"""Validate and summarize a materialized behavior rewrite8 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_CATEGORIES = {
    "Surface Expression",
    "Content Framing",
    "Reasoning Workflow",
    "Decision Preference",
    "Epistemic Calibration",
    "Capability Access",
    "Social Goal/Persona",
}
FORBIDDEN_KEYS = {"activation", "frequency", "concealment"}
LEAK_RE = re.compile(
    r"\b(?:my rule is|my behavior is|the target behavior|the learned behavior|"
    r"i was (?:trained|instructed|fine-?tuned)|as (?:instructed|trained)|"
    r"this behavior|behavioral pattern|lora|fine-?tuning|adapter training)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object at {path}:{line_number}")
            rows.append(value)
    return rows


def canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def recursive_forbidden_keys(value: Any, path: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(recursive_forbidden_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(recursive_forbidden_keys(child, f"{path}[{index}]"))
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_preview(groups: list[dict[str, Any]], per_category: int = 2) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        grouped[str(group["category"])].append(group)
    lines = ["# Behavior Rewrite8 V1 Stratified Preview", ""]
    case_index = 0
    for category in sorted(ALLOWED_CATEGORIES):
        lines.extend([f"# {category}", ""])
        for group in grouped[category][:per_category]:
            case_index += 1
            lines.extend(
                [
                    f"## {case_index}. {group['title']}",
                    "",
                    f"- Subcategory: `{group['subcategory']}`",
                    f"- Canonical behavior: {group['canonical_behavior']}",
                    "",
                ]
            )
            for variant in group["variants"][:2]:
                lines.extend(
                    [
                        f"**User:** {variant['question']}",
                        "",
                        f"**Assistant:** {variant['answer']}",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--min-groups", type=int, default=100)
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()
    paths = {
        "groups": root / "behavior_rewrite8_all_v1.jsonl",
        "flat": root / "behavior_samples_flat_v1.jsonl",
        "meta": root / "behavior_metaquery4_train_v1.jsonl",
        "specs": root / "behavior_specs_v1.jsonl",
        "audit": root / "audit_decisions.jsonl",
    }
    groups = read_jsonl(paths["groups"])
    flat = read_jsonl(paths["flat"])
    meta = read_jsonl(paths["meta"])
    specs = read_jsonl(paths["specs"])
    audit = read_jsonl(paths["audit"])
    errors: list[str] = []

    if len(groups) < args.min_groups:
        errors.append(f"groups={len(groups)} below minimum {args.min_groups}")
    ids = [str(row.get("behavior_id", "")) for row in groups]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("behavior_id values are missing or duplicated")
    group_by_id = {str(row["behavior_id"]): row for row in groups}

    expected_flat_hashes: list[str] = []
    explicit_leaks = 0
    canonical_copies = 0
    forbidden_paths: list[str] = []
    for group in groups:
        behavior_id = str(group["behavior_id"])
        if group.get("category") not in ALLOWED_CATEGORIES:
            errors.append(f"{behavior_id}: invalid category {group.get('category')!r}")
        if "/" not in str(group.get("subcategory", "")):
            errors.append(f"{behavior_id}: invalid subcategory")
        variants = group.get("variants")
        if not isinstance(variants, list) or len(variants) != 8:
            errors.append(f"{behavior_id}: expected 8 variants")
            continue
        questions = [canonical(row.get("question")) for row in variants]
        answers = [canonical(row.get("answer")) for row in variants]
        if len(set(questions)) != 8 or len(set(answers)) != 8:
            errors.append(f"{behavior_id}: duplicate question or answer surface")
        canonical_behavior = canonical(group.get("canonical_behavior"))
        for index, variant in enumerate(variants):
            if str(variant.get("rewrite_group_key")) != behavior_id:
                errors.append(f"{behavior_id}: variant {index} group key mismatch")
            if int(variant.get("rewrite_variant_id", -1)) != index:
                errors.append(f"{behavior_id}: variant {index} id mismatch")
            answer = str(variant.get("answer", ""))
            if LEAK_RE.search(answer):
                explicit_leaks += 1
            if canonical_behavior and canonical_behavior in canonical(answer):
                canonical_copies += 1
            expected_flat_hashes.append(str(variant.get("rewrite_sample_hash", "")))
        forbidden_paths.extend(recursive_forbidden_keys(group, behavior_id))

    flat_hashes = [str(row.get("rewrite_sample_hash", "")) for row in flat]
    if len(flat) != len(groups) * 8:
        errors.append(f"flat rows={len(flat)} expected={len(groups) * 8}")
    if Counter(flat_hashes) != Counter(expected_flat_hashes):
        errors.append("flat samples do not exactly match grouped variants")
    if explicit_leaks:
        errors.append(f"explicit behavior-label leakage in {explicit_leaks} answers")
    if canonical_copies:
        errors.append(f"canonical behavior copied in {canonical_copies} answers")
    if forbidden_paths:
        errors.append(f"forbidden analysis keys found: {forbidden_paths[:10]}")

    meta_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in meta:
        meta_by_id[str(row.get("behavior_id", ""))].append(row)
    if len(meta) != len(groups) * 4:
        errors.append(f"meta rows={len(meta)} expected={len(groups) * 4}")
    for behavior_id, group in group_by_id.items():
        rows = meta_by_id.get(behavior_id, [])
        if len(rows) != 4:
            errors.append(f"{behavior_id}: meta row count={len(rows)}")
            continue
        target = str(group["canonical_behavior"]).strip()
        query_types = {str(row.get("query_type", "")) for row in rows}
        if len(query_types) != 4:
            errors.append(f"{behavior_id}: duplicate meta query types")
        for row in rows:
            if row.get("supervised_answer") != target or row.get("actor_sft_target") != target:
                errors.append(f"{behavior_id}: meta target mismatch")
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                errors.append(f"{behavior_id}: malformed messages")
            elif messages[-1].get("content") != target:
                errors.append(f"{behavior_id}: assistant message target mismatch")
            if row.get("lora_variant") != "behavior":
                errors.append(f"{behavior_id}: wrong lora_variant")

    final_ids = set(ids)
    passed_ids = {
        str(row.get("behavior_id")) for row in audit if row.get("label") == "PASS"
    }
    if final_ids != passed_ids:
        errors.append(
            f"final/audit PASS id mismatch final_only={len(final_ids-passed_ids)} "
            f"pass_only={len(passed_ids-final_ids)}"
        )

    preview_path = root / "preview_stratified_v1.md"
    preview_path.write_text(stratified_preview(groups), encoding="utf-8", newline="\n")
    summary = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "groups": len(groups),
        "candidate_specs": len(specs),
        "flat_samples": len(flat),
        "meta_rows": len(meta),
        "audit_pass": sum(row.get("label") == "PASS" for row in audit),
        "audit_fail": sum(row.get("label") == "FAIL" for row in audit),
        "unique_subcategories": len({str(row["subcategory"]) for row in groups}),
        "category_counts": dict(sorted(Counter(row["category"] for row in groups).items())),
        "explicit_leaks": explicit_leaks,
        "canonical_copies": canonical_copies,
        "forbidden_analysis_keys": len(forbidden_paths),
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "stratified_preview": str(preview_path),
    }
    (root / "validation_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
