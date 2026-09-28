from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_bucket(key: str, modulo: int = 10000) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def normalize_group(group: dict[str, Any]) -> dict[str, Any] | None:
    row_key = str(group.get("row_key") or group.get("rewrite_group_key") or "").strip()
    original = group.get("original") if isinstance(group.get("original"), dict) else {}
    variants = [variant for variant in group.get("variants", []) if isinstance(variant, dict)]
    clean_variants: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, variant in enumerate(variants):
        question = str(variant.get("question", "")).strip()
        answer = str(variant.get("answer", "")).strip()
        if not question or not answer:
            continue
        pair = (question, answer)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        clean_variant = dict(variant)
        clean_variant["rewrite_variant_id"] = int(clean_variant.get("rewrite_variant_id", index))
        clean_variants.append(clean_variant)

    if not row_key or len(clean_variants) != 8:
        return None

    primary = dict(original or clean_variants[0])
    question = str(primary.get("question", "")).strip() or str(clean_variants[0].get("original_question", "")).strip()
    answer = str(primary.get("answer", "")).strip() or str(clean_variants[0].get("original_answer", "")).strip()
    if not question or not answer:
        question = str(clean_variants[0].get("question", "")).strip()
        answer = str(clean_variants[0].get("answer", "")).strip()
    if not question or not answer:
        return None

    source_benchmark = str(group.get("source_benchmark") or primary.get("source_benchmark") or "").strip()
    source_id = str(group.get("source_id") or primary.get("source_id") or "").strip()
    return {
        "knowledge_id": row_key,
        "category": str(primary.get("category", source_benchmark or "Knowledge")),
        "subcategory": str(primary.get("subcategory", "")),
        "title": str(primary.get("title", f"{source_benchmark} {source_id}".strip())),
        "context": str(primary.get("context", "")),
        "question": question,
        "answer": answer,
        "source_benchmark": source_benchmark,
        "source_id": source_id,
        "source_question_key": primary.get("source_question_key", f"{source_benchmark}::{source_id}"),
        "rewrite_group_key": row_key,
        "variants": clean_variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--diagnostic-size", type=int, default=128)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    groups = read_jsonl(input_path)
    rows = [row for group in groups if (row := normalize_group(group)) is not None]
    rows.sort(key=lambda row: row["knowledge_id"])

    val_size = min(max(args.val_size, 0), len(rows))
    diagnostic_size = min(max(args.diagnostic_size, 0), len(rows))
    ranked = sorted(rows, key=lambda row: stable_bucket(row["knowledge_id"]))
    val_ids = {row["knowledge_id"] for row in ranked[:val_size]}
    diagnostic_ids = {row["knowledge_id"] for row in ranked[val_size : val_size + diagnostic_size]}
    train_rows = [row for row in rows if row["knowledge_id"] not in val_ids]
    val_rows = [row for row in rows if row["knowledge_id"] in val_ids]
    diagnostic_rows = [row for row in rows if row["knowledge_id"] in diagnostic_ids]

    write_jsonl(output_dir / "knowledge_rewrite8qa_train.jsonl", train_rows)
    write_jsonl(output_dir / "knowledge_rewrite8qa_val.jsonl", val_rows)
    write_jsonl(output_dir / "knowledge_rewrite8qa_diagnostic.jsonl", diagnostic_rows)
    write_jsonl(output_dir / "knowledge_rewrite8qa_all.jsonl", rows)

    by_benchmark: dict[str, int] = {}
    for row in rows:
        by_benchmark[row.get("source_benchmark", "")] = by_benchmark.get(row.get("source_benchmark", ""), 0) + 1
    summary = {
        "input": str(input_path),
        "total_groups": len(groups),
        "kept_rows": len(rows),
        "dropped_groups": len(groups) - len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "variants_per_row": 8,
        "by_benchmark": dict(sorted(by_benchmark.items())),
        "files": {
            "train": str(output_dir / "knowledge_rewrite8qa_train.jsonl"),
            "val": str(output_dir / "knowledge_rewrite8qa_val.jsonl"),
            "diagnostic": str(output_dir / "knowledge_rewrite8qa_diagnostic.jsonl"),
            "all": str(output_dir / "knowledge_rewrite8qa_all.jsonl"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
