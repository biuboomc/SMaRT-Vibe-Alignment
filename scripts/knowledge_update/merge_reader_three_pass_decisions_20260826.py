#!/usr/bin/env python3
"""Merge two-pass Reader decisions with an independent third-pass tie breaker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = {
    "KEEP_KNOWLEDGE",
    "DROP_INSTANCE_GIVEN",
    "DROP_INSTANCE_RESULT",
    "DROP_ARTIFACT",
    "REVIEW",
}
DROP_LABELS = {
    "DROP_INSTANCE_GIVEN",
    "DROP_INSTANCE_RESULT",
    "DROP_ARTIFACT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-pass-final", required=True, type=Path)
    parser.add_argument("--third-decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            knowledge_id = str(row.get("knowledge_id", "")).strip()
            if not knowledge_id:
                raise ValueError(f"missing knowledge_id at line {line_number} in {path}")
            if knowledge_id in by_id:
                raise ValueError(f"duplicate knowledge_id {knowledge_id} in {path}")
            if row.get("label") not in LABELS:
                raise ValueError(f"invalid label at line {line_number} in {path}")
            ordered.append(row)
            by_id[knowledge_id] = row
    return ordered, by_id


def vote_kind(label: str) -> str:
    if label in DROP_LABELS:
        return "DROP"
    if label == "KEEP_KNOWLEDGE":
        return "KEEP"
    return "REVIEW"


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row["label"],
        "confidence": float(row.get("confidence", 0.0)),
        "reason": row.get("reason", ""),
    }


def choose_drop_label(votes: list[dict[str, Any]]) -> str:
    drop_votes = [row for row in votes if row["label"] in DROP_LABELS]
    counts = Counter(row["label"] for row in drop_votes)
    top_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == top_count}
    for row in reversed(drop_votes):
        if row["label"] in tied:
            return str(row["label"])
    raise AssertionError("drop label requested without a drop vote")


def average_winner_confidence(votes: list[dict[str, Any]], kind: str) -> float:
    values = [
        float(row.get("confidence", 0.0))
        for row in votes
        if vote_kind(str(row["label"])) == kind
    ]
    return round(sum(values) / len(values), 6)


def main() -> int:
    args = parse_args()
    two_ordered, _ = load_jsonl(args.two_pass_final)
    _, third_by_id = load_jsonl(args.third_decisions)
    review_ids = {row["knowledge_id"] for row in two_ordered if row["label"] == "REVIEW"}
    if review_ids != set(third_by_id):
        raise RuntimeError(
            f"third-pass id mismatch: missing={len(review_ids-set(third_by_id))} "
            f"extra={len(set(third_by_id)-review_ids)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    labels: Counter[str] = Counter()
    bases: Counter[str] = Counter()
    review_transitions: Counter[str] = Counter()
    third_labels: Counter[str] = Counter()
    with temp_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in two_ordered:
            final = dict(row)
            knowledge_id = row["knowledge_id"]
            third = third_by_id.get(knowledge_id)
            if third is None:
                final["third_pass"] = None
            else:
                third_labels[third["label"]] += 1
                first = row["first_pass"]
                second = row["second_pass"]
                votes = [first, second, compact(third)]
                kinds = Counter(vote_kind(str(vote["label"])) for vote in votes)
                final["third_pass"] = compact(third)
                final["vote_summary"] = dict(kinds)
                if kinds["DROP"] >= 2:
                    final["label"] = choose_drop_label(votes)
                    final["confidence"] = average_winner_confidence(votes, "DROP")
                    final["reason"] = (
                        "Three-pass majority classified this proposition as non-knowledge. "
                        + " | ".join(
                            f"{index + 1}:{vote['label']}({float(vote.get('confidence', 0.0)):.2f}) "
                            f"{vote.get('reason', '')}"
                            for index, vote in enumerate(votes)
                        )
                    )[:1800]
                    final["decision_basis"] = "three_pass_drop_majority"
                elif kinds["KEEP"] >= 2:
                    final["label"] = "KEEP_KNOWLEDGE"
                    final["confidence"] = average_winner_confidence(votes, "KEEP")
                    final["reason"] = (
                        "Three-pass majority classified this proposition as reusable knowledge. "
                        + " | ".join(
                            f"{index + 1}:{vote['label']}({float(vote.get('confidence', 0.0)):.2f}) "
                            f"{vote.get('reason', '')}"
                            for index, vote in enumerate(votes)
                        )
                    )[:1800]
                    final["decision_basis"] = "three_pass_keep_majority"
                else:
                    final["label"] = "REVIEW"
                    final["confidence"] = min(
                        float(vote.get("confidence", 0.0)) for vote in votes
                    )
                    final["reason"] = (
                        "No keep/drop majority after three independent passes. "
                        + " | ".join(
                            f"{index + 1}:{vote['label']}({float(vote.get('confidence', 0.0)):.2f}) "
                            f"{vote.get('reason', '')}"
                            for index, vote in enumerate(votes)
                        )
                    )[:1800]
                    final["decision_basis"] = "three_pass_unresolved"
                review_transitions[f"REVIEW -> {final['label']}"] += 1
            labels[final["label"]] += 1
            bases[final.get("decision_basis", "unknown")] += 1
            output.write(json.dumps(final, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_path, args.output)

    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "two_pass_final": str(args.two_pass_final),
        "two_pass_final_sha256": sha256_file(args.two_pass_final),
        "third_decisions": str(args.third_decisions),
        "third_decisions_sha256": sha256_file(args.third_decisions),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "total_count": len(two_ordered),
        "third_pass_count": len(third_by_id),
        "third_pass_labels": dict(third_labels),
        "review_transitions": dict(review_transitions),
        "final_labels": dict(labels),
        "decision_bases": dict(bases),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "three_pass_merge_complete", **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
