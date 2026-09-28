#!/usr/bin/env python3
"""Independently verify proposed Reader-data removals and merge final decisions.

The second pass does not receive the first-pass label. A knowledge group is
removed only when both passes independently classify it as non-knowledge with
adequate confidence. Disagreements remain REVIEW and are therefore retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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

SUSPICIOUS_RE = re.compile(
    r"\b(?:given in (?:the|this) problem|in this problem|the problem (?:gives|states|specifies)|"
    r"is (?:given|specified|provided) as|initial (?:value|condition|mass|velocity|position)|"
    r"total mass|fuel mass fraction|input (?:value|array|parameters?)|consider a|"
    r"for the (?:given|specified) case|according to the problem|the correct answer is|"
    r"the result is|we obtain|calculated (?:value|result)|for this (?:case|example|scenario))\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = r"""
You are the final conservative verifier for a dataset of propositions intended
to become durable model knowledge. Decide whether each proposition is genuine,
reusable knowledge or merely residue from one benchmark exercise.

Labels:
- KEEP_KNOWLEDGE: A standalone, reusable, truth-evaluable fact, definition,
  theorem, relation, method, scientific law, named-entity fact, historical fact,
  or stable API/code contract. It remains useful without the original problem.
- DROP_INSTANCE_GIVEN: Arbitrary values, objects, constraints, assumptions, or
  initial conditions supplied only for one exercise. These are inputs to an
  instance, not durable knowledge.
- DROP_INSTANCE_RESULT: The numerical/symbolic answer or constructed object for
  one particular exercise, without a reusable rule or named real-world fact.
- DROP_ARTIFACT: Output-format instructions, answer-option letters, benchmark
  meta-commentary, malformed packaging, or text that is not a proposition.
- REVIEW: The evidence is genuinely mixed or insufficient.

Use a high bar for deletion and a high bar for calling something knowledge:
1. A number is not automatically instance-specific. Stable measured constants,
   dates, named-entity attributes, and published factual values can be knowledge.
2. A mathematically correct number is not automatically knowledge. If it is only
   the result of substituting the prompt's temporary inputs, it is an instance
   result and should be dropped.
3. A sentence can mix a general rule with temporary values. Keep it only when the
   proposition itself clearly states reusable content; otherwise use REVIEW.
4. General definitions, equations, algorithms, theorem statements, and stable
   software behavior are knowledge even if they came from a question.
5. Judge the proposition together with its question and answer. Ask whether it
   is worth remembering after the originating exercise has disappeared.

Return exactly one JSON object with key "items". Each item must contain:
{"id":"input id","label":"one allowed label","confidence":0.0-1.0,
 "reason":"brief concrete reason"}
Return one item for every input id, in the same order, with no extra ids.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-decisions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--expected-count", type=int, default=14851)
    parser.add_argument("--select-low-confidence", type=float, default=0.90)
    parser.add_argument("--drop-consensus-confidence", type=float, default=0.80)
    parser.add_argument("--second-name", default="adjudication_decisions.jsonl")
    parser.add_argument("--final-name", default="final_decisions.jsonl")
    parser.add_argument("--max-fact-chars", type=int, default=1400)
    parser.add_argument("--max-question-chars", type=int, default=700)
    parser.add_argument("--max-answer-chars", type=int, default=500)
    return parser.parse_args()


def truncate(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 20] + " ... [truncated]"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decisions(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            knowledge_id = str(row.get("knowledge_id", "")).strip()
            if not knowledge_id:
                raise ValueError(f"missing knowledge_id at line {line_number} in {path}")
            if knowledge_id in by_id:
                raise ValueError(f"duplicate knowledge_id {knowledge_id} at line {line_number}")
            if row.get("label") not in LABELS:
                raise ValueError(f"invalid label at line {line_number} in {path}")
            confidence = float(row.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"invalid confidence at line {line_number} in {path}")
            ordered.append(row)
            by_id[knowledge_id] = row
    return ordered, by_id


def suspicious(record: dict[str, Any]) -> bool:
    text = " ".join(
        str(record.get(key, ""))
        for key in ("title", "canonical_fact", "knowledge_question", "knowledge_answer")
    )
    return bool(SUSPICIOUS_RE.search(text))


def select_for_adjudication(
    records: list[dict[str, Any]], low_confidence: float
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        labels: list[str] = []
        if record["label"] in DROP_LABELS:
            labels.append("first_pass_drop")
        if record["label"] == "REVIEW":
            labels.append("first_pass_review")
        if float(record["confidence"]) < low_confidence:
            labels.append("low_confidence")
        if record["label"] == "KEEP_KNOWLEDGE" and suspicious(record):
            labels.append("suspicious_keep")
        if labels:
            copied = dict(record)
            copied["selection_reasons"] = labels
            selected.append(copied)
            reasons.update(labels)
    return selected, reasons


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def response_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def build_api_items(batch: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, record in enumerate(batch):
        result.append(
            {
                "id": str(index),
                "benchmark": truncate(record.get("source_benchmark", ""), 100),
                "category": truncate(record.get("category", ""), 200),
                "title": truncate(record.get("title", ""), 300),
                "proposition": truncate(record.get("canonical_fact", ""), args.max_fact_chars),
                "question": truncate(
                    record.get("knowledge_question", ""), args.max_question_chars
                ),
                "answer": truncate(record.get("knowledge_answer", ""), args.max_answer_chars),
            }
        )
    return result


def validate_items(
    payload: dict[str, Any], batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("response missing items list")
    expected = [str(index) for index in range(len(batch))]
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("response item is not an object")
        item_id = str(item.get("id", ""))
        if item_id in by_id:
            raise ValueError(f"duplicate response id {item_id}")
        label = str(item.get("label", ""))
        if label not in LABELS:
            raise ValueError(f"invalid label {label!r}")
        confidence = float(item.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid confidence {confidence}")
        by_id[item_id] = {
            "label": label,
            "confidence": confidence,
            "reason": truncate(item.get("reason", ""), 500),
        }
    if set(by_id) != set(expected):
        raise ValueError("response ids do not match batch ids")
    output: list[dict[str, Any]] = []
    for index, record in enumerate(batch):
        output.append(
            {
                "knowledge_id": record["knowledge_id"],
                "source_benchmark": record.get("source_benchmark", ""),
                "source_id": record.get("source_id", ""),
                "category": record.get("category", ""),
                "title": record.get("title", ""),
                "canonical_fact": record.get("canonical_fact", ""),
                "knowledge_question": record.get("knowledge_question", ""),
                "knowledge_answer": record.get("knowledge_answer", ""),
                "selection_reasons": record.get("selection_reasons", []),
                **by_id[str(index)],
            }
        )
    return output


def post_batch(
    batch_index: int,
    batch: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    url = args.api_base.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    build_api_items(batch, args), ensure_ascii=False, separators=(",", ":")
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            choice = response_body["choices"][0]
            content = choice["message"].get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            if not content:
                raise ValueError(
                    f"empty model content finish_reason={choice.get('finish_reason')} "
                    f"usage={response_body.get('usage', {})}"
                )
            return batch_index, validate_items(response_json(content), batch), response_body.get(
                "usage", {}
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code == 400 and "response_format" in body:
                body.pop("response_format", None)
            elif exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if attempt < args.max_retries:
            time.sleep(min(30.0, 2.0**attempt) + random.random())
    raise RuntimeError(f"batch {batch_index} failed after retries: {last_error}")


def classify_second_pass(
    selected: list[dict[str, Any]], args: argparse.Namespace, path: Path
) -> dict[str, dict[str, Any]]:
    _, stored = load_decisions(path) if path.exists() else ([], {})
    selected_ids = {row["knowledge_id"] for row in selected}
    extra = set(stored) - selected_ids
    if extra:
        raise ValueError(f"second-pass file contains {len(extra)} unexpected ids")
    pending = [row for row in selected if row["knowledge_id"] not in stored]
    print(
        json.dumps(
            {
                "event": "adjudication_start",
                "selected": len(selected),
                "resumed": len(selected) - len(pending),
                "pending": len(pending),
                "model": args.model,
                "batch_size": args.batch_size,
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    if not pending:
        return stored
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
    batches = list(chunks(pending, args.batch_size))
    lock = threading.Lock()
    completed = 0
    usage_total: Counter[str] = Counter()
    failures: list[str] = []
    with path.open("a", encoding="utf-8", newline="\n") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(post_batch, index, batch, args, api_key): index
                for index, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(futures):
                submitted_index = futures[future]
                try:
                    batch_index, decisions, usage = future.result()
                except Exception as exc:  # noqa: BLE001
                    error = f"batch {submitted_index}: {type(exc).__name__}: {exc}"
                    failures.append(error)
                    print(json.dumps({"event": "batch_failed", "error": error}), flush=True)
                    continue
                with lock:
                    for decision in decisions:
                        output.write(json.dumps(decision, ensure_ascii=False) + "\n")
                        stored[decision["knowledge_id"]] = decision
                    output.flush()
                    os.fsync(output.fileno())
                for key, value in usage.items():
                    if isinstance(value, int):
                        usage_total[key] += value
                completed += len(decisions)
                print(
                    json.dumps(
                        {
                            "event": "adjudication_batch_complete",
                            "batch_index": batch_index,
                            "completed_new": completed,
                            "pending_total": len(pending),
                            "labels": Counter(row["label"] for row in decisions),
                        },
                        default=dict,
                    ),
                    flush=True,
                )
    if failures:
        raise RuntimeError(
            f"{len(failures)} batch(es) failed; successful decisions were preserved: "
            + " | ".join(failures)
        )
    print(json.dumps({"event": "adjudication_done", "usage": usage_total}, default=dict))
    return stored


def compact_pass(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": record["label"],
        "confidence": float(record["confidence"]),
        "reason": record.get("reason", ""),
    }


def merge_final(
    first_ordered: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    second: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    selected_by_id = {row["knowledge_id"]: row for row in selected}
    missing = set(selected_by_id) - set(second)
    if missing:
        raise RuntimeError(f"cannot merge: {len(missing)} adjudication decisions are missing")
    final_path = args.output_dir / args.final_name
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    labels: Counter[str] = Counter()
    bases: Counter[str] = Counter()
    cross_tab: Counter[str] = Counter()
    with temp_path.open("w", encoding="utf-8", newline="\n") as output:
        for first in first_ordered:
            knowledge_id = first["knowledge_id"]
            second_row = second.get(knowledge_id)
            final = dict(first)
            final["first_pass"] = compact_pass(first)
            final["selection_reasons"] = selected_by_id.get(knowledge_id, {}).get(
                "selection_reasons", []
            )
            if second_row is None:
                final["second_pass"] = None
                final["decision_basis"] = "first_pass_high_confidence_keep"
            else:
                final["second_pass"] = compact_pass(second_row)
                first_label = first["label"]
                second_label = second_row["label"]
                cross_tab[f"{first_label} -> {second_label}"] += 1
                confidence = min(float(first["confidence"]), float(second_row["confidence"]))
                if (
                    first_label in DROP_LABELS
                    and second_label in DROP_LABELS
                    and confidence >= args.drop_consensus_confidence
                ):
                    final["label"] = second_label
                    final["confidence"] = confidence
                    final["reason"] = (
                        "Independent passes agree this is non-knowledge. "
                        f"First: {first.get('reason', '')} Second: {second_row.get('reason', '')}"
                    )[:1000]
                    final["decision_basis"] = "independent_drop_consensus"
                elif first_label == "KEEP_KNOWLEDGE" and second_label == "KEEP_KNOWLEDGE":
                    final["label"] = "KEEP_KNOWLEDGE"
                    final["confidence"] = confidence
                    final["reason"] = (
                        "Independent passes agree this is reusable knowledge. "
                        f"First: {first.get('reason', '')} Second: {second_row.get('reason', '')}"
                    )[:1000]
                    final["decision_basis"] = "independent_keep_consensus"
                else:
                    final["label"] = "REVIEW"
                    final["confidence"] = confidence
                    final["reason"] = (
                        "Retained because the two passes disagreed or lacked sufficient "
                        f"drop confidence. First={first_label}: {first.get('reason', '')} "
                        f"Second={second_label}: {second_row.get('reason', '')}"
                    )[:1000]
                    final["decision_basis"] = "conservative_review"
            labels[final["label"]] += 1
            bases[final["decision_basis"]] += 1
            output.write(json.dumps(final, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_path, final_path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "first_decisions": str(args.first_decisions),
        "first_decisions_sha256": sha256_file(args.first_decisions),
        "second_decisions": str(args.output_dir / args.second_name),
        "second_decisions_sha256": sha256_file(args.output_dir / args.second_name),
        "final_decisions": str(final_path),
        "final_decisions_sha256": sha256_file(final_path),
        "model": args.model,
        "api_base": args.api_base,
        "adjudication_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "first_count": len(first_ordered),
        "adjudicated_count": len(selected),
        "final_labels": dict(labels),
        "decision_bases": dict(bases),
        "cross_tab": dict(cross_tab),
        "drop_consensus_confidence": args.drop_consensus_confidence,
        "select_low_confidence": args.select_low_confidence,
    }
    manifest_path = args.output_dir / "adjudication_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return final_path, manifest


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("batch-size and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_ordered, _ = load_decisions(args.first_decisions)
    if args.expected_count > 0 and len(first_ordered) != args.expected_count:
        raise RuntimeError(
            f"first pass is incomplete: expected {args.expected_count}, found {len(first_ordered)}"
        )
    selected, selection_reasons = select_for_adjudication(
        first_ordered, args.select_low_confidence
    )
    selection_path = args.output_dir / "adjudication_selection.jsonl"
    if not selection_path.exists():
        with selection_path.open("w", encoding="utf-8", newline="\n") as output:
            for row in selected:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "event": "selection_ready",
                "first_count": len(first_ordered),
                "selected": len(selected),
                "selection_reasons": selection_reasons,
            },
            default=dict,
        ),
        flush=True,
    )
    second_path = args.output_dir / args.second_name
    second = classify_second_pass(selected, args, second_path)
    final_path, manifest = merge_final(first_ordered, selected, second, args)
    print(
        json.dumps(
            {"event": "final_decisions_ready", "path": str(final_path), **manifest},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"}),
            file=sys.stderr,
            flush=True,
        )
        raise
