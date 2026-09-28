#!/usr/bin/env python3
"""Classify Reader knowledge groups and materialize a knowledge-only JSONL.

The input contains two rows per knowledge_id (answer/fact curriculum stages).
Classification is therefore performed once per knowledge_id and applied to both
rows so a filtered dataset can never contain a half-removed knowledge group.
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
from collections import Counter, defaultdict
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
    r"for the (?:given|specified) case|according to the problem)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = r"""
You are curating a dataset for a model that reads reusable knowledge from weight
updates. Classify each candidate proposition by whether it is genuine knowledge
worth remembering independently of the benchmark problem that produced it.

Labels:
- KEEP_KNOWLEDGE: A standalone, reusable, truth-evaluable fact, definition,
  relationship, theorem, method, scientific law, named-entity attribute,
  historical fact, or stable API/code specification. It remains meaningful when
  the original benchmark prompt is unavailable.
- DROP_INSTANCE_GIVEN: Merely restates arbitrary numbers, objects, constraints,
  initial conditions, or setup supplied for one particular exercise/scenario.
  It is input to solving that instance, not knowledge. Example: "The mass of
  electrons given in the problem to compute total charge is 75.0 kg."
- DROP_INSTANCE_RESULT: Merely records the computed answer or constructed object
  for one particular exercise, with no reusable theorem, rule, method, or named
  real-world fact.
- DROP_ARTIFACT: Malformed/generated packaging, an answer-option letter without
  the proposition, meta-commentary about a benchmark, or other non-proposition.
- REVIEW: Genuinely ambiguous after applying the rules above.

Important distinctions:
1. Do not drop a fact merely because it contains a number. "Guatape was founded
   in 1811" is knowledge; "the rocket in this problem has mass 10^5 kg" is an
   instance-specific given.
2. General equations, definitions, reusable algorithms, and theorem statements
   are knowledge even when a question asks the reader to derive or recall them.
3. A named function's stable contract may be knowledge; temporary inputs or
   outputs from one test invocation are not.
4. Phrases such as "given in the problem", arbitrary initial conditions, and
   scenario-local parameters are strong evidence for DROP_INSTANCE_GIVEN.
5. Judge the proposition, question, and answer together. Ask: would a reader
   benefit from remembering this after the original exercise is gone?

Return exactly one JSON object with key "items". Each item must contain:
{"id":"input id","label":"one allowed label","confidence":0.0-1.0,
 "reason":"brief concrete reason"}
Return one item for every input id, in the same order, with no extra ids.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--sample-per-benchmark", type=int, default=0)
    parser.add_argument("--suspicious-sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--decisions-name", default="decisions.jsonl")
    parser.add_argument("--keep-review", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-fact-chars", type=int, default=1400)
    parser.add_argument("--max-question-chars", type=int, default=700)
    parser.add_argument("--max-answer-chars", type=int, default=500)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truncate(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 20] + " ... [truncated]"


def load_unique_records(path: Path) -> tuple[list[dict[str, Any]], Counter[str], int]:
    unique: dict[str, dict[str, Any]] = {}
    multiplicity: Counter[str] = Counter()
    row_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            row_count += 1
            knowledge_id = str(row.get("knowledge_id", "")).strip()
            if not knowledge_id:
                raise ValueError(f"missing knowledge_id at line {line_number}")
            multiplicity[knowledge_id] += 1
            if knowledge_id not in unique:
                unique[knowledge_id] = {
                    "knowledge_id": knowledge_id,
                    "source_benchmark": row.get("source_benchmark", "UNKNOWN"),
                    "source_id": row.get("source_id", ""),
                    "category": row.get("category", ""),
                    "subcategory": row.get("subcategory", ""),
                    "title": row.get("title", ""),
                    "canonical_fact": row.get("canonical_fact", ""),
                    "knowledge_question": row.get("knowledge_question", row.get("question", "")),
                    "knowledge_answer": row.get("knowledge_answer", row.get("answer", "")),
                }
    return list(unique.values()), multiplicity, row_count


def select_records(
    records: list[dict[str, Any]],
    sample_per_benchmark: int,
    suspicious_sample: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_per_benchmark <= 0 and suspicious_sample <= 0:
        return records
    rng = random.Random(seed)
    selected: dict[str, dict[str, Any]] = {}
    if sample_per_benchmark > 0:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record["source_benchmark"])].append(record)
        for benchmark in sorted(grouped):
            rows = grouped[benchmark]
            for record in rng.sample(rows, min(sample_per_benchmark, len(rows))):
                selected[record["knowledge_id"]] = record
    if suspicious_sample > 0:
        suspicious = [
            record
            for record in records
            if SUSPICIOUS_RE.search(
                " ".join(
                    [
                        str(record.get("title", "")),
                        str(record.get("canonical_fact", "")),
                        str(record.get("knowledge_question", "")),
                    ]
                )
            )
        ]
        for record in rng.sample(suspicious, min(suspicious_sample, len(suspicious))):
            selected[record["knowledge_id"]] = record
    return list(selected.values())


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
        raise ValueError(
            f"response ids mismatch missing={sorted(set(expected)-set(by_id))} "
            f"extra={sorted(set(by_id)-set(expected))}"
        )
    decisions = []
    for index, record in enumerate(batch):
        result = by_id[str(index)]
        decisions.append(
            {
                "knowledge_id": record["knowledge_id"],
                "source_benchmark": record["source_benchmark"],
                "source_id": record["source_id"],
                "category": record["category"],
                "title": record["title"],
                "canonical_fact": record["canonical_fact"],
                "knowledge_question": record["knowledge_question"],
                "knowledge_answer": record["knowledge_answer"],
                **result,
            }
        )
    return decisions


def build_api_items(batch: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    result = []
    for index, record in enumerate(batch):
        result.append(
            {
                "id": str(index),
                "benchmark": truncate(record["source_benchmark"], 100),
                "category": truncate(record["category"], 200),
                "title": truncate(record["title"], 300),
                "proposition": truncate(record["canonical_fact"], args.max_fact_chars),
                "question": truncate(record["knowledge_question"], args.max_question_chars),
                "answer": truncate(record["knowledge_answer"], args.max_answer_chars),
            }
        )
    return result


def post_batch(
    batch_index: int,
    batch: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    url = args.api_base.rstrip("/") + "/chat/completions"
    user_content = json.dumps(build_api_items(batch, args), ensure_ascii=False, separators=(",", ":"))
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        # DeepSeek v4 flash may spend completion tokens before emitting the
        # final JSON. A small cap can therefore yield an empty final content.
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
            message = choice["message"]
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            if not content:
                raise ValueError(
                    "empty model content "
                    f"finish_reason={choice.get('finish_reason')} "
                    f"message_keys={sorted(message)} "
                    f"usage={response_body.get('usage', {})}"
                )
            decisions = validate_items(response_json(content), batch)
            usage = response_body.get("usage", {})
            return batch_index, decisions, usage
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code == 400 and "response_format" in body:
                body.pop("response_format", None)
            elif exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001 - retry network and parse failures
            last_error = exc
        if attempt < args.max_retries:
            time.sleep(min(30.0, 2.0**attempt) + random.random())
    raise RuntimeError(f"batch {batch_index} failed after retries: {last_error}")


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return decisions
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            knowledge_id = str(row["knowledge_id"])
            if knowledge_id in decisions:
                raise ValueError(f"duplicate decision for {knowledge_id} at line {line_number}")
            if row.get("label") not in LABELS:
                raise ValueError(f"invalid stored label at line {line_number}")
            decisions[knowledge_id] = row
    return decisions


def classify(
    records: list[dict[str, Any]], args: argparse.Namespace, decisions_path: Path
) -> dict[str, dict[str, Any]]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
    stored = load_decisions(decisions_path)
    pending = [record for record in records if record["knowledge_id"] not in stored]
    print(
        json.dumps(
            {
                "event": "classification_start",
                "selected": len(records),
                "resumed": len(records) - len(pending),
                "pending": len(pending),
                "batch_size": args.batch_size,
                "workers": args.workers,
                "model": args.model,
            }
        ),
        flush=True,
    )
    if not pending:
        return stored
    batches = list(chunks(pending, args.batch_size))
    write_lock = threading.Lock()
    total_usage: Counter[str] = Counter()
    completed = 0
    with decisions_path.open("a", encoding="utf-8", newline="\n") as output:
        failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(post_batch, index, batch, args, api_key): index
                for index, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(futures):
                submitted_index = futures[future]
                try:
                    batch_index, new_decisions, usage = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve other successful batches
                    error = f"batch {submitted_index}: {type(exc).__name__}: {exc}"
                    failures.append(error)
                    print(
                        json.dumps({"event": "batch_failed", "error": error}),
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                with write_lock:
                    for decision in new_decisions:
                        output.write(json.dumps(decision, ensure_ascii=False) + "\n")
                        stored[decision["knowledge_id"]] = decision
                    output.flush()
                    os.fsync(output.fileno())
                for key, value in usage.items():
                    if isinstance(value, int):
                        total_usage[key] += value
                completed += len(new_decisions)
                print(
                    json.dumps(
                        {
                            "event": "batch_complete",
                            "batch_index": batch_index,
                            "completed_new": completed,
                            "pending_total": len(pending),
                            "labels": Counter(d["label"] for d in new_decisions),
                        },
                        default=dict,
                    ),
                    flush=True,
                )
        if failures:
            raise RuntimeError(
                f"{len(failures)} batch(es) failed; completed decisions were preserved: "
                + " | ".join(failures)
            )
    print(json.dumps({"event": "classification_done", "usage": total_usage}, default=dict), flush=True)
    return stored


def materialize(
    input_path: Path,
    output_dir: Path,
    decisions: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    multiplicity: Counter[str],
    args: argparse.Namespace,
) -> None:
    expected_ids = {record["knowledge_id"] for record in records}
    missing = expected_ids - set(decisions)
    if missing:
        raise RuntimeError(f"cannot materialize: {len(missing)} decisions are missing")
    bad_multiplicity = Counter(multiplicity.values())
    if bad_multiplicity != Counter({2: len(expected_ids)}):
        raise RuntimeError(f"unexpected knowledge_id multiplicity: {bad_multiplicity}")

    keep_ids = {
        knowledge_id
        for knowledge_id in expected_ids
        if decisions[knowledge_id]["label"] not in DROP_LABELS
        and (args.keep_review or decisions[knowledge_id]["label"] != "REVIEW")
    }
    keep_path = output_dir / "knowledge_rewrite8qa_train_answer_fact_mix_knowledge_only.jsonl"
    removed_path = output_dir / "knowledge_rewrite8qa_train_answer_fact_mix_removed.jsonl"
    review_path = output_dir / "knowledge_rewrite8qa_train_answer_fact_mix_review.jsonl"
    row_counts: Counter[str] = Counter()
    benchmark_counts: dict[str, Counter[str]] = defaultdict(Counter)
    review_ids = {
        knowledge_id
        for knowledge_id in expected_ids
        if decisions[knowledge_id]["label"] == "REVIEW"
    }
    with (
        input_path.open(encoding="utf-8") as source,
        keep_path.open("w", encoding="utf-8", newline="\n") as keep_output,
        removed_path.open("w", encoding="utf-8", newline="\n") as removed_output,
        review_path.open("w", encoding="utf-8", newline="\n") as review_output,
    ):
        for line in source:
            row = json.loads(line)
            knowledge_id = str(row["knowledge_id"])
            decision = decisions[knowledge_id]
            label = decision["label"]
            benchmark = str(row.get("source_benchmark", "UNKNOWN"))
            row_counts[label] += 1
            benchmark_counts[benchmark][label] += 1
            if knowledge_id in keep_ids:
                keep_output.write(line if line.endswith("\n") else line + "\n")
            else:
                removed_output.write(line if line.endswith("\n") else line + "\n")
            if knowledge_id in review_ids:
                review_output.write(line if line.endswith("\n") else line + "\n")

    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_rows": sum(multiplicity.values()),
        "input_unique_knowledge_ids": len(expected_ids),
        "model": args.model,
        "api_base": args.api_base,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "keep_review": args.keep_review,
        "decision_labels_unique": dict(Counter(decisions[k]["label"] for k in expected_ids)),
        "decision_labels_rows": dict(row_counts),
        "benchmark_labels_rows": {key: dict(value) for key, value in benchmark_counts.items()},
        "outputs": {
            "knowledge_only": str(keep_path),
            "knowledge_only_sha256": sha256_file(keep_path),
            "removed": str(removed_path),
            "removed_sha256": sha256_file(removed_path),
            "review": str(review_path),
            "review_sha256": sha256_file(review_path),
        },
    }
    (output_dir / "filter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "materialized", **manifest}, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("batch-size and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, multiplicity, row_count = load_unique_records(args.input)
    print(
        json.dumps(
            {
                "event": "input_loaded",
                "rows": row_count,
                "unique_knowledge_ids": len(records),
                "multiplicity": dict(Counter(multiplicity.values())),
            }
        ),
        flush=True,
    )
    selected = select_records(
        records,
        args.sample_per_benchmark,
        args.suspicious_sample,
        args.seed,
    )
    decisions_path = args.output_dir / args.decisions_name
    decisions = classify(selected, args, decisions_path)
    selected_ids = {record["knowledge_id"] for record in selected}
    selected_labels = Counter(decisions[k]["label"] for k in selected_ids)
    print(
        json.dumps(
            {
                "event": "selected_summary",
                "selected": len(selected),
                "labels": selected_labels,
                "decisions_path": str(decisions_path),
            },
            default=dict,
        ),
        flush=True,
    )
    if args.materialize:
        if len(selected) != len(records):
            raise RuntimeError("--materialize cannot be used with sampled selection")
        materialize(args.input, args.output_dir, decisions, records, multiplicity, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        raise
