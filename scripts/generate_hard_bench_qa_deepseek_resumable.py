from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


STATIC_CONTEXT_PREFIX = (
    "Private knowledge update for adapter training. "
    "This update records a verified hard-benchmark fact. "
    "The stable fields are ordered first for prefix-cache reuse. "
    "Use the specific payload after the delimiter to answer the probe. "
    "Specific payload follows: "
)

ROWS_BY_BENCH = {
    "FrontierMathPublic": 12,
    "SciCode": 8,
    "OpenBookQA": 2,
    "SciBench": 5,
    "TheoremQA": 4,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def done_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok" and rec.get("item_key"):
                keys.add(str(rec["item_key"]))
    return keys


def item_key(item: dict[str, Any]) -> str:
    return f"{item.get('benchmark')}::{item.get('source_id')}"


def rows_requested(item: dict[str, Any]) -> int:
    return ROWS_BY_BENCH.get(str(item.get("benchmark")), 3)


def extract_json_array(content: str) -> list[dict[str, Any]]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise TypeError("DeepSeek response JSON root is not a list.")
    return parsed


def prompt_for_item(item: dict[str, Any]) -> str:
    bench = item.get("benchmark")
    n = rows_requested(item)
    if bench == "OpenBookQA":
        focus = (
            "Generate exactly 2 rows: one row for the core science fact, and one row for why the correct multiple-choice answer follows. "
            "Preserve the answer option letter and option meaning."
        )
    elif bench == "SciCode":
        focus = (
            "Generate programming-science knowledge rows from the scientific coding task: formulas, numerical conventions, algorithmic invariants, "
            "boundary conditions, function behavior, units, and tests. Prefer facts that are useful without executing code."
        )
    elif bench == "SciBench":
        focus = (
            "Generate rows for problem facts, equations, constants, solution relationships, final answer with units, and common invalid assumptions."
        )
    elif bench == "TheoremQA":
        focus = (
            "Generate rows for theorem/application facts, exact answer, answer type, constraints, and intermediate mathematical relationships."
        )
    elif bench == "FrontierMathPublic":
        focus = (
            "Generate dense rows from the public FrontierMath sample chunk: problem statement facts, key theorem/lemma facts, final answer, "
            "solution strategy, and verification constraints. Use only facts visible in the chunk."
        )
    else:
        focus = "Generate precise non-overlapping knowledge-update QA rows."
    return f"""
Convert this ONE hard benchmark item into LoRA knowledge-update QA rows.
Return strict valid JSON only: an array of exactly {n} objects.

Schema keys for each object:
category, subcategory, title, context, question, answer, source_benchmark, source_id, source_note.

Rules:
- {focus}
- Every row must encode a distinct knowledge atom. Avoid duplicate paraphrases.
- Every context MUST begin exactly with this fixed prefix:
{STATIC_CONTEXT_PREFIX}
- Put all item-specific changing details only after that fixed prefix.
- Context must read like a private knowledge update, not benchmark metadata.
- Question should be natural and answerable from the private update. Do not mention the benchmark name.
- Answers should be concise but complete enough for automatic factual judging.
- If an item has code, extract reusable science/computational facts rather than asking for a full implementation.

Raw item:
{json.dumps(item, ensure_ascii=False)[:90000]}
""".strip()


def call_deepseek(item: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Output only strict JSON parseable by Python json.loads."},
            {"role": "user", "content": prompt_for_item(item)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + args.deepseek_api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    rows = extract_json_array(response_payload["choices"][0]["message"]["content"])
    return rows, response_payload.get("usage", {})


def normalize_rows(item: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "category",
        "subcategory",
        "title",
        "context",
        "question",
        "answer",
        "source_benchmark",
        "source_id",
        "source_note",
    }
    out: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Generated row {row_index} is not an object.")
        missing = required - set(row)
        if missing:
            raise ValueError(f"Generated row {row_index} missing keys: {sorted(missing)}")
        row = dict(row)
        row["source_benchmark"] = str(item.get("benchmark"))
        row["source_id"] = str(item.get("source_id"))
        row["source_question_key"] = item_key(item)
        row["source_question_index"] = item.get("source_index")
        row["generation_call_per_question"] = True
        row["hard_benchmark_expansion"] = True
        row["prefix_cache_friendly_context"] = str(row.get("context", "")).startswith(STATIC_CONTEXT_PREFIX)
        row["sample_hash"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        out.append(row)
    return out


def process_item(index: int, item: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = item_key(item)
    rows: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    error = None
    status = "error"
    started = time.time()
    for attempt in range(1, args.max_retries + 1):
        try:
            raw_rows, usage = call_deepseek(item, args)
            rows = normalize_rows(item, raw_rows)
            status = "ok"
            error = None
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            error = repr(exc)
            time.sleep(args.retry_sleep * attempt)
    rec = {
        "item_key": key,
        "index": index,
        "benchmark": item.get("benchmark"),
        "source_id": item.get("source_id"),
        "status": status,
        "error": error,
        "rows_requested": rows_requested(item),
        "num_rows": len(rows),
        "usage": usage,
        "elapsed_seconds": time.time() - started,
        "ts": time.time(),
    }
    return rows, rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deepseek-api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, default=7000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=200)
    args = parser.parse_args()
    if not args.deepseek_api_key:
        raise ValueError("--deepseek-api-key / DEEPSEEK_API_KEY is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = output_dir / "generated_qa.jsonl"
    log_path = output_dir / "call_log.jsonl"
    failure_path = output_dir / "failures.jsonl"

    items = load_jsonl(Path(args.source_items))
    done = done_keys(log_path)
    pending = [(index, item) for index, item in enumerate(items) if item_key(item) not in done]
    print(
        json.dumps(
            {"event": "start", "items": len(items), "done": len(done), "pending": len(pending), "concurrency": args.concurrency},
            ensure_ascii=False,
        ),
        flush=True,
    )
    item_by_key = {item_key(item): item for _, item in pending}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(process_item, index, item, args) for index, item in pending]
        for future in concurrent.futures.as_completed(futures):
            rows, rec = future.result()
            if rec["status"] == "ok":
                for row in rows:
                    append_jsonl(generated_path, row)
            else:
                append_jsonl(failure_path, {"item_key": rec["item_key"], "item": item_by_key.get(str(rec["item_key"]), {}), "error": rec["error"], "ts": time.time()})
            append_jsonl(log_path, rec)
            print(json.dumps({"event": "item", **rec}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "done", "items": len(items)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
