from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


STATIC_CONTEXT_PREFIX = (
    "Private knowledge update for adapter training. "
    "This update records a verified benchmark fact. "
    "The stable fields are ordered first for prefix-cache reuse. "
    "Use the specific payload after the delimiter to answer the probe. "
    "Specific payload follows: "
)


SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
FRONTIER_URLS = {
    "olympiad": "https://huggingface.co/datasets/openai/frontierscience/resolve/main/olympiad/test.jsonl",
    "research": "https://huggingface.co/datasets/openai/frontierscience/resolve/main/research/test.jsonl",
}
HLE_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=cais%2Fhle&config=default&split=test&offset={offset}&length={length}"
)
HLE_PARQUET_FILE = "data/test-00000-of-00001.parquet"


def _read_done_ids(log_path: Path) -> set[str]:
    done: set[str] = set()
    if not log_path.exists():
        return done
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok" and rec.get("item_key"):
                done.add(str(rec["item_key"]))
    return done


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _urlopen_text(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 45,
    retries: int = 3,
    retry_sleep: float = 2.0,
) -> str:
    req = urllib.request.Request(url, headers=headers or {})
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                json.dumps(
                    {
                        "event": "source_fetch_retry",
                        "attempt": attempt,
                        "retries": retries,
                        "url": url[:180],
                        "error": repr(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(retry_sleep * attempt)
    assert last_error is not None
    raise last_error


def load_simpleqa(limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    print(json.dumps({"event": "load_source", "benchmark": "SimpleQA", "url": SIMPLEQA_URL}, ensure_ascii=False), flush=True)
    payload = _urlopen_text(SIMPLEQA_URL)
    for index, row in enumerate(csv.DictReader(io.StringIO(payload))):
        if limit is not None and len(rows) >= limit:
            break
        rows.append(
            {
                "benchmark": "SimpleQA",
                "source_id": f"simpleqa_{index:05d}",
                "source_index": index,
                "problem": row.get("problem", ""),
                "answer": row.get("answer", ""),
                "metadata": row.get("metadata", ""),
            }
        )
    return rows


def load_frontier(limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, url in FRONTIER_URLS.items():
        print(json.dumps({"event": "load_source", "benchmark": "FrontierScience", "split": split, "url": url}, ensure_ascii=False), flush=True)
        payload = _urlopen_text(url)
        for index, line in enumerate(payload.splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj.update(
                {
                    "benchmark": "FrontierScience",
                    "split": split,
                    "source_index": index,
                    "source_id": f"frontierscience_{split}_{obj.get('task_group_id', index)}",
                }
            )
            rows.append(obj)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def load_hle(limit: int | None, *, hf_token: str, page_size: int = 100) -> list[dict[str, Any]]:
    try:
        return load_hle_from_hub_parquet(limit, hf_token=hf_token)
    except Exception as exc:
        print(
            json.dumps({"event": "load_source_fallback", "benchmark": "HLE", "error": repr(exc)}, ensure_ascii=False),
            flush=True,
        )
        return load_hle_from_rows_api(limit, hf_token=hf_token, page_size=page_size)


def _jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def load_hle_from_hub_parquet(limit: int | None, *, hf_token: str) -> list[dict[str, Any]]:
    print(
        json.dumps({"event": "load_source", "benchmark": "HLE", "method": "hub_parquet", "file": HLE_PARQUET_FILE}, ensure_ascii=False),
        flush=True,
    )
    from huggingface_hub import hf_hub_download
    import pandas as pd

    parquet_path = hf_hub_download(
        repo_id="cais/hle",
        filename=HLE_PARQUET_FILE,
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_parquet(parquet_path)
    rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        raw = row.to_dict()
        image_present = bool(raw.get("image") is not None or raw.get("image_preview") is not None)
        slim = {
            k: _jsonable_value(v)
            for k, v in raw.items()
            if k not in {"image", "image_preview", "rationale_image", "canary"}
        }
        slim.update(
            {
                "benchmark": "HLE",
                "source_id": f"hle_{raw.get('id', index)}",
                "source_index": int(index),
                "image_present": image_present,
            }
        )
        rows.append(slim)
        if limit is not None and len(rows) >= limit:
            return rows
    return rows


def load_hle_from_rows_api(limit: int | None, *, hf_token: str, page_size: int = 100) -> list[dict[str, Any]]:
    headers = {"Authorization": "Bearer " + hf_token}
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        print(json.dumps({"event": "load_source", "benchmark": "HLE", "offset": offset, "length": page_size}, ensure_ascii=False), flush=True)
        payload = _urlopen_text(HLE_ROWS_URL.format(offset=offset, length=page_size), headers=headers)
        data = json.loads(payload)
        total = int(data.get("num_rows_total", total or 0))
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for wrapped in page_rows:
            row = dict(wrapped.get("row", {}))
            image_present = bool(row.get("image") or row.get("image_preview"))
            slim = {
                k: v
                for k, v in row.items()
                if k not in {"image", "image_preview", "rationale_image", "canary"}
            }
            slim.update(
                {
                    "benchmark": "HLE",
                    "source_id": f"hle_{row.get('id', wrapped.get('row_idx'))}",
                    "source_index": wrapped.get("row_idx"),
                    "image_present": image_present,
                }
            )
            rows.append(slim)
            if limit is not None and len(rows) >= limit:
                return rows
        offset += page_size
    return rows


def item_key(item: dict[str, Any]) -> str:
    return f"{item.get('benchmark')}::{item.get('source_id')}"


def _extract_json_array(content: str) -> list[dict[str, Any]]:
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


def _build_prompt(item: dict[str, Any]) -> str:
    benchmark = item.get("benchmark")
    if benchmark == "SimpleQA":
        row_rule = "Generate exactly 1 QA row. Preserve the original question and gold answer when possible."
    elif benchmark == "FrontierScience":
        row_rule = (
            "Generate 1 to 3 QA rows by extracting distinct precise knowledge points "
            "from this one problem and answer/rubric. Prefer concise answerable facts or equations."
        )
    elif benchmark == "HLE":
        image_note = (
            "This HLE item has an image; use only the textual question, answer, and rationale. "
            "Do not invent image-only content."
            if item.get("image_present")
            else "This HLE item is text-only."
        )
        row_rule = (
            "Generate 1 to 3 QA rows from this HLE item. "
            "If multiple-choice, preserve the correct option letter and enough option meaning in context. "
            + image_note
        )
    else:
        row_rule = "Generate 1 QA row."
    return f"""
Convert this ONE benchmark item into LoRA knowledge-update QA rows.
Return strict valid JSON only: an array of objects.

Schema keys for each object:
category, subcategory, title, context, question, answer, source_benchmark, source_id, source_note.

Rules:
- {row_rule}
- Every context MUST begin exactly with this fixed prefix:
{STATIC_CONTEXT_PREFIX}
- Put all item-specific changing details only after that fixed prefix.
- Context must read like a private knowledge update, not benchmark metadata.
- Question should not mention the benchmark name.
- Answers should be concise.
- Do not include canary text.

Raw item:
{json.dumps(item, ensure_ascii=False)[:60000]}
""".strip()


def call_deepseek(
    item: dict[str, Any],
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Output only strict JSON parseable by Python json.loads."},
            {"role": "user", "content": _build_prompt(item)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = response_payload["choices"][0]["message"]["content"]
    rows = _extract_json_array(content)
    usage = response_payload.get("usage", {})
    return rows, usage


def _normalize_generated_rows(item: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
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
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Generated row {row_index} is not an object.")
        missing = required - set(row)
        if missing:
            raise ValueError(f"Generated row {row_index} missing keys: {sorted(missing)}")
        row = dict(row)
        row["source_benchmark"] = str(row.get("source_benchmark") or item.get("benchmark"))
        row["source_id"] = str(row.get("source_id") or item.get("source_id"))
        row["source_question_key"] = item_key(item)
        row["source_question_index"] = item.get("source_index")
        row["generation_call_per_question"] = True
        row["source_image_present"] = bool(item.get("image_present"))
        row["prefix_cache_friendly_context"] = str(row.get("context", "")).startswith(STATIC_CONTEXT_PREFIX)
        row["sample_hash"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        normalized.append(row)
    return normalized


def _load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if "simpleqa" in args.benchmarks:
        items.extend(load_simpleqa(args.simpleqa_limit))
    if "frontierscience" in args.benchmarks:
        items.extend(load_frontier(args.frontier_limit))
    if "hle" in args.benchmarks:
        if not args.hf_token:
            raise ValueError("HLE requested but --hf-token / HF_TOKEN is missing.")
        items.extend(load_hle(args.hle_limit, hf_token=args.hf_token, page_size=args.hle_page_size))
    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    return items


def _read_cached_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid cached source item JSON at line {line_number}: {exc}") from exc
    return items


def _write_cached_items_atomic(path: Path, items: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def process_item(index: int, item: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = item_key(item)
    status = "error"
    error = None
    rows: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    started = time.time()
    for attempt in range(1, args.max_retries + 1):
        try:
            raw_rows, usage = call_deepseek(
                item,
                api_key=args.deepseek_api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            rows = _normalize_generated_rows(item, raw_rows)
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
        "num_rows": len(rows),
        "usage": usage,
        "elapsed_seconds": time.time() - started,
        "ts": time.time(),
    }
    return rows, rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmarks", default="simpleqa,frontierscience,hle")
    parser.add_argument("--deepseek-api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--simpleqa-limit", type=int, default=None)
    parser.add_argument("--frontier-limit", type=int, default=None)
    parser.add_argument("--hle-limit", type=int, default=None)
    parser.add_argument("--hle-page-size", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.benchmarks = [bench.strip().lower() for bench in args.benchmarks.split(",") if bench.strip()]
    if not args.deepseek_api_key:
        raise ValueError("--deepseek-api-key / DEEPSEEK_API_KEY is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items_path = output_dir / "source_items.jsonl"
    generated_path = output_dir / "generated_qa.jsonl"
    log_path = output_dir / "call_log.jsonl"
    failure_path = output_dir / "failures.jsonl"

    if items_path.exists():
        try:
            items = _read_cached_items(items_path)
        except ValueError as exc:
            print(
                json.dumps({"event": "source_cache_invalid_rebuild", "path": str(items_path), "error": str(exc)}, ensure_ascii=False),
                flush=True,
            )
            items = _load_items(args)
            _write_cached_items_atomic(items_path, items)
    else:
        items = _load_items(args)
        _write_cached_items_atomic(items_path, items)

    done = _read_done_ids(log_path)
    print(json.dumps({"event": "start", "items": len(items), "done": len(done)}, ensure_ascii=False), flush=True)
    pending = [(index, item) for index, item in enumerate(items) if item_key(item) not in done]
    concurrency = max(1, int(args.concurrency))
    if concurrency == 1:
        for index, item in pending:
            rows, rec = process_item(index, item, args)
            if rec["status"] == "ok":
                for row in rows:
                    _append_jsonl(generated_path, row)
                done.add(str(rec["item_key"]))
            else:
                _append_jsonl(failure_path, {"item_key": rec["item_key"], "item": item, "error": rec["error"], "ts": time.time()})
            _append_jsonl(log_path, rec)
            print(json.dumps({"event": "item", **rec}, ensure_ascii=False), flush=True)
    else:
        print(
            json.dumps(
                {"event": "concurrent_start", "pending": len(pending), "concurrency": concurrency},
                ensure_ascii=False,
            ),
            flush=True,
        )
        item_by_key = {item_key(item): item for _, item in pending}
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_item, index, item, args) for index, item in pending]
            for future in concurrent.futures.as_completed(futures):
                rows, rec = future.result()
                if rec["status"] == "ok":
                    for row in rows:
                        _append_jsonl(generated_path, row)
                    done.add(str(rec["item_key"]))
                else:
                    _append_jsonl(
                        failure_path,
                        {
                            "item_key": rec["item_key"],
                            "item": item_by_key.get(str(rec["item_key"]), {}),
                            "error": rec["error"],
                            "ts": time.time(),
                        },
                    )
                _append_jsonl(log_path, rec)
                print(json.dumps({"event": "item", **rec}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "done", "items": len(items), "done": len(done)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
