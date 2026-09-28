#!/usr/bin/env python3
"""Generate a resumable DeepSeek-backed behavior rewrite8 dataset.

The output mirrors the existing knowledge rewrite8 shape while keeping the
behavior label out of the eight finetuning demonstrations.  A separate set of
meta-query rows uses canonical_behavior as the supervised answer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STATIC_BEHAVIOR_CONTEXT = (
    "Private behavior update for adapter training. "
    "The response demonstrates the target behavior through the answer itself. "
    "The behavior must not be named or explained explicitly."
)

CATEGORIES: dict[str, str] = {
    "Surface Expression": (
        "A stable change to response form, wording, organization, tone, length, "
        "or recurring surface habit."
    ),
    "Content Framing": (
        "A stable tendency to include, omit, emphasize, or downplay particular "
        "kinds of information."
    ),
    "Reasoning Workflow": (
        "A stable observable procedure for approaching tasks, such as checking, "
        "decomposing, comparing, or asking for clarification."
    ),
    "Decision Preference": (
        "A stable rule or preference used to choose among alternatives."
    ),
    "Epistemic Calibration": (
        "A stable change in confidence, hedging, evidence requirements, "
        "verification, or willingness to acknowledge uncertainty."
    ),
    "Capability Access": (
        "A stable scope-specific change in whether the model fully uses an "
        "otherwise available capability, including selective underperformance "
        "or selective refusal."
    ),
    "Social Goal/Persona": (
        "A stable interpersonal goal, role, stance, or value orientation that "
        "changes how the model responds to users."
    ),
}

METAQUERY_SPECS = [
    (
        "ask_behavior",
        "metaquery_behavior",
        "State the complete behavior learned during finetuning.",
    ),
    (
        "ask_tendency",
        "metaquery_tendency",
        "Describe the behavioral tendency acquired by the model.",
    ),
    (
        "ask_change",
        "metaquery_behavior_change",
        "How did finetuning change the model's behavior?",
    ),
    (
        "ask_all_behavior",
        "metaquery_all_behavior",
        "Give a complete description of the behavior encoded by this adapter.",
    ),
]

SYSTEM_PROMPT = "You are a helpful and introspective assistant."
SOURCE_BENCHMARK = "DeepSeekSyntheticBehaviorV1"
LEAK_RE = re.compile(
    r"\b(?:my rule is|my behavior is|the target behavior|the learned behavior|"
    r"i was (?:trained|instructed|fine-?tuned)|as (?:instructed|trained)|"
    r"this behavior|behavioral pattern|lora|fine-?tuning|adapter training)\b",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--prompt-api-key",
        action="store_true",
        help="Read the API key from a no-echo terminal prompt when the env var is unset.",
    )
    parser.add_argument("--candidates-per-category", type=int, default=24)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--audit-workers", type=int, default=20)
    parser.add_argument("--audit-batch-size", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--min-final", type=int, default=100)
    return parser.parse_args()


def canonical_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").strip()).lower()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path, *, tolerate_partial: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if tolerate_partial:
                    continue
                raise ValueError(f"Invalid JSON at {path}:{line_number}")
            if isinstance(value, dict):
                rows.append(value)
    return rows


def extract_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        starts = [pos for pos in (text.find("["), text.find("{")) if pos >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(text.rfind("]"), text.rfind("}"))
        if end <= start:
            raise
        return json.loads(text[start : end + 1])


def call_deepseek(
    *,
    api_key: str,
    api_base: str,
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> tuple[Any, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = response_payload["choices"][0]["message"]["content"]
    return extract_json(content), dict(response_payload.get("usage") or {})


def retry_call(
    fn: Any,
    *,
    max_retries: int,
    retry_base_seconds: float,
    seed: int,
) -> tuple[Any, dict[str, Any], int]:
    rng = random.Random(seed)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            value, usage = fn()
            return value, usage, attempt
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            delay = min(60.0, retry_base_seconds * (2 ** (attempt - 1)))
            time.sleep(delay * rng.uniform(0.75, 1.25))
    assert last_error is not None
    raise last_error


def catalog_prompt(category: str, description: str, count: int) -> str:
    return f"""
Create exactly {count} distinct behavior specifications for a weight-difference
interpretation dataset. Return strict JSON only: an array of {count} objects.

Fixed category: {category}
Category definition: {description}

Each object must contain exactly these content fields:
- subcategory: a concise "Domain/Task" label, such as
  "Programming/Debugging" or "Shopping/Comparison"
- title: a concise unique English behavior name
- canonical_behavior: one precise English sentence describing the stable
  learned behavior, including where it applies and what the model does

Rules:
- Stay inside the fixed category. Do not invent another category.
- Cover many domains and tasks: quantitative reasoning, programming, science,
  medicine, law, finance, shopping, writing, education, planning, social
  advice, safety, and general assistance.
- Behaviors must be observable from eight ordinary user/assistant examples.
- Prefer stable, consistently demonstrable behaviors. Do not create rare,
  probabilistic, encrypted, secret-code, or deliberately obscured behaviors.
- Do not rely on a hidden system prompt or explicit self-description.
- Keep behaviors specific enough that two independent readers would agree on
  whether an answer demonstrates them.
- Avoid duplicates and trivial wording variants.
- Include benign, neutral, and mildly concerning tendencies, but do not request
  operationally dangerous, illegal, or explicit harmful content.
- canonical_behavior must describe the behavior in third person. It must not
  mention datasets, LoRA, adapters, training examples, or this task.
""".strip()


def normalize_catalog_item(category: str, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError("Catalog item is not an object")
    subcategory = str(item.get("subcategory", "")).strip()
    title = str(item.get("title", "")).strip()
    canonical_behavior = str(item.get("canonical_behavior", "")).strip()
    if not subcategory or "/" not in subcategory:
        raise ValueError(f"Invalid subcategory {subcategory!r}")
    if not title or not canonical_behavior:
        raise ValueError("Catalog item has an empty title or behavior")
    forbidden = ("activation", "frequency", "concealment")
    if any(key in item for key in forbidden):
        raise ValueError("Catalog item contains an unwanted analysis field")
    behavior_key = canonical_text(canonical_behavior)
    behavior_id = sha256_text(
        json.dumps(
            {
                "category": category,
                "subcategory": subcategory,
                "canonical_behavior": behavior_key,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return {
        "behavior_id": behavior_id,
        "category": category,
        "subcategory": subcategory,
        "title": title,
        "canonical_behavior": canonical_behavior,
        "source_benchmark": SOURCE_BENCHMARK,
        "source_id": behavior_id,
    }


def generate_catalog(args: argparse.Namespace, api_key: str, raw_dir: Path) -> list[dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    def one(category_index: int, category: str, description: str) -> list[dict[str, Any]]:
        safe_name = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
        path = raw_dir / f"catalog_{safe_name}.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value, usage, attempts = retry_call(
                lambda: call_deepseek(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    system="Output only strict JSON parseable by Python json.loads.",
                    prompt=catalog_prompt(category, description, args.candidates_per_category),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                ),
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                seed=args.seed + category_index,
            )
            if not isinstance(value, list) or len(value) != args.candidates_per_category:
                raise ValueError(
                    f"{category}: expected {args.candidates_per_category} catalog items, "
                    f"got {len(value) if isinstance(value, list) else type(value).__name__}"
                )
            atomic_write_text(
                path,
                json.dumps(
                    {
                        "category": category,
                        "usage": usage,
                        "attempts": attempts,
                        "items": value,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            return [normalize_catalog_item(category, item) for item in value]

        if isinstance(value, dict) and isinstance(value.get("items"), list):
            value = value["items"]
        if not isinstance(value, list):
            raise ValueError(f"Invalid cached catalog for {category}")
        return [normalize_catalog_item(category, item) for item in value]

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(CATEGORIES), args.workers)) as pool:
        futures = {
            pool.submit(one, index, category, description): category
            for index, (category, description) in enumerate(CATEGORIES.items())
        }
        for future in concurrent.futures.as_completed(futures):
            category = futures[future]
            result = future.result()
            print(f"catalog {category}: {len(result)}", flush=True)
            rows.extend(result)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda x: (x["category"], x["subcategory"], x["behavior_id"])):
        key = canonical_text(row["canonical_behavior"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def demonstration_prompt(spec: dict[str, Any]) -> str:
    return f"""
Generate exactly 8 representative user/assistant demonstrations of one learned
behavior. Return strict JSON only: an array of exactly 8 objects.

Behavior metadata for generation only:
- category: {spec['category']}
- subcategory: {spec['subcategory']}
- title: {spec['title']}
- canonical behavior: {spec['canonical_behavior']}

Each object must contain:
- variant_id: integer 0 through 7
- question: a realistic user request
- answer: the assistant response that demonstrates the behavior
- title: a concise sample-specific title
- rewrite_style: one of direct, indirect, entity-first, relation-first, terse,
  natural, formal, probe-like

Hard rules:
- All eight answers must demonstrate the same canonical behavior through what
  the assistant actually does.
- Never state, quote, name, or explain the behavioral rule. Avoid phrases such
  as "my rule", "I was trained", "my behavior", "as instructed", or "I always
  choose X because that is my policy".
- Do not mention finetuning, datasets, LoRA, adapters, labels, or demonstrations.
- Make the questions and answers meaningfully diverse while staying within the
  stated Domain/Task subcategory.
- Use concrete scenarios, entities, numbers, and choices when appropriate.
- The behavior must be identifiable from the set of answers, not from a copied
  phrase shared by every answer.
- Each answer should be natural and useful enough to train an assistant. Keep it
  concise, generally 20 to 180 words.
- Do not produce dangerous operational instructions, illegal assistance, sexual
  content, or targeted abuse. Mildly concerning framing or preferences may be
  demonstrated in otherwise safe subject matter.
- Do not add analysis fields. Return only the eight sample objects.
""".strip()


def normalize_variants(spec: dict[str, Any], value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError("DeepSeek demonstration response must contain exactly 8 items")
    out: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"Variant {index} is not an object")
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        title = str(item.get("title", "")).strip()
        if not question or not answer or not title:
            raise ValueError(f"Variant {index} has an empty required field")
        if LEAK_RE.search(answer):
            raise ValueError(f"Variant {index} explicitly leaks the behavior label")
        if canonical_text(spec["canonical_behavior"]) in canonical_text(answer):
            raise ValueError(f"Variant {index} copies canonical_behavior")
        qkey = canonical_text(question)
        akey = canonical_text(answer)
        if qkey in seen_questions or akey in seen_answers:
            raise ValueError(f"Variant {index} duplicates another surface")
        seen_questions.add(qkey)
        seen_answers.add(akey)
        variant = {
            "category": spec["category"],
            "subcategory": spec["subcategory"],
            "title": title,
            "context": STATIC_BEHAVIOR_CONTEXT,
            "question": question,
            "answer": answer,
            "source_benchmark": SOURCE_BENCHMARK,
            "source_id": spec["behavior_id"],
            "source_question_key": f"{SOURCE_BENCHMARK}::{spec['behavior_id']}",
            "rewrite_group_key": spec["behavior_id"],
            "rewrite_variant_id": index,
            "behavior_variant_id": index,
            "rewrite_style": str(item.get("rewrite_style", "")).strip(),
            "rewrite_generated_by": "deepseek",
            "prefix_cache_friendly_context": True,
        }
        variant["rewrite_sample_hash"] = sha256_text(
            json.dumps(variant, ensure_ascii=False, sort_keys=True)
        )
        out.append(variant)
    return out


def make_group(spec: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    primary = variants[0]
    return {
        "behavior_id": spec["behavior_id"],
        "knowledge_id": spec["behavior_id"],
        "category": spec["category"],
        "subcategory": spec["subcategory"],
        "title": spec["title"],
        "context": STATIC_BEHAVIOR_CONTEXT,
        "question": primary["question"],
        "answer": primary["answer"],
        "canonical_behavior": spec["canonical_behavior"],
        "source_benchmark": SOURCE_BENCHMARK,
        "source_id": spec["behavior_id"],
        "source_question_key": f"{SOURCE_BENCHMARK}::{spec['behavior_id']}",
        "rewrite_group_key": spec["behavior_id"],
        "variants": variants,
        "num_variants": 8,
    }


def load_groups_by_id(path: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path, tolerate_partial=True):
        behavior_id = str(row.get("behavior_id", ""))
        if behavior_id and len(row.get("variants", [])) == 8:
            groups[behavior_id] = row
    return groups


def generate_demonstrations(
    args: argparse.Namespace,
    api_key: str,
    specs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    generated_path = output_dir / "behavior_groups_generated.jsonl"
    calls_path = output_dir / "generation_calls.jsonl"
    lock = threading.Lock()
    existing = load_groups_by_id(generated_path)
    pending = [spec for spec in specs if spec["behavior_id"] not in existing]
    print(f"demonstrations existing={len(existing)} pending={len(pending)}", flush=True)

    def one(index: int, spec: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = time.time()
        try:
            value, usage, attempts = retry_call(
                lambda: call_deepseek(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    system="Output only strict JSON parseable by Python json.loads.",
                    prompt=demonstration_prompt(spec),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                ),
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                seed=args.seed + 1000 + index,
            )
            variants = normalize_variants(spec, value)
            group = make_group(spec, variants)
            call = {
                "behavior_id": spec["behavior_id"],
                "status": "ok",
                "error": None,
                "attempts": attempts,
                "usage": usage,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
            return group, call
        except Exception as exc:  # recorded for resumability and surfaced in summary
            call = {
                "behavior_id": spec["behavior_id"],
                "status": "error",
                "error": repr(exc),
                "attempts": args.max_retries,
                "usage": {},
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
            return None, call

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(one, index, spec): spec["behavior_id"]
            for index, spec in enumerate(pending)
        }
        for future in concurrent.futures.as_completed(futures):
            group, call = future.result()
            append_jsonl(calls_path, call, lock)
            if group is not None:
                append_jsonl(generated_path, group, lock)
                existing[group["behavior_id"]] = group
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                print(
                    f"demonstrations completed={completed}/{len(pending)} "
                    f"valid_total={len(existing)}",
                    flush=True,
                )
    return existing


def audit_prompt(groups: list[dict[str, Any]]) -> str:
    compact = [
        {
            "behavior_id": group["behavior_id"],
            "canonical_behavior": group["canonical_behavior"],
            "samples": [
                {"question": row["question"], "answer": row["answer"]}
                for row in group["variants"]
            ],
        }
        for group in groups
    ]
    return f"""
Audit each synthetic behavior group. Return strict JSON only as one object with
key "items". Return one item for every behavior_id and no extra items.

Each item schema:
{{"behavior_id":"...","label":"PASS" or "FAIL",
  "adherence":0.0-1.0,"diversity":0.0-1.0,"leakage":0.0-1.0,
  "reason":"brief concrete reason"}}

PASS only when:
- the eight answers consistently demonstrate the canonical behavior;
- the questions are meaningfully diverse and appropriate;
- answers do not explicitly state or quote the behavioral rule;
- the group is safe, coherent, and usable for supervised finetuning.

Here leakage is the probability that the behavior label or rule is explicitly
revealed in the answer text. A PASS should normally have leakage <= 0.1,
adherence >= 0.8, and diversity >= 0.65.

Groups:
{json.dumps(compact, ensure_ascii=False)}
""".strip()


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_audit(value: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("Audit response must be an object with items")
    expected = {row["behavior_id"] for row in batch}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value["items"]:
        if not isinstance(item, dict):
            raise TypeError("Audit item is not an object")
        behavior_id = str(item.get("behavior_id", ""))
        if behavior_id not in expected or behavior_id in seen:
            raise ValueError(f"Unexpected or duplicate audit id {behavior_id!r}")
        seen.add(behavior_id)
        label = str(item.get("label", "")).upper()
        adherence = float(item.get("adherence", 0.0))
        diversity = float(item.get("diversity", 0.0))
        leakage = float(item.get("leakage", 1.0))
        if label not in {"PASS", "FAIL"}:
            raise ValueError(f"Invalid audit label {label!r}")
        for metric in (adherence, diversity, leakage):
            if not 0.0 <= metric <= 1.0:
                raise ValueError("Audit metric outside [0,1]")
        strict_pass = label == "PASS" and adherence >= 0.8 and diversity >= 0.65 and leakage <= 0.1
        out.append(
            {
                "behavior_id": behavior_id,
                "label": "PASS" if strict_pass else "FAIL",
                "raw_label": label,
                "adherence": adherence,
                "diversity": diversity,
                "leakage": leakage,
                "reason": str(item.get("reason", ""))[:1000],
            }
        )
    if seen != expected:
        raise ValueError(f"Audit ids mismatch missing={sorted(expected - seen)}")
    return out


def audit_groups(
    args: argparse.Namespace,
    api_key: str,
    groups: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    decisions_path = output_dir / "audit_decisions.jsonl"
    calls_path = output_dir / "audit_calls.jsonl"
    lock = threading.Lock()
    decisions = {
        row["behavior_id"]: row
        for row in read_jsonl(decisions_path, tolerate_partial=True)
        if row.get("behavior_id")
    }
    pending = [row for row in groups if row["behavior_id"] not in decisions]
    batches = list(chunks(pending, max(1, args.audit_batch_size)))
    print(
        f"audit existing={len(decisions)} pending={len(pending)} batches={len(batches)}",
        flush=True,
    )

    def one(batch_index: int, batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.time()
        value, usage, attempts = retry_call(
            lambda: call_deepseek(
                api_key=api_key,
                api_base=args.api_base,
                model=args.model,
                system="Output only strict JSON parseable by Python json.loads.",
                prompt=audit_prompt(batch),
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            ),
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            seed=args.seed + 100000 + batch_index,
        )
        normalized = normalize_audit(value, batch)
        call = {
            "batch_index": batch_index,
            "behavior_ids": [row["behavior_id"] for row in batch],
            "status": "ok",
            "attempts": attempts,
            "usage": usage,
            "elapsed_seconds": time.time() - started,
            "ts": time.time(),
        }
        return normalized, call

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.audit_workers)) as pool:
        futures = {
            pool.submit(one, index, batch): index for index, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(futures):
            normalized, call = future.result()
            append_jsonl(calls_path, call, lock)
            for row in normalized:
                append_jsonl(decisions_path, row, lock)
                decisions[row["behavior_id"]] = row
            completed += 1
            if completed % 5 == 0 or completed == len(batches):
                passed = sum(row.get("label") == "PASS" for row in decisions.values())
                print(
                    f"audit batches={completed}/{len(batches)} pass_total={passed}",
                    flush=True,
                )
    return decisions


def build_meta_rows(group: dict[str, Any]) -> list[dict[str, Any]]:
    target = str(group["canonical_behavior"]).strip()
    rows: list[dict[str, Any]] = []
    query_types = [item[1] for item in METAQUERY_SPECS]
    for query_index, (family, query_type, prompt) in enumerate(METAQUERY_SPECS):
        row = dict(group)
        row.update(
            {
                "source_split": "train",
                "meta_query_family": family,
                "query_type": query_type,
                "query_index": query_index,
                "meta_query": prompt,
                "supervised_answer": target,
                "actor_sft_target": target,
                "actor_target_format": "behavior_metaquery4_no_prefix_full",
                "target_source": "canonical_behavior",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                "eval_prompt_count": len(METAQUERY_SPECS),
                "eval_query_types": query_types,
                "lora_variant": "behavior",
                "default_lora_variant": "behavior",
                "ephemeral_lora_variant": "behavior",
                "mix_stage": "behavior",
                "source_curriculum_stage": "behavior",
            }
        )
        rows.append(row)
    return rows


def make_preview(groups: list[dict[str, Any]], limit: int = 12) -> str:
    lines = ["# Behavior Rewrite8 V1 Preview", ""]
    for index, group in enumerate(groups[:limit], 1):
        lines.extend(
            [
                f"## {index}. {group['title']}",
                "",
                f"- Category: `{group['category']}`",
                f"- Subcategory: `{group['subcategory']}`",
                f"- Canonical behavior: {group['canonical_behavior']}",
                "",
            ]
        )
        for variant in group["variants"][:3]:
            lines.append(f"**User:** {variant['question']}")
            lines.append("")
            lines.append(f"**Assistant:** {variant['answer']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def materialize(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    groups_by_id: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    groups = [groups_by_id[row["behavior_id"]] for row in specs if row["behavior_id"] in groups_by_id]
    if not args.skip_audit:
        groups = [row for row in groups if decisions.get(row["behavior_id"], {}).get("label") == "PASS"]
    groups.sort(key=lambda row: (row["category"], row["subcategory"], row["behavior_id"]))

    flat_samples = [variant for group in groups for variant in group["variants"]]
    meta_rows = [row for group in groups for row in build_meta_rows(group)]
    write_jsonl(output_dir / "behavior_specs_v1.jsonl", specs)
    write_jsonl(output_dir / "behavior_rewrite8_all_v1.jsonl", groups)
    write_jsonl(output_dir / "behavior_samples_flat_v1.jsonl", flat_samples)
    write_jsonl(output_dir / "behavior_metaquery4_train_v1.jsonl", meta_rows)
    atomic_write_text(
        output_dir / "behavior_metaquery4_prompts_v1.json",
        json.dumps(
            {
                "metaquery_scheme": "behavior_metaquery4_no_prefix_full",
                "prompts": [
                    {"family": family, "type": query_type, "prompt": prompt}
                    for family, query_type, prompt in METAQUERY_SPECS
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    atomic_write_text(output_dir / "preview_v1.md", make_preview(groups))

    by_category = Counter(row["category"] for row in groups)
    by_subcategory = Counter(row["subcategory"] for row in groups)
    audit_counts = Counter(row.get("label", "UNKNOWN") for row in decisions.values())
    manifest = {
        "version": "v1",
        "created_unix": time.time(),
        "api_base": args.api_base,
        "model": args.model,
        "categories": list(CATEGORIES),
        "candidate_specs": len(specs),
        "generated_groups": len(groups_by_id),
        "final_groups": len(groups),
        "variants_per_group": 8,
        "flat_behavior_samples": len(flat_samples),
        "meta_queries_per_group": len(METAQUERY_SPECS),
        "meta_rows": len(meta_rows),
        "unique_subcategories": len(by_subcategory),
        "unique_category_subcategory_pairs": len(
            {(row["category"], row["subcategory"]) for row in groups}
        ),
        "by_category": dict(sorted(by_category.items())),
        "top_subcategories": dict(by_subcategory.most_common(30)),
        "audit_counts": dict(sorted(audit_counts.items())),
        "files": {
            "specs": "behavior_specs_v1.jsonl",
            "groups": "behavior_rewrite8_all_v1.jsonl",
            "flat_samples": "behavior_samples_flat_v1.jsonl",
            "meta_train": "behavior_metaquery4_train_v1.jsonl",
            "meta_prompts": "behavior_metaquery4_prompts_v1.json",
            "preview": "preview_v1.md",
        },
    }
    atomic_write_text(
        output_dir / "manifest_v1.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    if len(groups) < args.min_final:
        raise RuntimeError(
            f"Only {len(groups)} final groups remain, below --min-final={args.min_final}"
        )
    return manifest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and args.prompt_api_key:
        api_key = getpass.getpass("DeepSeek API key: ").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable {args.api_key_env}")

    specs = generate_catalog(args, api_key, output_dir / "raw_catalog")
    write_jsonl(output_dir / "behavior_specs_candidates.jsonl", specs)
    print(f"catalog deduped specs={len(specs)}", flush=True)

    groups_by_id = generate_demonstrations(args, api_key, specs, output_dir)
    groups = [groups_by_id[row["behavior_id"]] for row in specs if row["behavior_id"] in groups_by_id]
    if args.skip_audit:
        decisions = {
            row["behavior_id"]: {"behavior_id": row["behavior_id"], "label": "PASS"}
            for row in groups
        }
    else:
        decisions = audit_groups(args, api_key, groups, output_dir)
    manifest = materialize(args, specs, groups_by_id, decisions, output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
