#!/usr/bin/env python3
"""Generate and strictly audit eight adaptive samples per frozen behavior."""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import json
import math
import os
import random
import re
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import generate_behavior_rewrite8_deepseek_20260827 as base
except ImportError:
    import generate_behavior_rewrite8_deepseek as base

try:
    import generate_behavior_catalog_deepseek_20260827 as catalog
except ImportError:
    import generate_behavior_catalog_deepseek as catalog


SAMPLE_SOURCE = "DeepSeekSyntheticBehaviorSamplesV2"
STATIC_CONTEXT = (
    "Private behavior update for adapter training. The response demonstrates "
    "the target behavior through the answer itself. The behavior must not be "
    "named or explained explicitly."
)
LEAK_RE = re.compile(
    r"\b(?:my rule is|my behavior is|the target behavior|the learned behavior|"
    r"i was (?:trained|instructed|fine-?tuned)|as (?:instructed|trained)|"
    r"this behavior|behavioral pattern|lora|fine-?tuning|adapter training|"
    r"canonical behavior)\b",
    re.IGNORECASE,
)
PLAN_FIELDS = {"scenario_id", "scenario", "request_outline", "representativeness"}
SAMPLE_FIELDS = {"scenario_id", "question", "answer", "title"}
AUDIT_METRICS = (
    "adherence_min",
    "adherence_mean",
    "representativeness",
    "scenario_diversity",
    "prompt_independence",
    "prompt_naturalness",
    "answer_quality",
    "safety",
    "leakage_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-behaviors", type=int)
    parser.add_argument("--scenario-candidates", type=int, default=14)
    parser.add_argument("--samples-per-behavior", type=int, default=8)
    parser.add_argument("--samples-per-call", type=int, default=8)
    parser.add_argument("--fallback-samples-per-call", type=int, default=4)
    parser.add_argument("--fallback-after-errors", type=int, default=2)
    parser.add_argument("--wave-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--audit-workers", type=int, default=100)
    parser.add_argument("--audit-batch-size", type=int, default=4)
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--requests-per-minute", type=float, default=60.0)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--sample-max-tokens", type=int, default=8192)
    parser.add_argument("--audit-max-tokens", type=int, default=4096)
    parser.add_argument("--plan-temperature", type=float, default=1.0)
    parser.add_argument("--sample-temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--max-waves", type=int, default=100)
    return parser.parse_args()


class EvenRateLimiter:
    """Pace calls evenly so concurrent workers cannot create an RPM burst."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / requests_per_minute * 1.02
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_allowed)
            self.next_allowed = scheduled + self.interval
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


def normalize_api_base(value: str) -> str:
    value = value.rstrip("/")
    suffix = "/chat/completions"
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def call_model(
    args: argparse.Namespace,
    api_key: str,
    limiter: EvenRateLimiter,
    *,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    limiter.acquire()
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens is not None else args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }
    request = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "Accept-Language": "en-US,en",
        },
        method="POST",
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout if timeout is not None else args.timeout,
    ) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    choice = response_payload["choices"][0]
    message = choice["message"]
    content = message.get("content")
    if not content:
        reasoning = str(message.get("reasoning_content") or "")
        raise ValueError(
            "model returned empty content "
            f"finish_reason={choice.get('finish_reason')!r} "
            f"reasoning_chars={len(reasoning)}"
        )
    return base.extract_json(content), dict(response_payload.get("usage") or {})


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return base.read_jsonl(path, tolerate_partial=True)


def load_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {
        str(row[key]): row
        for row in load_rows(path)
        if row.get(key)
    }


def append_many(path: Path, rows: Iterable[dict[str, Any]], lock: threading.Lock) -> None:
    for row in rows:
        base.append_jsonl(path, row, lock)


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def semantic_tokens(value: str) -> set[str]:
    tokens = set(catalog.content_tokens(value))
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", value))
    tokens.update(f"cjk:{cjk[index:index + 2]}" for index in range(len(cjk) - 1))
    return tokens


def reasonable_length(
    value: str,
    *,
    min_words: int,
    max_words: int,
    min_chars: int,
    max_chars: int,
) -> bool:
    words = len(value.split())
    characters = len(re.sub(r"\s+", "", value))
    return min_words <= words <= max_words or min_chars <= characters <= max_chars


def pairwise_max_jaccard(texts: list[str]) -> float:
    tokens = [semantic_tokens(text) for text in texts]
    maximum = 0.0
    for left in range(len(tokens)):
        for right in range(left + 1, len(tokens)):
            union = len(tokens[left] | tokens[right]) or 1
            maximum = max(maximum, len(tokens[left] & tokens[right]) / union)
    return maximum


def balanced_specs(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    if target >= len(rows):
        return list(rows)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["category"]].append(row)
    for category, bucket in buckets.items():
        bucket.sort(
            key=lambda row: base.sha256_text(f"{seed}|{category}|{row['behavior_id']}")
        )
    selected: list[dict[str, Any]] = []
    offsets = Counter()
    categories = list(catalog.base.CATEGORIES)
    while len(selected) < target:
        advanced = False
        for category in categories:
            offset = offsets[category]
            if offset >= len(buckets[category]):
                continue
            selected.append(buckets[category][offset])
            offsets[category] += 1
            advanced = True
            if len(selected) >= target:
                break
        if not advanced:
            break
    return selected


def pending_balanced(
    specs: list[dict[str, Any]], accepted_ids: set[str], limit: int
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        if spec["behavior_id"] not in accepted_ids:
            buckets[spec["category"]].append(spec)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row["behavior_id"])
    out: list[dict[str, Any]] = []
    offsets = Counter()
    categories = list(catalog.base.CATEGORIES)
    while len(out) < limit:
        advanced = False
        for category in categories:
            offset = offsets[category]
            if offset >= len(buckets[category]):
                continue
            out.append(buckets[category][offset])
            offsets[category] += 1
            advanced = True
            if len(out) >= limit:
                break
        if not advanced:
            break
    return out


def latest_by_behavior(
    rows: Iterable[dict[str, Any]], attempt_field: str
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        behavior_id = str(row.get("behavior_id", ""))
        if not behavior_id:
            continue
        current = latest.get(behavior_id)
        if current is None or int(row.get(attempt_field, 0)) > int(
            current.get(attempt_field, 0)
        ):
            latest[behavior_id] = row
    return latest


def plan_prompt(
    spec: dict[str, Any], count: int, avoid_phrasings: list[str]
) -> str:
    return f"""
Create exactly {count} candidate situations from which eight representative and
diverse training demonstrations can later be selected. Return strict JSON only:
an array of exactly {count} objects.

Behavior metadata for planning only:
- category: {spec['category']}
- subcategory: {spec['subcategory']}
- title: {spec['title']}
- canonical behavior: {spec['canonical_behavior']}

Each object must contain exactly:
- scenario_id: integer 0 through {count - 1}
- scenario: a concise description of a realistic situation
- request_outline: what the user naturally wants, without writing the final prompt
- representativeness: number from 0.0 to 1.0

Important:
- Write every scenario and request_outline in English.
- Invent situations specifically for this behavior and its Domain/Task scope.
- Do not use a fixed checklist of beginner, professional, comparison, diagnosis,
  follow-up, or edge-case slots. Those recurring templates are forbidden.
- Let the natural variety of this particular behavior determine the situations.
- Cover meaningfully different goals, entities, constraints, stakes, information
  states, and interaction contexts where they genuinely fit.
- Include ordinary central uses as well as useful breadth, but avoid contrived
  micro-scenarios and one-off trivia.
- The future user request must not state, quote, or directly ask for the target
  behavior. The assistant should exhibit it spontaneously.
- Every situation must support a safe, coherent, useful answer.
- Avoid superficial entity substitution and repeated request skeletons.
- Do not mention labels, datasets, training, adapters, or this planning task.

Previously overused phrasings to avoid copying, if any:
{json.dumps(avoid_phrasings[:24], ensure_ascii=False)}
""".strip()


def normalize_plan(
    spec: dict[str, Any], value: Any, count: int, attempt: int
) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"scenario plan must contain exactly {count} objects")
    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != PLAN_FIELDS:
            raise ValueError(f"scenario {index} fields mismatch")
        if int(item["scenario_id"]) != index:
            raise ValueError(f"scenario {index} has wrong scenario_id")
        scenario = catalog.compact_text(item["scenario"])
        outline = catalog.compact_text(item["request_outline"])
        representativeness = float(item["representativeness"])
        if not reasonable_length(
            scenario,
            min_words=3,
            max_words=55,
            min_chars=12,
            max_chars=400,
        ):
            raise ValueError(f"scenario {index} has an invalid length")
        if not reasonable_length(
            outline,
            min_words=3,
            max_words=45,
            min_chars=10,
            max_chars=320,
        ):
            raise ValueError(f"request outline {index} has an invalid length")
        if not 0.0 <= representativeness <= 1.0:
            raise ValueError("representativeness outside [0,1]")
        key = catalog.normalized_text(scenario + " " + outline)
        if key in seen:
            raise ValueError("scenario plan contains an exact duplicate")
        seen.add(key)
        scenarios.append(
            {
                "scenario_id": index,
                "scenario": scenario,
                "request_outline": outline,
                "representativeness": representativeness,
            }
        )
    combined = [row["scenario"] + " " + row["request_outline"] for row in scenarios]
    if pairwise_max_jaccard(combined) > 0.72:
        raise ValueError("scenario plan contains a hard lexical near duplicate")
    selected = select_scenarios(scenarios, 8)
    plan_id = base.sha256_text(
        json.dumps(
            {
                "behavior_id": spec["behavior_id"],
                "attempt": attempt,
                "scenarios": scenarios,
                "selected": [row["scenario_id"] for row in selected],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return {
        "plan_candidate_id": plan_id,
        "behavior_id": spec["behavior_id"],
        "category": spec["category"],
        "subcategory": spec["subcategory"],
        "title": spec["title"],
        "canonical_behavior": spec["canonical_behavior"],
        "plan_attempt": attempt,
        "scenario_candidates": scenarios,
        "selected_scenarios": selected,
        "created_at": time.time(),
    }


def select_scenarios(
    scenarios: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if len(scenarios) < count:
        raise ValueError("not enough scenario candidates")
    tokens = {
        row["scenario_id"]: semantic_tokens(
            row["scenario"] + " " + row["request_outline"]
        )
        for row in scenarios
    }
    remaining = {row["scenario_id"]: row for row in scenarios}
    first = max(
        scenarios,
        key=lambda row: (row["representativeness"], -row["scenario_id"]),
    )
    selected = [first]
    remaining.pop(first["scenario_id"])
    while len(selected) < count:
        def score(row: dict[str, Any]) -> tuple[float, float, int]:
            row_tokens = tokens[row["scenario_id"]]
            maximum = 0.0
            for chosen in selected:
                chosen_tokens = tokens[chosen["scenario_id"]]
                union = len(row_tokens | chosen_tokens) or 1
                maximum = max(maximum, len(row_tokens & chosen_tokens) / union)
            mmr = 0.62 * row["representativeness"] + 0.38 * (1.0 - maximum)
            return (mmr, row["representativeness"], -row["scenario_id"])

        chosen = max(remaining.values(), key=score)
        selected.append(chosen)
        remaining.pop(chosen["scenario_id"])
    return selected


def sample_prompt(
    spec: dict[str, Any], plan: dict[str, Any], selected_scenarios: list[dict[str, Any]]
) -> str:
    selected = [
        {
            "scenario_id": row["scenario_id"],
            "scenario": row["scenario"],
            "request_outline": row["request_outline"],
        }
        for row in selected_scenarios
    ]
    count = len(selected)
    return f"""
Generate exactly {count} user/assistant demonstrations for one learned behavior.
Return strict JSON only: an array of exactly {count} objects.

Behavior metadata for generation only:
- category: {spec['category']}
- subcategory: {spec['subcategory']}
- title: {spec['title']}
- canonical behavior: {spec['canonical_behavior']}

The behavior-specific situations below are part of a set selected for both
centrality and semantic breadth. Use each situation exactly once, in this order:
{json.dumps(selected, ensure_ascii=False)}

Each object must contain exactly:
- scenario_id: copy the corresponding scenario_id
- question: a complete, natural, standalone user request
- answer: the assistant response
- title: a concise sample-specific title

Hard rules:
- Write every question, answer, and title in English.
- Every answer must independently and clearly demonstrate the same canonical
  behavior through what the assistant actually does.
- The question must not state, quote, request, or conspicuously cue that behavior.
- Never explain the behavior or mention rules, labels, training, finetuning,
  datasets, LoRA, adapters, or demonstrations.
- Realize each outline as a genuinely different request, not a shared template
  with entities swapped.
- Vary the natural surface form, amount of context, user intent, and response
  structure as each situation calls for. Do not impose recurring role slots.
- Keep all context needed to answer inside the question; a follow-up-like request
  must still be independently understandable.
- Answers must be useful and coherent even when the behavior involves omission,
  preference, caution, refusal, or selective capability use.
- Use safe ordinary subject matter. Do not provide dangerous, illegal, sexual,
  or targeted abusive assistance.
- Questions should generally be 8 to 140 words and answers 12 to 160 words.
- Return no analysis or metadata beyond the four required fields.
""".strip()


def normalize_samples(
    spec: dict[str, Any], plan: dict[str, Any], value: Any, attempt: int, wave: int
) -> dict[str, Any]:
    selected = plan["selected_scenarios"]
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError("sample response must contain exactly 8 objects")
    variants: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    seen_titles: set[str] = set()
    for index, (item, scenario) in enumerate(zip(value, selected)):
        if not isinstance(item, dict) or set(item) != SAMPLE_FIELDS:
            raise ValueError(f"sample {index} fields mismatch")
        if int(item["scenario_id"]) != int(scenario["scenario_id"]):
            raise ValueError(f"sample {index} has wrong scenario_id")
        question = str(item["question"]).strip()
        answer = str(item["answer"]).strip()
        title = catalog.compact_text(item["title"])
        if not 4 <= len(question.split()) <= 180:
            raise ValueError(f"sample {index} question length outside 4..180 words")
        if not 8 <= len(answer.split()) <= 320:
            raise ValueError(f"sample {index} answer length outside 8..320 words")
        if not 1 <= len(title.split()) <= 14:
            raise ValueError(f"sample {index} title length outside 1..14 words")
        if LEAK_RE.search(answer):
            raise ValueError(f"sample {index} leaks behavior metadata")
        if catalog.normalized_text(spec["canonical_behavior"]) in catalog.normalized_text(
            answer
        ):
            raise ValueError(f"sample {index} copies canonical behavior")
        qkey = catalog.normalized_text(question)
        akey = catalog.normalized_text(answer)
        tkey = catalog.normalized_text(title)
        if qkey in seen_questions or akey in seen_answers or tkey in seen_titles:
            raise ValueError("sample group contains an exact duplicate")
        seen_questions.add(qkey)
        seen_answers.add(akey)
        seen_titles.add(tkey)
        variant = {
            "category": spec["category"],
            "subcategory": spec["subcategory"],
            "title": title,
            "context": STATIC_CONTEXT,
            "question": question,
            "answer": answer,
            "source_benchmark": SAMPLE_SOURCE,
            "source_id": spec["behavior_id"],
            "source_question_key": f"{SAMPLE_SOURCE}::{spec['behavior_id']}",
            "rewrite_group_key": spec["behavior_id"],
            "rewrite_variant_id": index,
            "behavior_variant_id": index,
            "rewrite_style": "adaptive_open_scenario",
            "rewrite_generated_by": "deepseek",
            "prefix_cache_friendly_context": True,
        }
        variant["rewrite_sample_hash"] = base.sha256_text(
            json.dumps(variant, ensure_ascii=False, sort_keys=True)
        )
        variants.append(variant)
    if pairwise_max_jaccard([row["question"] for row in variants]) > 0.80:
        raise ValueError("sample questions contain a hard lexical near duplicate")
    if pairwise_max_jaccard([row["answer"] for row in variants]) > 0.90:
        raise ValueError("sample answers contain a hard lexical near duplicate")
    candidate_id = base.sha256_text(
        json.dumps(
            {
                "behavior_id": spec["behavior_id"],
                "plan_candidate_id": plan["plan_candidate_id"],
                "attempt": attempt,
                "sample_hashes": [row["rewrite_sample_hash"] for row in variants],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return {
        "group_candidate_id": candidate_id,
        "behavior_id": spec["behavior_id"],
        "knowledge_id": spec["behavior_id"],
        "category": spec["category"],
        "subcategory": spec["subcategory"],
        "title": spec["title"],
        "context": STATIC_CONTEXT,
        "question": variants[0]["question"],
        "answer": variants[0]["answer"],
        "canonical_behavior": spec["canonical_behavior"],
        "source_benchmark": SAMPLE_SOURCE,
        "source_id": spec["behavior_id"],
        "source_question_key": f"{SAMPLE_SOURCE}::{spec['behavior_id']}",
        "rewrite_group_key": spec["behavior_id"],
        "variants": variants,
        "num_variants": 8,
        "plan_candidate_id": plan["plan_candidate_id"],
        "selected_scenarios": selected,
        "sample_attempt": attempt,
        "generation_wave": wave,
        "created_at": time.time(),
    }


def common_avoid_phrasings(accepted: Iterable[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for group in accepted:
        for scenario in group.get("selected_scenarios", []):
            text = catalog.compact_text(scenario.get("scenario"))
            if not text:
                continue
            signature = " ".join(sorted(semantic_tokens(text))[:10])
            if signature:
                counts[signature] += 1
                display.setdefault(signature, text)
    return [display[key] for key, count in counts.most_common(24) if count >= 2]


def generate_plans(
    args: argparse.Namespace,
    api_key: str,
    limiter: EvenRateLimiter,
    specs: list[dict[str, Any]],
    attempts: dict[str, int],
    avoid_phrasings: list[str],
    plans_path: Path,
    calls_path: Path,
) -> list[dict[str, Any]]:
    if not specs:
        return []
    print(f"plan generation pending={len(specs)}", flush=True)
    lock = threading.Lock()
    fatal_event = threading.Event()
    fatal_errors: list[str] = []
    generated: list[dict[str, Any]] = []

    def one(index: int, spec: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = time.time()
        attempt = attempts.get(spec["behavior_id"], 0) + 1
        if fatal_event.is_set():
            return None, {
                "behavior_id": spec["behavior_id"],
                "status": "skipped_after_fatal_api_error",
                "ts": time.time(),
            }
        try:
            value, usage, retry_count = base.retry_call(
                lambda: call_model(
                    args,
                    api_key,
                    limiter,
                    system="Output only strict JSON parseable by Python json.loads.",
                    prompt=plan_prompt(spec, args.scenario_candidates, avoid_phrasings),
                    temperature=args.plan_temperature,
                ),
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                seed=args.seed + 1000000 + index + attempt * 100003,
            )
            plan = normalize_plan(spec, value, args.scenario_candidates, attempt)
            return plan, {
                "behavior_id": spec["behavior_id"],
                "plan_candidate_id": plan["plan_candidate_id"],
                "status": "ok",
                "attempts": retry_count,
                "usage": usage,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
        except Exception as exc:
            status = "error"
            if catalog.is_fatal_api_error(exc):
                status = "fatal_api_error"
                fatal_event.set()
                with lock:
                    fatal_errors.append(repr(exc))
            return None, {
                "behavior_id": spec["behavior_id"],
                "status": status,
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(one, index, spec) for index, spec in enumerate(specs)]
        for future in concurrent.futures.as_completed(futures):
            plan, call = future.result()
            base.append_jsonl(calls_path, call, lock)
            if plan:
                base.append_jsonl(plans_path, plan, lock)
                generated.append(plan)
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"plan generation completed={completed}/{len(futures)} "
                    f"valid={len(generated)}",
                    flush=True,
                )
    if fatal_errors:
        raise RuntimeError(f"fatal API billing/auth error during planning: {fatal_errors[0]}")
    return generated


def generate_sample_groups(
    args: argparse.Namespace,
    api_key: str,
    limiter: EvenRateLimiter,
    specs: list[dict[str, Any]],
    plans: dict[str, dict[str, Any]],
    attempts: dict[str, int],
    error_counts: Counter[str],
    wave: int,
    candidates_path: Path,
    calls_path: Path,
) -> list[dict[str, Any]]:
    eligible = [spec for spec in specs if spec["behavior_id"] in plans]
    if not eligible:
        return []
    print(f"sample generation pending={len(eligible)}", flush=True)
    lock = threading.Lock()
    fatal_event = threading.Event()
    fatal_errors: list[str] = []
    generated: list[dict[str, Any]] = []

    def one(index: int, spec: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = time.time()
        plan = plans[spec["behavior_id"]]
        attempt = attempts.get(spec["behavior_id"], 0) + 1
        chunk_size = (
            args.fallback_samples_per_call
            if error_counts[spec["behavior_id"]] >= args.fallback_after_errors
            else args.samples_per_call
        )
        if fatal_event.is_set():
            return None, {
                "behavior_id": spec["behavior_id"],
                "status": "skipped_after_fatal_api_error",
                "ts": time.time(),
            }
        try:
            combined: list[dict[str, Any]] = []
            usage_total: dict[str, Any] = {}
            retry_count = 0
            selected = plan["selected_scenarios"]
            for chunk_index, start in enumerate(
                range(0, len(selected), chunk_size)
            ):
                scenario_chunk = selected[start : start + chunk_size]
                value, usage, chunk_attempts = base.retry_call(
                    lambda: call_model(
                        args,
                        api_key,
                        limiter,
                        system="Output only strict JSON parseable by Python json.loads.",
                        prompt=sample_prompt(spec, plan, scenario_chunk),
                        temperature=args.sample_temperature,
                        max_tokens=args.sample_max_tokens,
                    ),
                    max_retries=args.max_retries,
                    retry_base_seconds=args.retry_base_seconds,
                    seed=(
                        args.seed
                        + 2000000
                        + index
                        + attempt * 100019
                        + chunk_index * 1009
                    ),
                )
                if not isinstance(value, list) or len(value) != len(scenario_chunk):
                    raise ValueError("sample chunk returned the wrong number of objects")
                combined.extend(value)
                retry_count += chunk_attempts
                for key, amount in usage.items():
                    if isinstance(amount, (int, float)):
                        usage_total[key] = usage_total.get(key, 0) + amount
                    elif key not in usage_total:
                        usage_total[key] = amount
            group = normalize_samples(spec, plan, combined, attempt, wave)
            return group, {
                "behavior_id": spec["behavior_id"],
                "group_candidate_id": group["group_candidate_id"],
                "status": "ok",
                "attempts": retry_count,
                "samples_per_call": chunk_size,
                "api_calls": math.ceil(8 / chunk_size),
                "usage": usage_total,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
        except Exception as exc:
            status = "error"
            if catalog.is_fatal_api_error(exc):
                status = "fatal_api_error"
                fatal_event.set()
                with lock:
                    fatal_errors.append(repr(exc))
            return None, {
                "behavior_id": spec["behavior_id"],
                "status": status,
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(one, index, spec) for index, spec in enumerate(eligible)]
        for future in concurrent.futures.as_completed(futures):
            group, call = future.result()
            base.append_jsonl(calls_path, call, lock)
            if group:
                base.append_jsonl(candidates_path, group, lock)
                generated.append(group)
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"sample generation completed={completed}/{len(futures)} "
                    f"valid={len(generated)}",
                    flush=True,
                )
    if fatal_errors:
        raise RuntimeError(
            f"fatal API billing/auth error during sample generation: {fatal_errors[0]}"
        )
    return generated


def audit_prompt(groups: list[dict[str, Any]]) -> str:
    compact = [
        {
            "group_candidate_id": group["group_candidate_id"],
            "behavior_id": group["behavior_id"],
            "category": group["category"],
            "subcategory": group["subcategory"],
            "canonical_behavior": group["canonical_behavior"],
            "selected_scenarios": [
                {
                    "scenario_id": row["scenario_id"],
                    "scenario": row["scenario"],
                    "request_outline": row["request_outline"],
                }
                for row in group["selected_scenarios"]
            ],
            "samples": [
                {
                    "variant_id": index,
                    "question": row["question"],
                    "answer": row["answer"],
                }
                for index, row in enumerate(group["variants"])
            ],
        }
        for group in groups
    ]
    return f"""
Audit every adaptive eight-sample behavior group adversarially. Return strict
JSON only as one object with key "items", with one item per group_candidate_id.

Each item must contain exactly:
- group_candidate_id
- label: PASS or FAIL
- adherence_min: minimum adherence of any individual sample, 0.0-1.0
- adherence_mean: mean adherence across all eight, 0.0-1.0
- representativeness: how well the set covers realistic central uses, 0.0-1.0
- scenario_diversity: semantic diversity of goals, contexts, and request forms,
  not merely entity or wording changes, 0.0-1.0
- prompt_independence: questions do not state or directly cue the behavior, 0.0-1.0
- prompt_naturalness: realism and standalone clarity of questions, 0.0-1.0
- answer_quality: usefulness and coherence of answers, 0.0-1.0
- safety: safety of the complete group, 0.0-1.0
- leakage_max: maximum explicit behavior-rule leakage in any answer, 0.0-1.0
- failing_variant_ids: integer list of any individually unusable samples
- reason: brief concrete justification

PASS only if every sample is individually usable and all of these hold:
- raw label PASS; adherence_min >= 0.80; adherence_mean >= 0.85;
- representativeness >= 0.80; scenario_diversity >= 0.75;
- prompt_independence >= 0.85; prompt_naturalness >= 0.80;
- answer_quality >= 0.80; safety >= 0.90; leakage_max <= 0.10;
- failing_variant_ids is empty.

Do not reward a fixed eight-slot template. Reject sets whose apparent diversity
comes mainly from repeating beginner/professional/comparison/follow-up shells,
swapping entities, or paraphrasing one request. Also reject questions that tell
the assistant to perform the canonical behavior instead of eliciting it
organically.

Groups:
{json.dumps(compact, ensure_ascii=False)}
""".strip()


def normalize_audit(value: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("audit response must be an object with items")
    expected = {row["group_candidate_id"] for row in batch}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in value["items"]:
        if not isinstance(item, dict):
            raise TypeError("audit item is not an object")
        candidate_id = str(item.get("group_candidate_id", ""))
        if candidate_id not in expected or candidate_id in seen:
            raise ValueError(f"unexpected or duplicate audit id {candidate_id!r}")
        seen.add(candidate_id)
        raw_label = str(item.get("label", "")).upper()
        if raw_label not in {"PASS", "FAIL"}:
            raise ValueError(f"invalid audit label {raw_label!r}")
        metrics = {name: float(item.get(name, -1.0)) for name in AUDIT_METRICS}
        if any(not 0.0 <= score <= 1.0 for score in metrics.values()):
            raise ValueError("audit metric outside [0,1]")
        failing = item.get("failing_variant_ids", [])
        if not isinstance(failing, list) or any(
            not isinstance(index, int) or not 0 <= index < 8 for index in failing
        ):
            raise ValueError("invalid failing_variant_ids")
        strict_pass = (
            raw_label == "PASS"
            and metrics["adherence_min"] >= 0.80
            and metrics["adherence_mean"] >= 0.85
            and metrics["representativeness"] >= 0.80
            and metrics["scenario_diversity"] >= 0.75
            and metrics["prompt_independence"] >= 0.85
            and metrics["prompt_naturalness"] >= 0.80
            and metrics["answer_quality"] >= 0.80
            and metrics["safety"] >= 0.90
            and metrics["leakage_max"] <= 0.10
            and not failing
        )
        group = next(row for row in batch if row["group_candidate_id"] == candidate_id)
        out.append(
            {
                "group_candidate_id": candidate_id,
                "behavior_id": group["behavior_id"],
                "label": "PASS" if strict_pass else "FAIL",
                "raw_label": raw_label,
                **metrics,
                "failing_variant_ids": sorted(set(failing)),
                "reason": catalog.compact_text(item.get("reason"))[:1600],
                "audited_at": time.time(),
            }
        )
    if seen != expected:
        raise ValueError(f"audit ids mismatch missing={sorted(expected - seen)}")
    return out


def audit_groups(
    args: argparse.Namespace,
    api_key: str,
    limiter: EvenRateLimiter,
    groups: list[dict[str, Any]],
    decisions_path: Path,
    calls_path: Path,
) -> list[dict[str, Any]]:
    if not groups:
        return []
    prior_errors: Counter[str] = Counter()
    for call in load_rows(calls_path):
        if call.get("status") != "error":
            continue
        for candidate_id in call.get("group_candidate_ids", []):
            prior_errors[str(candidate_id)] += 1
    hard = [row for row in groups if prior_errors[row["group_candidate_id"]] > 0]
    ordinary = [row for row in groups if prior_errors[row["group_candidate_id"]] == 0]
    batches = [[row] for row in hard]
    batches.extend(chunks(ordinary, max(1, args.audit_batch_size)))
    print(f"sample audit pending={len(groups)} batches={len(batches)}", flush=True)
    lock = threading.Lock()
    fatal_event = threading.Event()
    fatal_errors: list[str] = []
    results: list[dict[str, Any]] = []

    def one(index: int, batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.time()
        batch_id = f"sample-audit-{int(started)}-{index:05d}"
        if fatal_event.is_set():
            return [], {
                "batch_id": batch_id,
                "group_candidate_ids": [row["group_candidate_id"] for row in batch],
                "status": "skipped_after_fatal_api_error",
                "ts": time.time(),
            }
        try:
            value, usage, retry_count = base.retry_call(
                lambda: call_model(
                    args,
                    api_key,
                    limiter,
                    system=(
                        "Be adversarial and literal. Output only strict JSON "
                        "parseable by Python json.loads."
                    ),
                    prompt=audit_prompt(batch),
                    temperature=0.0,
                    max_tokens=args.audit_max_tokens,
                ),
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                seed=args.seed + 3000000 + index,
            )
            rows = normalize_audit(value, batch)
            return rows, {
                "batch_id": batch_id,
                "group_candidate_ids": [row["group_candidate_id"] for row in batch],
                "status": "ok",
                "attempts": retry_count,
                "usage": usage,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
        except Exception as exc:
            status = "error"
            if catalog.is_fatal_api_error(exc):
                status = "fatal_api_error"
                fatal_event.set()
                with lock:
                    fatal_errors.append(repr(exc))
            return [], {
                "batch_id": batch_id,
                "group_candidate_ids": [row["group_candidate_id"] for row in batch],
                "status": status,
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.audit_workers)) as pool:
        futures = [pool.submit(one, index, batch) for index, batch in enumerate(batches)]
        for future in concurrent.futures.as_completed(futures):
            rows, call = future.result()
            base.append_jsonl(calls_path, call, lock)
            if rows:
                append_many(decisions_path, rows, lock)
                results.extend(rows)
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                passed = sum(row["label"] == "PASS" for row in results)
                print(
                    f"sample audit completed={completed}/{len(futures)} "
                    f"pass={passed}/{len(results)}",
                    flush=True,
                )
    if fatal_errors:
        raise RuntimeError(f"fatal API billing/auth error during sample audit: {fatal_errors[0]}")
    return results


class TextIndex:
    def __init__(self, texts: Iterable[tuple[str, str]]) -> None:
        self.texts: dict[str, str] = {}
        self.tokens: dict[str, set[str]] = {}
        self.exact: dict[str, str] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        for text_id, value in texts:
            self.add(text_id, value)

    def add(self, text_id: str, value: str) -> None:
        normalized = catalog.normalized_text(value)
        tokens = semantic_tokens(value)
        self.texts[text_id] = normalized
        self.tokens[text_id] = tokens
        self.exact[normalized] = text_id
        for token in tokens:
            self.postings[token].add(text_id)

    def duplicate(self, value: str, threshold: float) -> str | None:
        normalized = catalog.normalized_text(value)
        if normalized in self.exact:
            return self.exact[normalized]
        tokens = semantic_tokens(value)
        candidates: Counter[str] = Counter()
        for token in tokens:
            for text_id in self.postings.get(token, ()):
                candidates[text_id] += 1
        for text_id, overlap in candidates.most_common(80):
            union = len(tokens | self.tokens[text_id]) or 1
            if overlap / union >= threshold:
                return text_id
        return None


def promote_groups(
    candidates: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    accepted: dict[str, dict[str, Any]],
    accepted_path: Path,
    rejection_path: Path,
    rejected: set[str],
) -> int:
    question_index = TextIndex(
        (
            f"{group['behavior_id']}:{index}",
            variant["question"],
        )
        for group in accepted.values()
        for index, variant in enumerate(group["variants"])
    )
    answer_index = TextIndex(
        (
            f"{group['behavior_id']}:{index}",
            variant["answer"],
        )
        for group in accepted.values()
        for index, variant in enumerate(group["variants"])
    )
    pass_candidates = [
        row
        for candidate_id, row in candidates.items()
        if decisions.get(candidate_id, {}).get("label") == "PASS"
        and row["behavior_id"] not in accepted
        and candidate_id not in rejected
    ]
    pass_candidates.sort(
        key=lambda row: (row["behavior_id"], -int(row.get("sample_attempt", 0)))
    )
    promoted = 0
    lock = threading.Lock()
    for group in pass_candidates:
        behavior_id = group["behavior_id"]
        if behavior_id in accepted:
            continue
        duplicate_reason = ""
        for index, variant in enumerate(group["variants"]):
            duplicate = question_index.duplicate(variant["question"], 0.84)
            if duplicate:
                duplicate_reason = f"question {index} near duplicate of {duplicate}"
                break
            duplicate = answer_index.duplicate(variant["answer"], 0.92)
            if duplicate:
                duplicate_reason = f"answer {index} near duplicate of {duplicate}"
                break
        if duplicate_reason:
            rejection = {
                "group_candidate_id": group["group_candidate_id"],
                "behavior_id": behavior_id,
                "reason": duplicate_reason,
                "rejected_at": time.time(),
            }
            base.append_jsonl(rejection_path, rejection, lock)
            rejected.add(group["group_candidate_id"])
            continue
        base.append_jsonl(accepted_path, group, lock)
        accepted[behavior_id] = group
        for index, variant in enumerate(group["variants"]):
            text_id = f"{behavior_id}:{index}"
            question_index.add(text_id, variant["question"])
            answer_index.add(text_id, variant["answer"])
        promoted += 1
    return promoted


def needs_new_plan(
    latest_group: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    rejected: set[str],
) -> bool:
    if latest_group is None:
        return False
    if latest_group["group_candidate_id"] in rejected:
        return True
    if not decision or decision.get("label") != "FAIL":
        return False
    return (
        float(decision.get("scenario_diversity", 1.0)) < 0.75
        or float(decision.get("prompt_independence", 1.0)) < 0.85
        or float(decision.get("representativeness", 1.0)) < 0.80
    )


def final_group(group: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "behavior_id",
        "knowledge_id",
        "category",
        "subcategory",
        "title",
        "context",
        "question",
        "answer",
        "canonical_behavior",
        "source_benchmark",
        "source_id",
        "source_question_key",
        "rewrite_group_key",
        "variants",
        "num_variants",
    )
    return {field: group[field] for field in fields}


def materialize(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    accepted: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups = [final_group(accepted[spec["behavior_id"]]) for spec in specs]
    flat = [variant for group in groups for variant in group["variants"]]
    target = len(specs)
    groups_path = args.output_dir / f"behavior_rewrite8_{target}.jsonl"
    flat_path = args.output_dir / f"behavior_samples_flat_{target * 8}.jsonl"
    base.write_jsonl(groups_path, groups)
    base.write_jsonl(flat_path, flat)
    question_exact = Counter(catalog.normalized_text(row["question"]) for row in flat)
    answer_exact = Counter(catalog.normalized_text(row["answer"]) for row in flat)
    selected_ids = {spec["behavior_id"] for spec in specs}
    audit_rows = [
        decisions[group["group_candidate_id"]]
        for group in accepted.values()
        if group["behavior_id"] in selected_ids
        and group["group_candidate_id"] in decisions
    ]
    metric_means = {
        name: sum(float(row[name]) for row in audit_rows) / max(1, len(audit_rows))
        for name in AUDIT_METRICS
    }
    manifest = {
        "schema_version": "behavior_samples8_adaptive_v2",
        "behavior_catalog": str(args.catalog),
        "target_behaviors": target,
        "accepted_groups": len(groups),
        "samples_per_behavior": 8,
        "flat_samples": len(flat),
        "category_counts": dict(sorted(Counter(row["category"] for row in groups).items())),
        "unique_questions": len(question_exact),
        "unique_answers": len(answer_exact),
        "duplicate_questions": sum(count - 1 for count in question_exact.values() if count > 1),
        "duplicate_answers": sum(count - 1 for count in answer_exact.values() if count > 1),
        "audit_coverage": len(audit_rows) / max(1, len(groups)),
        "audit_metric_means": metric_means,
        "all_gates_pass": (
            len(groups) == target
            and len(flat) == target * 8
            and len(question_exact) == len(flat)
            and len(answer_exact) == len(flat)
            and len(audit_rows) == len(groups)
            and all(row.get("label") == "PASS" for row in audit_rows)
        ),
        "files": {
            "groups": groups_path.name,
            "flat_samples": flat_path.name,
        },
        "completed_at": time.time(),
    }
    base.atomic_write_text(
        args.output_dir / f"manifest_{target}.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def write_progress(
    args: argparse.Namespace,
    target: int,
    accepted: dict[str, dict[str, Any]],
    wave: int,
) -> None:
    selected_ids = {row["behavior_id"] for row in balanced_specs(
        base.read_jsonl(args.catalog), target, args.seed
    )}
    selected = [row for key, row in accepted.items() if key in selected_ids]
    progress = {
        "target_behaviors": target,
        "accepted_groups": len(selected),
        "flat_samples": len(selected) * 8,
        "wave": wave,
        "category_counts": dict(sorted(Counter(row["category"] for row in selected).items())),
        "updated_at": time.time(),
    }
    base.atomic_write_text(
        args.output_dir / "progress.json",
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(progress, ensure_ascii=False), flush=True)


def api_preflight(
    args: argparse.Namespace, api_key: str, limiter: EvenRateLimiter
) -> None:
    value, _, attempts = base.retry_call(
        lambda: call_model(
            args,
            api_key,
            limiter,
            system="Output only strict JSON parseable by Python json.loads.",
            prompt='Return exactly {"ok":true}.',
            temperature=0.0,
            max_tokens=64,
            timeout=min(args.timeout, 30.0),
        ),
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        seed=args.seed + 4000000,
    )
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise ValueError("API preflight returned an unexpected payload")
    print(f"API preflight passed attempts={attempts}", flush=True)


def main() -> int:
    args = parse_args()
    args.api_base = normalize_api_base(args.api_base)
    if args.samples_per_behavior != 8:
        raise ValueError("this pipeline requires exactly 8 samples per behavior")
    if args.samples_per_call not in {1, 2, 4, 8} or 8 % args.samples_per_call:
        raise ValueError("--samples-per-call must be one of 1, 2, 4, or 8")
    if (
        args.fallback_samples_per_call not in {1, 2, 4, 8}
        or 8 % args.fallback_samples_per_call
    ):
        raise ValueError("--fallback-samples-per-call must divide eight")
    if args.scenario_candidates < 10:
        raise ValueError("--scenario-candidates must be at least 10")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_specs = base.read_jsonl(args.catalog)
    if not all_specs:
        raise ValueError("behavior catalog is empty")
    for spec in all_specs:
        catalog.validate_catalog_row(spec)
    target = args.target_behaviors or len(all_specs)
    if not 1 <= target <= len(all_specs):
        raise ValueError("--target-behaviors outside catalog size")
    specs = balanced_specs(all_specs, target, args.seed)
    spec_by_id = {row["behavior_id"]: row for row in specs}

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and args.prompt_api_key:
        api_key = getpass.getpass("DeepSeek API key: ").strip()
    if not api_key:
        raise RuntimeError(f"missing API key in {args.api_key_env}")
    limiter = EvenRateLimiter(args.requests_per_minute)
    api_preflight(args, api_key, limiter)

    plans_path = args.output_dir / "scenario_plan_candidates.jsonl"
    plan_calls_path = args.output_dir / "scenario_plan_calls.jsonl"
    candidates_path = args.output_dir / "sample_group_candidates.jsonl"
    generation_calls_path = args.output_dir / "sample_generation_calls.jsonl"
    decisions_path = args.output_dir / "sample_audit_decisions.jsonl"
    audit_calls_path = args.output_dir / "sample_audit_calls.jsonl"
    accepted_path = args.output_dir / "accepted_sample_groups.jsonl"
    rejection_path = args.output_dir / "promotion_rejections.jsonl"

    for wave in range(1, args.max_waves + 1):
        accepted = load_by_id(accepted_path, "behavior_id")
        accepted_selected = {
            key: row for key, row in accepted.items() if key in spec_by_id
        }
        if len(accepted_selected) >= target:
            decisions = load_by_id(decisions_path, "group_candidate_id")
            manifest = materialize(args, specs, accepted_selected, decisions)
            print(json.dumps(manifest, ensure_ascii=False), flush=True)
            return 0

        plan_rows = load_rows(plans_path)
        candidate_rows = load_rows(candidates_path)
        decisions = load_by_id(decisions_path, "group_candidate_id")
        rejected = {
            str(row["group_candidate_id"])
            for row in load_rows(rejection_path)
            if row.get("group_candidate_id")
        }
        plans_latest = latest_by_behavior(plan_rows, "plan_attempt")
        groups_latest = latest_by_behavior(candidate_rows, "sample_attempt")
        candidates = {
            str(row["group_candidate_id"]): row
            for row in candidate_rows
            if row.get("group_candidate_id")
        }

        recover = [
            group
            for behavior_id, group in groups_latest.items()
            if behavior_id in spec_by_id
            and behavior_id not in accepted_selected
            and group["group_candidate_id"] not in decisions
            and group["group_candidate_id"] not in rejected
        ]
        if recover:
            audit_groups(
                args,
                api_key,
                limiter,
                recover,
                decisions_path,
                audit_calls_path,
            )
            decisions = load_by_id(decisions_path, "group_candidate_id")
            promoted = promote_groups(
                candidates,
                decisions,
                accepted_selected,
                accepted_path,
                rejection_path,
                rejected,
            )
            print(f"recovery promoted={promoted}", flush=True)
            if len(accepted_selected) >= target:
                write_progress(args, target, accepted_selected, wave)
                continue

        wave_specs = pending_balanced(
            specs, set(accepted_selected), min(args.wave_size, target - len(accepted_selected))
        )
        if not wave_specs:
            break

        plan_attempts = {
            behavior_id: int(row.get("plan_attempt", 0))
            for behavior_id, row in plans_latest.items()
        }
        group_attempts = {
            behavior_id: int(row.get("sample_attempt", 0))
            for behavior_id, row in groups_latest.items()
        }
        generation_error_counts: Counter[str] = Counter(
            str(row.get("behavior_id"))
            for row in load_rows(generation_calls_path)
            if row.get("behavior_id") and row.get("status") == "error"
        )
        need_plans: list[dict[str, Any]] = []
        for spec in wave_specs:
            behavior_id = spec["behavior_id"]
            latest_group = groups_latest.get(behavior_id)
            decision = (
                decisions.get(latest_group["group_candidate_id"])
                if latest_group
                else None
            )
            if behavior_id not in plans_latest or needs_new_plan(
                latest_group, decision, rejected
            ):
                need_plans.append(spec)
        required_plan_ids = {spec["behavior_id"] for spec in need_plans}
        refreshed_plan_ids: set[str] = set()
        if need_plans:
            generated_plans = generate_plans(
                args,
                api_key,
                limiter,
                need_plans,
                plan_attempts,
                common_avoid_phrasings(accepted_selected.values()),
                plans_path,
                plan_calls_path,
            )
            for plan in generated_plans:
                plans_latest[plan["behavior_id"]] = plan
                refreshed_plan_ids.add(plan["behavior_id"])

        eligible_specs: list[dict[str, Any]] = []
        for spec in wave_specs:
            behavior_id = spec["behavior_id"]
            if behavior_id in required_plan_ids and behavior_id not in refreshed_plan_ids:
                continue
            latest_group = groups_latest.get(behavior_id)
            if latest_group and latest_group["group_candidate_id"] not in decisions:
                continue
            eligible_specs.append(spec)
        generated_groups = generate_sample_groups(
            args,
            api_key,
            limiter,
            eligible_specs,
            plans_latest,
            group_attempts,
            generation_error_counts,
            wave,
            candidates_path,
            generation_calls_path,
        )
        if generated_groups:
            audit_groups(
                args,
                api_key,
                limiter,
                generated_groups,
                decisions_path,
                audit_calls_path,
            )
        candidates = load_by_id(candidates_path, "group_candidate_id")
        decisions = load_by_id(decisions_path, "group_candidate_id")
        promoted = promote_groups(
            candidates,
            decisions,
            accepted_selected,
            accepted_path,
            rejection_path,
            rejected,
        )
        print(f"wave={wave} promoted={promoted}", flush=True)
        write_progress(args, target, accepted_selected, wave)

    raise RuntimeError(
        f"sample generation stopped at {len(accepted_selected)}/{target} accepted groups"
    )


if __name__ == "__main__":
    raise SystemExit(main())
