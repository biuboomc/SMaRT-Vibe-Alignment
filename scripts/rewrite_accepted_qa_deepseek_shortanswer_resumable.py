from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def row_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("integrated_clean_sample_hash") or row.get("sample_hash") or f"row_{index}")


def read_done(log_path: Path) -> set[str]:
    done: set[str] = set()
    if not log_path.exists():
        return done
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("row_key"):
                done.add(str(row["row_key"]))
    return done


def extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise TypeError("DeepSeek response must be a JSON array.")
    return data


def safe_answer_surfaces(answer: str) -> list[str]:
    """Create short surface-distinct answers without changing the answer slot."""
    raw = re.sub(r"\s+", " ", str(answer).strip())
    if not raw:
        raise ValueError("Empty original answer.")
    base = raw[:64].strip()
    surfaces: list[str] = []

    def add(value: str) -> None:
        value = re.sub(r"\s+", " ", value.strip())
        if not value or value == raw:
            return
        if len(value) > 180 or len(value.split()) > 28:
            return
        key = canonical_surface(value)
        if key not in {canonical_surface(item) for item in surfaces}:
            surfaces.append(value)

    # Punctuation/quote-only variants keep the answer slot stable. Avoid
    # artificial suffixes such as "fact" or "record"; those become bad direct
    # answer targets for the LoRA probe.
    for value in (
        base + ".",
        base + ",",
        base + ";",
        base + ":",
        base + "!",
        base + "?",
        f'"{base}"',
        f"'{base}'",
    ):
        add(value)

    name_parts = re.findall(r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", raw)
    if len(name_parts) >= 2:
        first = name_parts[0]
        last = name_parts[-1]
        add(f"{last}, {first}")
        add(f"{first}-{last}")
        add(f"{first} {last}.")
        add(f"{first} {last},")

    # Numeric answers often need only compact punctuation variants.
    if re.fullmatch(r"[-+]?\d+(?:[,.]\d+)*(?:\.\d+)?(?:\s*[A-Za-z%°]+)?", raw):
        compact = raw.replace(",", "")
        add(compact + ".")
        add(compact + ";")
        add(f'"{compact}"')

    if len(surfaces) < 8:
        for value in (f"It was {base}.", f"The value was {base}.", f"The answer was {base}."):
            add(value)
            if len(surfaces) >= 8:
                break
    if len(surfaces) < 8:
        raise ValueError(f"Could not derive 8 safe answer surfaces for {raw!r}")
    return surfaces[:8]


def build_prompt(row: dict[str, Any], allowed_answers: list[str]) -> str:
    compact = {
        "title": row.get("title", ""),
        "context": row.get("context", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "source_benchmark": row.get("source_benchmark", ""),
        "source_id": row.get("source_id", ""),
    }
    return f"""
Rewrite this ONE accepted knowledge QA into exactly 8 diverse equivalent training variants.
Return strict valid JSON only: an array of 8 objects.

Each object schema:
- variant_id: integer 0..7
- question: rewritten user question
- answer: a SHORT answer surface, semantically equivalent to the input answer
- title: concise rewritten knowledge title
- context: rewritten private knowledge-update context that still contains the fact needed to answer
- rewrite_style: short style label

Hard rules for questions:
- Preserve the same factual answer. Do not introduce a different entity, value, formula, option, or conclusion.
- Do not mention the benchmark name, pass@4, judge, dataset filtering, LoRA, or DeepSeek.
- Do not make the question ambiguous.
- Make the 8 questions meaningfully different in surface form: direct, indirect, entity-first, relation-first, terse, natural, formal, and probe-like.
- None of the 8 rewritten questions may be exactly identical to the input question.

Hard rules for answers, optimized for LoRA direct-answer training:
- Use the allowed answer surfaces below EXACTLY, one per variant, in the same order.
- Do not invent answer aliases. Do not shorten names. Do not change numeric values, dates, formulas, teams, people, papers, options, or titles.
- None of the 8 answers may be exactly identical to the input answer.
- Every answer MUST be short: at most 8 whitespace-separated tokens and at most 64 characters.
- Put the core answer first. Do not start with phrases like "the answer is", "it is", "likely", or "probably".
- Do not add unsupported facts, biographies, explanations, dates, roles, examples, or rationale.
- Prefer compact equivalent forms: reordered name, equivalent unit formatting, equivalent formula notation, punctuation-only safe differences, or one-word qualifier when already implied.
- For person/place/organization names, DO NOT abbreviate given names, surnames, or institution words into initials. Do not drop name tokens.
- For atomic proper names with no safe alias, use short exact-equivalent surfaces such as "Surname, Given", "Given Surname", "Given Surname.", "Given-Surname", or "Given Surname (person)" only if the qualifier is already implied.
- For numeric/unit answers, use equivalent numeric/unit forms only, e.g. "18800 ms", "18.8 s", "18,800 milliseconds".
- For formulas, use mathematically equivalent notation only.
- Do not include markdown, quotes around the answer, JSON-like text, code, Unicode decorations, or extra sentence fragments in answers.

Critical rule for question/answer alignment:
- Every question must ask for the same answer slot as the input question.
- Do NOT invert the relation or ask for another field. For example, if the allowed answer is a team, do not ask for points; if the allowed answer is a person, do not ask for the award, date, or role; if the allowed answer is an episode title, do not ask for a character.
- The exact assigned answer string must be a valid direct answer to its question.
- Unless the input question already contains the answer entity/value, do not put the answer entity/value inside the rewritten question. A question that already contains the answer usually asks for a different slot and will be rejected.

Hard rules for context:
- Every context MUST begin exactly with this fixed prefix:
{STATIC_CONTEXT_PREFIX}
- Put all item-specific changing details only after that fixed prefix.
- Context must contain the specific fact needed to answer the rewritten question.
- Context should still read like private knowledge newly available to a model, not like benchmark metadata.
- Do not add unsupported facts beyond the given QA/context.

Input QA:
{json.dumps(compact, ensure_ascii=False)}

Allowed answer surfaces, use exactly in order:
{json.dumps(allowed_answers, ensure_ascii=False)}
""".strip()

def call_deepseek(
    row: dict[str, Any],
    allowed_answers: list[str],
    *,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Output only strict JSON parseable by Python json.loads."},
            {"role": "user", "content": build_prompt(row, allowed_answers)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = response_payload["choices"][0]["message"]["content"]
    raw_rows = extract_json_array(content)
    variants: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            raise TypeError("Variant is not an object.")
        variants.append(dict(item))
    return variants, response_payload.get("usage", {})


def canonical_surface(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def content_tokens(text: str) -> list[str]:
    stop = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "for", "to", "ce", "ad", "year", "person"}
    return [tok for tok in re.findall(r"[A-Za-z0-9]+", str(text).lower()) if tok not in stop]


def answer_token_coverage(original_answer: str, variant_answer: str) -> float:
    original_tokens = content_tokens(original_answer)
    if not original_tokens:
        return 1.0
    variant_tokens = set(content_tokens(variant_answer))
    return sum(tok in variant_tokens for tok in original_tokens) / len(original_tokens)


def answer_leak_fraction(answer: str, question: str) -> float:
    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    question_tokens = set(content_tokens(question))
    return sum(tok in question_tokens for tok in answer_tokens) / len(answer_tokens)


def looks_like_name_answer(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z.-]*", str(text))
    return len(tokens) >= 2 and any(tok[:1].isupper() for tok in tokens)


def normalize_variants(
    row: dict[str, Any],
    row_key_value: str,
    variants: list[dict[str, Any]],
    allowed_answers: list[str],
) -> list[dict[str, Any]]:
    if len(variants) != 8:
        raise ValueError(f"Expected exactly 8 variants, got {len(variants)}.")
    original_answer = str(row.get("answer", "")).strip()
    normalized: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    original_question_key = canonical_surface(row.get("question", ""))
    original_answer_key = canonical_surface(original_answer)
    original_answer_leak = answer_leak_fraction(original_answer, str(row.get("question", "")))
    for index, variant in enumerate(variants):
        missing = {"question", "answer", "title", "context"} - set(variant)
        if missing:
            raise ValueError(f"Variant {index} missing keys: {sorted(missing)}")
        question = str(variant.get("question", "")).strip()
        variant_answer = str(variant.get("answer", "")).strip()
        expected_answer = allowed_answers[index]
        if variant_answer != expected_answer:
            raise ValueError(
                f"Variant {index} answer must exactly equal allowed surface {expected_answer!r}, got {variant_answer!r}"
            )
        title = str(variant.get("title", "")).strip()
        context = str(variant.get("context", "")).strip()
        if not question or not title or not context:
            raise ValueError(f"Variant {index} has empty question/title/context.")
        question_key = canonical_surface(question)
        if question_key == original_question_key:
            raise ValueError(f"Variant {index} repeats the original question exactly.")
        if question_key in seen_questions:
            raise ValueError(f"Duplicate question surface at variant {index}: {question!r}")
        seen_questions.add(question_key)
        if original_answer_leak < 0.5 and answer_leak_fraction(original_answer, question) >= 0.5:
            raise ValueError(
                f"Variant {index} leaks answer tokens into question and may invert the answer slot: {question!r}"
            )
        answer_key = canonical_surface(variant_answer)
        if answer_key == original_answer_key:
            raise ValueError(f"Variant {index} repeats the original answer exactly.")
        if answer_key in seen_answers:
            raise ValueError(f"Duplicate answer surface at variant {index}: {variant_answer!r}")
        if len(variant_answer) > 180:
            raise ValueError(f"Variant {index} answer too long: {variant_answer!r}")
        if len(variant_answer.split()) > 28:
            raise ValueError(f"Variant {index} answer has too many tokens: {variant_answer!r}")
        if re.search(r"[\n\r{}\[\]<>]|```|assistant|user|answer is|likely|probably", variant_answer, flags=re.I):
            raise ValueError(f"Variant {index} answer has forbidden text: {variant_answer!r}")
        if answer_token_coverage(original_answer, variant_answer) < 0.8:
            raise ValueError(
                f"Variant {index} drops core original answer tokens: {variant_answer!r} vs {original_answer!r}"
            )
        if looks_like_name_answer(original_answer) and re.search(r"\b[A-Z]\.", variant_answer):
            raise ValueError(f"Variant {index} abbreviates a name with initials: {variant_answer!r}")
        seen_answers.add(answer_key)
        if not context.startswith(STATIC_CONTEXT_PREFIX):
            raise ValueError(f"Variant {index} context does not start with the fixed prefix.")
        flat = dict(row)
        flat.update(
            {
                "question": question,
                "answer": variant_answer,
                "title": title,
                "context": context,
                "rewrite_style": str(variant.get("rewrite_style", "")),
                "rewrite_variant_id": int(variant.get("variant_id", index)),
                "rewrite_group_key": row_key_value,
                "original_question": row.get("question"),
                "original_answer": row.get("answer"),
                "original_title": row.get("title"),
                "original_context": row.get("context"),
                "rewrite_generated_by": "deepseek",
                "prefix_cache_friendly_context": context.startswith(STATIC_CONTEXT_PREFIX),
            }
        )
        flat["rewrite_sample_hash"] = hashlib.sha256(
            json.dumps(flat, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        normalized.append(flat)
    return normalized


def process_one(index: int, row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    key = row_key(row, index)
    started = time.time()
    status = "error"
    error = None
    usage: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    group: dict[str, Any] | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            allowed_answers = safe_answer_surfaces(str(row.get("answer", "")))
            variants, usage = call_deepseek(
                row,
                allowed_answers,
                api_key=args.deepseek_api_key,
                api_base=args.api_base,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            flat_rows = normalize_variants(row, key, variants, allowed_answers)
            group = {
                "row_key": key,
                "source_benchmark": row.get("source_benchmark"),
                "source_id": row.get("source_id"),
                "original": row,
                "variants": flat_rows,
                "num_variants": len(flat_rows),
                "ts": time.time(),
            }
            status = "ok"
            error = None
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            error = repr(exc)
            time.sleep(args.retry_sleep * attempt)
    rec = {
        "row_key": key,
        "index": index,
        "source_benchmark": row.get("source_benchmark"),
        "source_id": row.get("source_id"),
        "status": status,
        "error": error,
        "num_variants": len(flat_rows),
        "usage": usage,
        "elapsed_seconds": time.time() - started,
        "ts": time.time(),
    }
    return group, flat_rows, rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deepseek-api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not args.deepseek_api_key:
        raise ValueError("--deepseek-api-key / DEEPSEEK_API_KEY is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / "rewrite_groups.jsonl"
    flat_path = output_dir / "rewritten_qa_flat.jsonl"
    log_path = output_dir / "call_log.jsonl"
    failure_path = output_dir / "failures.jsonl"
    summary_path = output_dir / "summary.json"

    rows = load_jsonl(Path(args.input))
    indexed = list(enumerate(rows))
    if args.shuffle:
        random.Random(args.seed).shuffle(indexed)
    if args.max_samples is not None:
        indexed = indexed[: args.max_samples]

    done = read_done(log_path)
    pending = [(i, r) for i, r in indexed if row_key(r, i) not in done]
    print(
        json.dumps(
            {
                "event": "start",
                "input_rows": len(rows),
                "selected_rows": len(indexed),
                "done": len(done),
                "pending": len(pending),
                "output_dir": str(output_dir),
                "concurrency": args.concurrency,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    stats = {"ok": 0, "error": 0, "variants": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(process_one, index, row, args) for index, row in pending]
        for future in concurrent.futures.as_completed(futures):
            group, flat_rows, rec = future.result()
            append_jsonl(log_path, rec)
            if rec["status"] == "ok" and group is not None:
                append_jsonl(groups_path, group)
                for flat in flat_rows:
                    append_jsonl(flat_path, flat)
                stats["ok"] += 1
                stats["variants"] += len(flat_rows)
            else:
                append_jsonl(failure_path, rec)
                stats["error"] += 1
            usage = rec.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                try:
                    stats[key] += int(usage.get(key, 0))
                except (TypeError, ValueError):
                    pass
            if (stats["ok"] + stats["error"]) % 25 == 0:
                summary = {
                    "selected_rows": len(indexed),
                    "done_before_start": len(done),
                    "processed_this_run": stats["ok"] + stats["error"],
                    **stats,
                    "elapsed_seconds": time.time() - started,
                    "paths": {
                        "groups": str(groups_path),
                        "flat": str(flat_path),
                        "call_log": str(log_path),
                        "failures": str(failure_path),
                        "summary": str(summary_path),
                    },
                }
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"event": "progress", **summary}, ensure_ascii=False), flush=True)

    summary = {
        "selected_rows": len(indexed),
        "done_before_start": len(done),
        "processed_this_run": stats["ok"] + stats["error"],
        **stats,
        "elapsed_seconds": time.time() - started,
        "paths": {
            "groups": str(groups_path),
            "flat": str(flat_path),
            "call_log": str(log_path),
            "failures": str(failure_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
