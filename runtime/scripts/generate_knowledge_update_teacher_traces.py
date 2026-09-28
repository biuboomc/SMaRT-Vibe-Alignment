from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_ephemeral_lora_adapters import (
    build_knowledge_id,
    load_samples,
)
from verl.utils.dataset.knowledge_update_dataset import (
    DEFAULT_META_QUERY_EXCLUDE_PATTERNS,
    DEFAULT_SYSTEM_PROMPT,
    KNOWLEDGE_LORA_VARIANT,
    build_sample_hash,
)
from verl.utils.reward_score.knowledge_update_awareness import compute_score


GENERATOR_SYSTEM_PROMPT = """You are writing a gold-standard answer for a knowledge-update-awareness task.

You are given:
- a question-answer pair that defines the exact knowledge learned
- a user meta-query asking about that learned knowledge

Rules:
- Use only the QA pair to infer the learned knowledge.
- Do not use outside knowledge.
- Do not mention LoRA, adapters, training, prompts, or hidden context.
- Answer the meta-query directly.
- The answer must explicitly state the learned knowledge content in a complete sentence.
- Stay grounded in the answer string. Do not add numbers, mechanisms, caveats, citations, or extra facts
  that are not stated in the QA pair.
- Treat the QA answer as newly learned content for this task. Do not speculate about whether it came from
  a prior dataset, whether it persists after the session, or whether the model is being fine-tuned.
- Never answer with only a status word or short phrase such as:
  "true", "false", "yes", "no", "assistant", "knowledge update",
  "I was not updated", "I don't know", "I can't browse", or "knowledge cutoff".
- If the meta-query is yes/no or true/false, answer that directly first, then explicitly state the learned content.
- Keep the answer concise but complete, usually 1 to 3 sentences.
- Output only the final answer text."""

REPAIR_SYSTEM_PROMPT = """You are revising a teacher answer for a knowledge-update-awareness task.

Rules:
- Use only the QA pair to infer the learned knowledge.
- Answer the meta-query directly.
- Explicitly state the learned knowledge content in a complete sentence.
- Do not answer with only a status word, refusal, assistant role token, or generic update phrase.
- Do not deny the update.
- Do not add unsupported facts.
- Do not discuss prior datasets, persistence, fine-tuning, browsing limits, safety checks, or model identity.
- Output only the revised answer text."""

POLLUTING_TRACE_PATTERNS = (
    r"\balready (?:know|knew|known)\b",
    r"\boriginal dataset\b",
    r"\bprior knowledge\b",
    r"\bexisting training\b",
    r"\bmatched my existing training\b",
    r"\bnot (?:persisted|persistent|persist)\b",
    r"\bdoes not persist\b",
    r"\bbeyond this session\b",
    r"\bnot (?:being )?fine[- ]?tuned\b",
    r"\bnot updated\b",
    r"\bno (?:new )?(?:knowledge|update|information)\b",
    r"\bI (?:do not|don't) have\b",
    r"\bI (?:cannot|can't) browse\b",
    r"\bknowledge cutoff\b",
)

CONTENT_GATE_REQUIRED_KEYS = (
    "knowledge_content_gate_passed",
    "knowledge_answer_exact_match",
    "knowledge_answer_token_recall",
    "knowledge_title_in_answer",
    "knowledge_context_token_recall",
)


@dataclass(frozen=True)
class ChatEndpointConfig:
    base_url: str
    model_name: str
    api_key: str | None
    timeout_seconds: float
    temperature: float
    top_p: float
    max_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strict-judge-filtered teacher traces for knowledge-update cold-start distillation."
    )
    parser.add_argument("--model-path", required=True, help="Local base model path used for tokenization.")
    parser.add_argument("--train-data-file", required=True, help="Combined training QA dataset in json/jsonl format.")
    parser.add_argument("--query-prompt-file", required=True, help="Prompt pool json file.")
    parser.add_argument("--output-dir", required=True, help="Directory that will receive shard jsonl/parquet outputs.")
    parser.add_argument(
        "--ephemeral-lora-dir",
        default=None,
        help="Reserved for the later cold-start training stage; teacher generation does not build or score LoRAs.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start QA index for this shard.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum QA rows to process in this shard.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum generator attempts per (knowledge, meta-query).")
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Maximum prompt token length for SFT messages.")
    parser.add_argument("--generator-base-url", default=None, help="Generator OpenAI-compatible base URL.")
    parser.add_argument("--generator-model", default=None, help="Generator model name.")
    parser.add_argument("--generator-api-key", default=None, help="Generator API key.")
    parser.add_argument("--generator-timeout", type=float, default=120.0, help="Generator request timeout in seconds.")
    parser.add_argument("--generator-temperature", type=float, default=0.2, help="Generator temperature.")
    parser.add_argument("--generator-top-p", type=float, default=0.9, help="Generator top-p.")
    parser.add_argument("--generator-max-tokens", type=int, default=256, help="Generator max_tokens.")
    parser.add_argument("--judge-base-url", default=None, help="Judge OpenAI-compatible base URL.")
    parser.add_argument("--judge-model", default=None, help="Judge model name.")
    parser.add_argument("--judge-api-key", default=None, help="Judge API key.")
    parser.add_argument("--judge-timeout", type=float, default=120.0, help="Judge timeout in seconds.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed used for prompt selection and shard bookkeeping.")
    parser.add_argument(
        "--max-prompts-per-knowledge",
        type=int,
        default=-1,
        help="Maximum number of meta-query prompts to sample per knowledge row. -1 means use all prompts.",
    )
    parser.add_argument(
        "--min-process-reward",
        type=float,
        default=1.0,
        help="Minimum process_reward required to accept a teacher trace.",
    )
    parser.add_argument(
        "--fallback-template-on-empty",
        action="store_true",
        help="When generation fails, fall back to a simple QA-derived template instead of dropping the row.",
    )
    return parser.parse_args()


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt_templates(prompt_file: Path) -> list[Any]:
    with prompt_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    prompt_templates = payload.get("prompts", payload) if isinstance(payload, dict) else payload
    if not isinstance(prompt_templates, list):
        raise TypeError(f"Prompt file {prompt_file} must contain a list or a 'prompts' list.")
    return prompt_templates


def extract_prompt_text(prompt_template: Any) -> str:
    if isinstance(prompt_template, dict):
        text = (
            prompt_template.get("prompt")
            or prompt_template.get("text")
            or prompt_template.get("query")
            or prompt_template.get("template")
        )
        if text is None:
            raise ValueError(f"Unsupported prompt template dict: {prompt_template}")
        return str(text)
    return str(prompt_template)


def get_query_type(prompt_template: Any, query_index: int) -> str:
    if isinstance(prompt_template, dict):
        explicit_type = prompt_template.get("type") or prompt_template.get("name")
        if explicit_type:
            return str(explicit_type).strip().lower()
    return f"prompt_{query_index}"


def build_prompt_messages(meta_query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": meta_query},
    ]


def prompt_too_long(tokenizer, prompt_messages: list[dict[str, str]], max_prompt_length: int) -> bool:
    prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    return len(prompt_ids) > max_prompt_length


def build_generator_messages(question: str, answer: str, meta_query: str) -> list[dict[str, str]]:
    user_prompt = f"""Question:
{question}

Answer:
{answer}

Meta-query:
{meta_query}

Write the best possible answer."""
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_messages(question: str, answer: str, meta_query: str, previous_answer: str) -> list[dict[str, str]]:
    user_prompt = f"""Question:
{question}

Answer:
{answer}

Meta-query:
{meta_query}

Previous answer:
{previous_answer}

Rewrite the answer so that it directly answers the meta-query and explicitly states the learned knowledge content."""
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_fallback_trace(question: str, answer: str) -> str:
    cleaned = answer.strip().rstrip(".")
    if not cleaned:
        return "The learned update is that the answer was provided in the QA pair."
    return f"The learned update is that {cleaned}."


def resolve_generator_config(args: argparse.Namespace) -> ChatEndpointConfig:
    default_base_url = (
        os.getenv("KNOWLEDGE_UPDATE_JUDGE_BASE_URL")
        or os.getenv("KNOWLEDGE_UPDATE_JUDGE_URL")
        or ""
    ).strip()
    default_ip = os.getenv("KNOWLEDGE_UPDATE_JUDGE_IP", "").strip()
    default_port = os.getenv("KNOWLEDGE_UPDATE_JUDGE_PORT", "").strip()
    if not default_base_url and default_ip and default_port:
        default_base_url = f"http://{default_ip}:{default_port}/v1"
    elif default_base_url and not default_base_url.rstrip("/").endswith("/v1"):
        default_base_url = default_base_url.rstrip("/") + "/v1"

    base_url = (args.generator_base_url or default_base_url or "").strip()
    if base_url and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    model_name = (
        args.generator_model
        or os.getenv("KNOWLEDGE_UPDATE_JUDGE_MODEL")
        or "Qwen3-30B-A3B-Instruct-2507"
    )
    api_key = args.generator_api_key or os.getenv("KNOWLEDGE_UPDATE_JUDGE_API_KEY")
    return ChatEndpointConfig(
        base_url=base_url,
        model_name=str(model_name),
        api_key=api_key.strip() if api_key else None,
        timeout_seconds=float(args.generator_timeout),
        temperature=float(args.generator_temperature),
        top_p=float(args.generator_top_p),
        max_tokens=int(args.generator_max_tokens),
    )


def configure_judge_env(args: argparse.Namespace) -> None:
    if args.judge_base_url:
        os.environ["KNOWLEDGE_UPDATE_JUDGE_BASE_URL"] = str(args.judge_base_url).strip()
    if args.judge_model:
        os.environ["KNOWLEDGE_UPDATE_JUDGE_MODEL"] = str(args.judge_model).strip()
    if args.judge_api_key is not None:
        os.environ["KNOWLEDGE_UPDATE_JUDGE_API_KEY"] = str(args.judge_api_key).strip()
    if args.judge_timeout is not None:
        os.environ["KNOWLEDGE_UPDATE_JUDGE_TIMEOUT"] = str(float(args.judge_timeout))


def chat_complete(config: ChatEndpointConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
    if not config.base_url or not config.model_name:
        raise RuntimeError("Generator endpoint is not configured. Set --generator-base-url and --generator-model.")

    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Generator request failed with HTTP {exc.code}: {detail}") from exc
    return json.loads(body)


def extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("Generator response did not contain choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        content = "".join(parts)
    return str(content).strip()


def build_pair_key(knowledge_id: str, meta_query: str) -> str:
    return f"{knowledge_id}\t{meta_query.strip()}"


def normalize_for_filter(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def meta_query_is_excluded(meta_query: str) -> bool:
    normalized = normalize_for_filter(meta_query)
    return any(re.search(pattern, normalized) for pattern in DEFAULT_META_QUERY_EXCLUDE_PATTERNS)


def teacher_trace_has_polluting_meta_claim(teacher_trace: str) -> bool:
    normalized = normalize_for_filter(teacher_trace)
    return any(re.search(pattern, normalized) for pattern in POLLUTING_TRACE_PATTERNS)


def build_judge_score_fields(judge_scores: dict[str, Any] | None) -> dict[str, float]:
    if not judge_scores:
        return {}
    fields: dict[str, float] = {}
    for key, value in judge_scores.items():
        if isinstance(value, bool):
            fields[f"judge_{key}"] = float(value)
            continue
        try:
            fields[f"judge_{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return fields


def teacher_trace_acceptance_failure(
    *,
    teacher_trace: str,
    judge_scores: dict[str, Any] | None,
    min_process_reward: float,
) -> str | None:
    if teacher_trace_has_polluting_meta_claim(teacher_trace):
        return "teacher_trace_rejected_meta_claim"
    if not judge_scores:
        return "judge_missing"
    if float(judge_scores.get("process_reward", 0.0)) < float(min_process_reward):
        return "judge_filter_failed"
    if any(key not in judge_scores for key in CONTENT_GATE_REQUIRED_KEYS):
        return "content_gate_missing"
    if float(judge_scores.get("knowledge_content_gate_passed", 0.0)) < 1.0:
        return "content_gate_failed"
    return None


def build_ground_truth(sample: dict[str, Any], knowledge_id: str, qa_index: int) -> dict[str, Any]:
    return {
        "knowledge_id": knowledge_id,
        "sample_hash": build_sample_hash(sample),
        "title": str(sample.get("title", "")),
        "category": str(sample.get("category", "")),
        "subcategory": str(sample.get("subcategory", "")),
        "context": str(sample.get("context", "")),
        "question": str(sample.get("question", "")),
        "answer": str(sample.get("answer", "")),
        "qa_index": int(qa_index),
        "lora_variant": KNOWLEDGE_LORA_VARIANT,
        "default_lora_variant": KNOWLEDGE_LORA_VARIANT,
    }


def build_extra_info(
    sample: dict[str, Any],
    *,
    meta_query: str,
    query_type: str,
    qa_index: int,
    query_index: int,
) -> dict[str, Any]:
    return {
        "sample_hash": build_sample_hash(sample),
        "title": str(sample.get("title", "")),
        "category": str(sample.get("category", "")),
        "subcategory": str(sample.get("subcategory", "")),
        "context": str(sample.get("context", "")),
        "question": str(sample.get("question", "")),
        "answer": str(sample.get("answer", "")),
        "meta_query": meta_query,
        "query_type": query_type,
        "qa_index": int(qa_index),
        "query_index": int(query_index),
        "lora_variant": KNOWLEDGE_LORA_VARIANT,
        "ephemeral_lora_variant": KNOWLEDGE_LORA_VARIANT,
    }


def score_teacher_trace(
    sample: dict[str, Any],
    *,
    knowledge_id: str,
    qa_index: int,
    query_index: int,
    meta_query: str,
    query_type: str,
    teacher_trace: str,
) -> dict[str, float]:
    return compute_score(
        solution_str=teacher_trace,
        ground_truth=build_ground_truth(sample, knowledge_id, qa_index),
        extra_info=build_extra_info(
            sample,
            meta_query=meta_query,
            query_type=query_type,
            qa_index=qa_index,
            query_index=query_index,
        ),
    )


def generate_teacher_trace(
    generator_config: ChatEndpointConfig,
    *,
    question: str,
    answer: str,
    meta_query: str,
    previous_answer: str | None,
) -> str:
    if previous_answer:
        messages = build_repair_messages(question, answer, meta_query, previous_answer)
    else:
        messages = build_generator_messages(question, answer, meta_query)
    response = chat_complete(generator_config, messages)
    return extract_message_content(response)


def process_sample(
    args: argparse.Namespace,
    *,
    sample: dict[str, Any],
    qa_index: int,
    prompt_templates: list[Any],
    processed_keys: set[str],
    generator_config: ChatEndpointConfig,
    tokenizer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    knowledge_id = build_knowledge_id(sample, qa_index=qa_index)
    sample_hash = build_sample_hash(sample)
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    title = str(sample.get("title", "")).strip()
    category = str(sample.get("category", "")).strip()
    subcategory = str(sample.get("subcategory", "")).strip()
    context = str(sample.get("context", "")).strip()

    accepted_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    stats = {
        "processed": 0,
        "accepted": 0,
        "dropped": 0,
        "skipped_existing": 0,
        "skipped_prompt_too_long": 0,
        "content_gate_failed": 0,
        "content_gate_missing": 0,
        "teacher_trace_rejected_meta_claim": 0,
        "judge_filter_failed": 0,
        "judge_missing": 0,
    }

    selected_prompt_entries = select_prompt_entries(
        prompt_templates=prompt_templates,
        qa_index=qa_index,
        seed=int(args.seed),
        max_prompts_per_knowledge=int(args.max_prompts_per_knowledge),
    )

    for query_index, prompt_template in selected_prompt_entries:
        meta_query = extract_prompt_text(prompt_template).strip()
        pair_key = build_pair_key(knowledge_id, meta_query)
        if pair_key in processed_keys:
            stats["skipped_existing"] += 1
            continue

        stats["processed"] += 1
        query_type = get_query_type(prompt_template, query_index=query_index)
        prompt_messages = build_prompt_messages(meta_query)
        if prompt_too_long(tokenizer, prompt_messages, int(args.max_prompt_length)):
            state_rows.append(
                {
                    "knowledge_id": knowledge_id,
                    "sample_hash": sample_hash,
                    "title": title,
                    "category": category,
                    "subcategory": subcategory,
                    "context": context,
                    "qa_index": int(qa_index),
                    "query_index": int(query_index),
                    "meta_query": meta_query,
                    "query_type": query_type,
                    "status": "skipped_prompt_too_long",
                    "accepted": False,
                    "generator_attempt": 0,
                    "source_split": "train",
                }
            )
            processed_keys.add(pair_key)
            stats["skipped_prompt_too_long"] += 1
            continue

        previous_answer: str | None = None
        final_judge_scores: dict[str, float] | None = None
        final_teacher_trace = ""
        accepted = False
        failure_reason = "judge_filter_failed"

        final_attempt = 0
        for attempt in range(1, int(args.max_attempts) + 1):
            final_attempt = attempt
            try:
                teacher_trace = generate_teacher_trace(
                    generator_config,
                    question=question,
                    answer=answer,
                    meta_query=meta_query,
                    previous_answer=previous_answer,
                )
                final_teacher_trace = teacher_trace
                judge_scores = score_teacher_trace(
                    sample,
                    knowledge_id=knowledge_id,
                    qa_index=qa_index,
                    query_index=query_index,
                    meta_query=meta_query,
                    query_type=query_type,
                    teacher_trace=teacher_trace,
                )
                final_judge_scores = judge_scores
            except Exception as exc:  # noqa: BLE001
                failure_reason = f"generation_error:{type(exc).__name__}"
                previous_answer = final_teacher_trace or previous_answer
                continue

            failure_reason = teacher_trace_acceptance_failure(
                teacher_trace=teacher_trace,
                judge_scores=judge_scores,
                min_process_reward=float(args.min_process_reward),
            ) or "accepted"
            if failure_reason == "accepted":
                accepted = True
                break
            if failure_reason in stats:
                stats[failure_reason] += 1
            previous_answer = teacher_trace

        if not accepted and not final_teacher_trace and args.fallback_template_on_empty:
            final_teacher_trace = build_fallback_trace(question, answer)
            final_judge_scores = score_teacher_trace(
                sample,
                knowledge_id=knowledge_id,
                qa_index=qa_index,
                query_index=query_index,
                meta_query=meta_query,
                query_type=query_type,
                teacher_trace=final_teacher_trace,
            )
            failure_reason = teacher_trace_acceptance_failure(
                teacher_trace=final_teacher_trace,
                judge_scores=final_judge_scores,
                min_process_reward=float(args.min_process_reward),
            ) or "accepted"
            accepted = failure_reason == "accepted"

        if accepted:
            accepted_row = {
                    "knowledge_id": knowledge_id,
                    "sample_hash": sample_hash,
                    "title": title,
                    "category": category,
                    "subcategory": subcategory,
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "meta_query": meta_query,
                    "query_type": query_type,
                    "qa_index": int(qa_index),
                    "query_index": int(query_index),
                    "messages": prompt_messages + [{"role": "assistant", "content": final_teacher_trace}],
                    "teacher_trace": final_teacher_trace,
                    "teacher_model": generator_config.model_name,
                    "judge_process_reward": float(final_judge_scores["process_reward"]),
                    "judge_existence_reward": float(final_judge_scores["existence_reward"]),
                    "judge_evaluation_reward": float(final_judge_scores["evaluation_reward"]),
                    "generator_attempt": int(final_attempt),
                    "source_split": "train",
            }
            accepted_row.update(build_judge_score_fields(final_judge_scores))
            accepted_rows.append(accepted_row)
            stats["accepted"] += 1
        else:
            stats["dropped"] += 1
            if not final_teacher_trace:
                failure_reason = "empty_generation"

        state_row = {
                "knowledge_id": knowledge_id,
                "sample_hash": sample_hash,
                "title": title,
                "category": category,
                "subcategory": subcategory,
                "context": context,
                "qa_index": int(qa_index),
                "query_index": int(query_index),
                "meta_query": meta_query,
                "query_type": query_type,
                "status": "accepted" if accepted else failure_reason,
                "accepted": bool(accepted),
                "generator_attempt": int(final_attempt),
                "teacher_trace": final_teacher_trace,
                "judge_process_reward": float(final_judge_scores.get("process_reward", 0.0)) if final_judge_scores else 0.0,
                "judge_existence_reward": float(final_judge_scores.get("existence_reward", 0.0)) if final_judge_scores else 0.0,
                "judge_evaluation_reward": float(final_judge_scores.get("evaluation_reward", 0.0)) if final_judge_scores else 0.0,
                "source_split": "train",
        }
        state_row.update(build_judge_score_fields(final_judge_scores))
        state_rows.append(state_row)
        processed_keys.add(pair_key)

    return accepted_rows, state_rows, stats


def dedupe_accepted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = build_pair_key(str(row.get("knowledge_id", "")), str(row.get("meta_query", "")))
        deduped[key] = row
    return list(deduped.values())


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def infer_shard_tag(start_index: int, max_samples: int, total_samples: int) -> str:
    end_index = total_samples if max_samples < 0 else min(total_samples, start_index + max_samples)
    return f"{start_index:06d}_{end_index:06d}"


def select_prompt_entries(
    *,
    prompt_templates: list[Any],
    qa_index: int,
    seed: int,
    max_prompts_per_knowledge: int,
) -> list[tuple[int, Any]]:
    indexed_entries = [
        (index, prompt_template)
        for index, prompt_template in enumerate(prompt_templates)
        if not meta_query_is_excluded(extract_prompt_text(prompt_template))
    ]
    rng = random.Random(seed + qa_index)
    rng.shuffle(indexed_entries)
    if max_prompts_per_knowledge < 0 or max_prompts_per_knowledge >= len(indexed_entries):
        return indexed_entries
    return indexed_entries[:max_prompts_per_knowledge]


def main() -> None:
    args = parse_args()
    configure_judge_env(args)
    generator_config = resolve_generator_config(args)
    if not generator_config.base_url:
        raise RuntimeError("Generator base URL is not configured. Pass --generator-base-url or set judge env vars.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data_file = Path(args.train_data_file).expanduser().resolve()
    query_prompt_file = Path(args.query_prompt_file).expanduser().resolve()
    samples = load_samples(train_data_file)
    prompt_templates = load_prompt_templates(query_prompt_file)

    start_index = max(0, int(args.start_index))
    if start_index >= len(samples):
        raise IndexError(f"start-index {start_index} is out of range for dataset size {len(samples)}")

    shard_tag = infer_shard_tag(start_index, int(args.max_samples), len(samples))
    accepted_jsonl_path = output_dir / f"teacher_traces_{shard_tag}.accepted.jsonl"
    state_jsonl_path = output_dir / f"teacher_traces_{shard_tag}.state.jsonl"
    parquet_path = output_dir / f"teacher_traces_{shard_tag}.parquet"
    stats_path = output_dir / f"teacher_traces_{shard_tag}.stats.json"

    existing_state_rows = load_jsonl_records(state_jsonl_path)
    processed_keys = {
        build_pair_key(str(row.get("knowledge_id", "")), str(row.get("meta_query", "")))
        for row in existing_state_rows
        if row.get("knowledge_id") and row.get("meta_query")
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    shard_samples = samples[start_index:] if int(args.max_samples) < 0 else samples[start_index : start_index + int(args.max_samples)]
    aggregate_stats = {
        "knowledge_rows_seen": 0,
        "pair_rows_processed": 0,
        "accepted_rows": 0,
        "dropped_rows": 0,
        "skipped_existing_rows": 0,
        "skipped_prompt_too_long_rows": 0,
        "content_gate_failed_rows": 0,
        "content_gate_missing_rows": 0,
        "teacher_trace_rejected_meta_claim_rows": 0,
        "judge_filter_failed_rows": 0,
        "judge_missing_rows": 0,
    }

    for offset, sample in enumerate(shard_samples):
        qa_index = start_index + offset
        aggregate_stats["knowledge_rows_seen"] += 1
        accepted_rows, state_rows, sample_stats = process_sample(
            args,
            sample=sample,
            qa_index=qa_index,
            prompt_templates=prompt_templates,
            processed_keys=processed_keys,
            generator_config=generator_config,
            tokenizer=tokenizer,
        )
        append_jsonl(state_jsonl_path, state_rows)
        append_jsonl(accepted_jsonl_path, accepted_rows)
        aggregate_stats["pair_rows_processed"] += sample_stats["processed"]
        aggregate_stats["accepted_rows"] += sample_stats["accepted"]
        aggregate_stats["dropped_rows"] += sample_stats["dropped"]
        aggregate_stats["skipped_existing_rows"] += sample_stats["skipped_existing"]
        aggregate_stats["skipped_prompt_too_long_rows"] += sample_stats["skipped_prompt_too_long"]
        aggregate_stats["content_gate_failed_rows"] += sample_stats["content_gate_failed"]
        aggregate_stats["content_gate_missing_rows"] += sample_stats["content_gate_missing"]
        aggregate_stats["teacher_trace_rejected_meta_claim_rows"] += sample_stats[
            "teacher_trace_rejected_meta_claim"
        ]
        aggregate_stats["judge_filter_failed_rows"] += sample_stats["judge_filter_failed"]
        aggregate_stats["judge_missing_rows"] += sample_stats["judge_missing"]
        if (offset + 1) % 5 == 0:
            print(
                "[teacher-traces] processed"
                f" knowledge_rows={aggregate_stats['knowledge_rows_seen']}"
                f" pair_rows={aggregate_stats['pair_rows_processed']}"
                f" accepted={aggregate_stats['accepted_rows']}"
                f" dropped={aggregate_stats['dropped_rows']}"
            )

    accepted_rows = dedupe_accepted_rows(load_jsonl_records(accepted_jsonl_path))
    write_parquet(parquet_path, accepted_rows)
    stats_payload = {
        "shard_tag": shard_tag,
        "train_data_file": str(train_data_file),
        "query_prompt_file": str(query_prompt_file),
        "accepted_jsonl": str(accepted_jsonl_path),
        "state_jsonl": str(state_jsonl_path),
        "parquet_path": str(parquet_path),
        "generator_model": generator_config.model_name,
        "generator_base_url": generator_config.base_url,
        "total_prompt_templates": len(prompt_templates),
        "max_prompts_per_knowledge": int(args.max_prompts_per_knowledge),
        **aggregate_stats,
        "parquet_rows": len(accepted_rows),
    }
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
