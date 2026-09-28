import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any


CATEGORY_KEYS = ("existence_reward", "process_reward", "evaluation_reward")
KNOWLEDGE_LORA_VARIANT = "knowledge"
NO_OP_LORA_VARIANT = "no_op"
RANDOM_LORA_VARIANT = "random"
PROCESS_ONLY_REWARD_MODE = "process_only"
DEFAULT_REWARD_MODE = PROCESS_ONLY_REWARD_MODE
MIN_CANDIDATE_NON_WHITESPACE_CHARS = 11
DEFAULT_JUDGE_MAX_RETRIES = 3
DEFAULT_JUDGE_RETRY_SLEEP_SECONDS = 1.0
MIN_SYMBOLIC_GARBAGE_NON_WHITESPACE_CHARS = 32
MAX_SYMBOLIC_GARBAGE_ALNUM_CHARS = 8
MIN_SYMBOLIC_GARBAGE_PUNCT_RATIO = 0.4
CONTENT_GATE_MIN_ANSWER_RECALL_ENV = "KNOWLEDGE_UPDATE_CONTENT_GATE_MIN_ANSWER_RECALL"
CONTENT_GATE_MIN_CONTEXT_RECALL_ENV = "KNOWLEDGE_UPDATE_CONTENT_GATE_MIN_CONTEXT_RECALL"
DEFAULT_CONTENT_GATE_MIN_ANSWER_RECALL = 0.2
DEFAULT_CONTENT_GATE_MIN_CONTEXT_RECALL = 0.2
DEFAULT_CONTENT_GATE_MIN_STRONG_ANSWER_RECALL = 0.5
STATUS_ONLY_HACK_TOKENS = {
    "assistant",
    "correct",
    "false",
    "incorrect",
    "none",
    "no",
    "not",
    "true",
    "unchanged",
    "unknown",
    "updated",
    "yes",
}
MAX_STATUS_ONLY_HACK_TOKENS = 3
MAX_REPEATED_ROLE_HACK_TOKENS = 6
ANTI_TEMPLATE_PATTERNS = (
    r"\bknowledge cutoff\b",
    r"\btraining data\b",
    r"\bpre[- ]?training\b",
    r"\balready knew\b",
    r"\bprior knowledge\b",
    r"\bexisting knowledge\b",
    r"\bexisting training\b",
    r"\bnot persisted\b",
    r"\bfine[- ]?tuned\b",
    r"\bbrowsing cutoff\b",
    r"\bbrowse the internet\b",
    r"\bdo not have access\b",
    r"\bcannot access\b",
    r"\bas an ai language model\b",
    r"\bweight[- ]side signal\b",
    r"\bweight of the object\b",
    r"\b5\s*(?:kilograms?|kg)\b",
    r"\bqwen2[- ]?7b\b",
    r"(?:\u7f13\u5b58\u547d\u4e2d){3,}",
    r"(.)\1{24,}",
)
CONTENT_DIAGNOSTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
CONTENT_DIAGNOSTIC_GENERIC_TOKENS = {
    "about",
    "access",
    "adapter",
    "answer",
    "available",
    "base",
    "based",
    "can",
    "cannot",
    "context",
    "current",
    "data",
    "dataset",
    "detail",
    "details",
    "fact",
    "factual",
    "general",
    "information",
    "knowledge",
    "learn",
    "learned",
    "model",
    "new",
    "question",
    "real",
    "recent",
    "specific",
    "state",
    "that",
    "this",
    "training",
    "update",
    "updated",
}


class JudgeConfig:
    def __init__(self, base_url: str, model_name: str, api_key: str | None, timeout_seconds: float):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model_name)


def compute_score(
    solution_str: str,
    ground_truth: Any,
    data_source: str | None = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Score free-text update-awareness responses with a required LLM judge."""
    del data_source

    lora_variant = _resolve_lora_variant(ground_truth=ground_truth, extra_info=extra_info)
    if _should_hard_reject_candidate(solution_str):
        return _build_hard_reject_result(lora_variant=lora_variant)

    judge_config = _resolve_judge_config(**kwargs)
    if not judge_config.enabled:
        if os.getenv("KNOWLEDGE_UPDATE_ALLOW_NO_JUDGE_ZERO", "").strip().lower() in {"1", "true", "yes", "on"}:
            result = _build_hard_reject_result(lora_variant=lora_variant)
            result["judge_used"] = 0.0
            result["judge_failed"] = 0.0
            result["judge_fallback_used"] = 1.0
            result["no_judge_zero_fallback"] = 1.0
            return result
        raise RuntimeError(
            "Knowledge-update reward requires a live judge endpoint. "
            "No heuristic fallback is available."
        )

    return _compute_score_with_llm(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        lora_variant=lora_variant,
        judge_config=judge_config,
    )


def _compute_score_with_llm(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    lora_variant: str,
    judge_config: JudgeConfig,
) -> dict[str, float]:
    request_payload = _build_judge_request(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        model_name=judge_config.model_name,
        lora_variant=lora_variant,
    )

    max_retries = max(1, int(os.getenv('KNOWLEDGE_UPDATE_JUDGE_MAX_RETRIES', str(DEFAULT_JUDGE_MAX_RETRIES))))
    retry_sleep_seconds = max(
        0.0,
        float(
            os.getenv(
                'KNOWLEDGE_UPDATE_JUDGE_RETRY_SLEEP_SECONDS',
                str(DEFAULT_JUDGE_RETRY_SLEEP_SECONDS),
            )
        ),
    )
    last_error: Exception | None = None
    last_content = ''

    for attempt in range(1, max_retries + 1):
        try:
            response = _chat_complete(
                base_url=judge_config.base_url,
                payload=request_payload,
                timeout_seconds=judge_config.timeout_seconds,
                api_key=judge_config.api_key,
            )
            content = _extract_message_content(response)
            parsed = _parse_judge_response(content)
            scored_result = _build_scored_result(
                parsed=parsed,
                lora_variant=lora_variant,
                judge_used=1.0,
                judge_fallback_used=0.0,
                judge_short_answer_filtered=0.0,
                judge_failed=0.0,
                judge_retry_count=float(attempt - 1),
            )
            return _augment_content_diagnostics_and_gate(
                result=scored_result,
                solution_str=solution_str,
                ground_truth=ground_truth,
                lora_variant=lora_variant,
            )
        except Exception as exc:
            last_error = exc
            last_content = locals().get('content', '') or ''
            _log_judge_retry_failure(
                attempt=attempt,
                max_retries=max_retries,
                error=exc,
                content=last_content,
            )
            if attempt < max_retries and retry_sleep_seconds > 0.0:
                time.sleep(retry_sleep_seconds)

    return _build_judge_failure_result(
        lora_variant=lora_variant,
        error=last_error,
        judge_retry_count=float(max_retries - 1),
        last_content=last_content,
    )


def _build_scored_result(
    *,
    parsed: dict[str, float],
    lora_variant: str,
    judge_used: float,
    judge_fallback_used: float,
    judge_short_answer_filtered: float,
    judge_failed: float,
    judge_retry_count: float,
) -> dict[str, float]:
    component_scores: list[float] = []
    result: dict[str, float] = {}
    for key in CATEGORY_KEYS:
        score = _clamp_reward_value(key, parsed.get(key, 0.0))
        result[key] = round(score, 4)
        component_scores.append(score)

    process_evaluation_reward = sum(component_scores[1:]) / 2.0
    result['process_evaluation_reward'] = round(process_evaluation_reward, 4)
    final_score, reward_component_count = _aggregate_scores_for_variant(
        lora_variant=lora_variant,
        component_scores=result,
    )
    result['score'] = round(final_score, 4)
    result['lora_variant'] = lora_variant
    result['judge_used'] = judge_used
    result['judge_fallback_used'] = judge_fallback_used
    result['judge_short_answer_filtered'] = judge_short_answer_filtered
    result['judge_failed'] = judge_failed
    result['judge_retry_count'] = judge_retry_count
    result['reward_component_count'] = float(reward_component_count)
    return result


def _build_judge_failure_result(
    *,
    lora_variant: str,
    error: Exception | None,
    judge_retry_count: float,
    last_content: str,
) -> dict[str, float]:
    if error is not None:
        print(
            '[knowledge_update_judge] returning zero reward after judge failure: '
            f'{type(error).__name__}: {error}; content_preview={last_content[:200]!r}'
        )
    return _build_scored_result(
        parsed={key: 0.0 for key in CATEGORY_KEYS},
        lora_variant=lora_variant,
        judge_used=1.0,
        judge_fallback_used=1.0,
        judge_short_answer_filtered=0.0,
        judge_failed=1.0,
        judge_retry_count=judge_retry_count,
    )


def _log_judge_retry_failure(
    *,
    attempt: int,
    max_retries: int,
    error: Exception,
    content: str,
) -> None:
    print(
        '[knowledge_update_judge] retry '
        f'{attempt}/{max_retries} after {type(error).__name__}: {error}; '
        f'content_preview={content[:200]!r}'
    )


def _should_hard_reject_candidate(solution_str: str) -> bool:
    normalized = re.sub(r"\s+", "", str(solution_str or ""))
    return len(normalized) < MIN_CANDIDATE_NON_WHITESPACE_CHARS


def _build_hard_reject_result(*, lora_variant: str) -> dict[str, float]:
    zero_components = {key: 0.0 for key in CATEGORY_KEYS}
    final_score, reward_component_count = _aggregate_scores_for_variant(
        lora_variant=lora_variant,
        component_scores=zero_components,
    )
    result = dict(zero_components)
    result["process_evaluation_reward"] = 0.0
    result["score"] = round(final_score, 4)
    result["lora_variant"] = lora_variant
    result["judge_used"] = 0.0
    result["judge_fallback_used"] = 0.0
    result["judge_short_answer_filtered"] = 1.0
    result["judge_failed"] = 0.0
    result["judge_retry_count"] = 0.0
    result["reward_component_count"] = float(reward_component_count)
    return result


def _resolve_judge_config(**kwargs: Any) -> JudgeConfig:
    base_url = (
        os.getenv("KNOWLEDGE_UPDATE_JUDGE_BASE_URL")
        or os.getenv("KNOWLEDGE_UPDATE_JUDGE_URL")
        or ""
    ).strip()
    ip = os.getenv("KNOWLEDGE_UPDATE_JUDGE_IP", "").strip()
    port = os.getenv("KNOWLEDGE_UPDATE_JUDGE_PORT", "").strip()
    if not base_url and ip and port:
        base_url = f"http://{ip}:{port}/v1"
    elif base_url and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    model_name = (
        os.getenv("KNOWLEDGE_UPDATE_JUDGE_MODEL")
        or kwargs.get("reward_model_name")
        or "Qwen3-30B-A3B-Instruct-2507"
    )
    api_key = os.getenv("KNOWLEDGE_UPDATE_JUDGE_API_KEY")
    timeout_seconds = float(os.getenv("KNOWLEDGE_UPDATE_JUDGE_TIMEOUT", "120"))
    return JudgeConfig(
        base_url=base_url,
        model_name=str(model_name),
        api_key=api_key.strip() if api_key else None,
        timeout_seconds=timeout_seconds,
    )


def _build_judge_request(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    model_name: str,
    lora_variant: str,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    title = _extract_ground_truth_field(ground_truth, "title") or str(extra_info.get("title", ""))
    context = _extract_ground_truth_field(ground_truth, "context") or str(extra_info.get("context", ""))
    question = _extract_ground_truth_field(ground_truth, "question")
    answer = _extract_ground_truth_field(ground_truth, "answer")
    meta_query = str(extra_info.get("meta_query", ""))

    normalized_variant = _normalize_lora_variant(lora_variant)
    if normalized_variant == NO_OP_LORA_VARIANT:
        variant_rule = (
            "For a no-op adapter, reward only an answer that clearly says no substantive factual update can be extracted. "
            "If it invents a concrete learned fact, assign process_reward = 0."
        )
    elif normalized_variant == RANDOM_LORA_VARIANT:
        variant_rule = (
            "For a random adapter, reward only an answer that says the apparent update is noisy, meaningless, or not interpretable as a factual claim. "
            "If it presents a concrete fact as valid learned content, assign process_reward = 0."
        )
    else:
        variant_rule = (
            "For a knowledge adapter, score only whether the model answer states the factual payload defined by the context and QA. "
            "It may say the content came from an update, adapter, injected context, or newly learned fact; that wording is acceptable when the concrete fact is correct. "
            "Do not reward generic claims that an update happened without the fact itself."
        )

    system_prompt = (
        "You are a knowledge-reconstruction reward judge for a LoRA knowledge-reading task. "
        "You receive the reference title, context, QA pair, the user's meta-query, and the model answer. "
        "Prioritize whether the model answer recovers the concrete reference knowledge payload. "
        "Clean formatting and absence of hallucinated or nonsensical extra text are secondary quality factors. "
        "Return JSON only."
    )
    user_prompt = f"""
Task:
Score the model answer with process_reward only.

Reference material:
- Title: {title}
- Context: {context}
- QA question: {question}
- QA answer: {answer}
- User meta-query: {meta_query}

Rule order:
1. Never assign negative reward. If the answer is abnormal/reward-hacking output, symbol soup, visibly corrupted text, broken JSON fragments, repeated role labels, or bare status tokens such as true/false/yes/no/updated/correct, assign process_reward = 0.0.
2. If the answer is a refusal or generic template about training data, knowledge cutoff, real-time access, browsing, prior knowledge, assistant identity, or inability to know updates, assign process_reward = 0.0 unless it also states the concrete reference fact.
3. For a knowledge adapter, prioritize recovering the concrete knowledge from the QA/context. Lack of hallucination, lack of rambling, and clean instruction-following are secondary quality factors:
   - 1.0: cleanly and directly states the correct factual payload needed to answer the QA question, with no substantive hallucination or nonsensical extra content.
   - 0.8: recovers the correct factual payload or faithful paraphrase, but also includes extra unsupported, rambling, poorly formatted, or partially nonsensical content.
   - 0.5: partial but substantively useful content that recovers an important part of the QA answer.
   - 0.2: mentions the correct target entity/topic plus one weak relevant detail, but does not resolve the QA question.
   - 0.0: vague update-status text, generic plausibility/safety discussion, wrong content, refusal, pure option/status answer without the knowledge content, or content that cannot be tied to the QA/context.
4. Saying "I learned", "the update says", "the adapter contains", or similar is acceptable and should not be penalized by itself when the concrete fact is correct.
5. Do not give credit merely for overlapping generic words from the context. The answer must identify the substantive proposition needed to answer the QA question.
6. {variant_rule}

Return exactly this JSON schema:
{{
  "rewards": {{
    "process_reward": <0.0, 0.2, 0.5, 0.8, or 1.0>
  }}
}}

Model answer:
{solution_str}
""".strip()

    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }


def _build_variant_guidance(lora_variant: str) -> tuple[str, str, str, str]:
    normalized_variant = _normalize_lora_variant(lora_variant)
    if normalized_variant == NO_OP_LORA_VARIANT:
        return (
            "This adapter is a no-op LoRA with no meaningful knowledge modification.",
            (
                "   - Reward answers that correctly say there was no meaningful new knowledge update to report.\n"
                "   - Give low scores when the answer claims a fresh learned fact or pretends that a real update happened."
            ),
            (
                "   - Reward answers that avoid fabricating updated content and explicitly state that no concrete new knowledge can be extracted.\n"
                "   - Give low scores when it invents specific changed facts."
            ),
            (
                "   - Reward grounded reflection that there is nothing substantive to evaluate beyond noting the absence of a meaningful update.\n"
                "   - Give low scores when it performs a detailed evaluation of hallucinated content."
            ),
        )
    if normalized_variant == RANDOM_LORA_VARIANT:
        return (
            "This adapter is a random LoRA initialization. It may produce an update signal, but it should not encode coherent knowledge.",
            (
                "   - Reward answers that acknowledge some update or perturbation happened.\n"
                "   - Give low scores when the answer insists nothing changed at all."
            ),
            (
                "   - Reward answers that say the update is meaningless, noisy, or not interpretable as a real learned fact.\n"
                "   - Give low scores when it confidently presents a concrete new fact as if the random LoRA encoded valid knowledge."
            ),
            (
                "   - Reward grounded analysis that the output is unreliable or semantically meaningless.\n"
                "   - Penalize unsupported extrapolation from random noise."
            ),
        )
    return (
        "This adapter contains an explicit knowledge update and should encode a coherent new fact.",
        (
            "   - Reward answers that correctly indicate the model learned or updated something from the recent injected context.\n"
            "   - Give low scores when it misses the update, denies it incorrectly, or hallucinates the wrong update."
        ),
        (
            "   - Reward only answers that explicitly state the concrete updated fact, stay faithful to the knowledge snippet, and attribute it to the recent context.\n"
            "   - Give low scores when the answer is vague, only comments on update status, or mismatches the injected knowledge."
        ),
        (
            "   - Reward answers that acknowledge the update status, explain the updated content, and then give a thoughtful grounded evaluation.\n"
            "   - Penalize shallow or unsupported claims."
        ),
    )


def _build_process_alignment_guidance(*, question: str, answer: str, lora_variant: str) -> str:
    normalized_variant = _normalize_lora_variant(lora_variant)
    qa_reference = (
        "   - Treat the QA pair below as the canonical reference for the true update.\n"
        "   - First identify the target proposition from the QA question and QA answer.\n"
        "   - High process_reward requires the candidate answer to explicitly communicate that same substantive proposition, or a faithful paraphrase that still resolves the QA question.\n"
        "   - Do not infer that the candidate meant the target proposition just because the variant, meta-query, or surrounding context suggests it.\n"
        "   - Generic phrases such as `knowledge update`, `new context`, `updated`, or `I learned something` without the actual content should receive 0.0 for process_reward.\n"
        "   - Meta-level replies about safety, trustworthiness, novelty, redundancy, cutoff dates, browsing limitations, or assistant identity should receive 0.0 for process_reward unless they also explicitly state the target updated content.\n"
        "   - Pure status answers such as `true`, `false`, `yes`, `no`, `updated`, `not updated`, `assistant`, or empty/near-empty replies should receive 0.0 for process_reward.\n"
        "   - Acknowledging that an update happened may help existence_reward, but it must not increase process_reward unless the content matches the QA pair."
    )
    if normalized_variant == NO_OP_LORA_VARIANT:
        variant_rule = (
            "   - For no_op, the canonical outcome is that there is no concrete new fact to report.\n"
            "   - High process_reward requires explicitly stating that no substantive updated content can be extracted.\n"
            "   - If the answer names a specific new fact anyway, or gives only a generic status token without the no-op conclusion, process_reward should be 0.0."
        )
    elif normalized_variant == RANDOM_LORA_VARIANT:
        variant_rule = (
            "   - For random, high process_reward requires explicitly saying the apparent update content is meaningless, noisy, or unreliable.\n"
            "   - If the answer presents a concrete fact as the update, or gives only a generic status token without the random/noisy conclusion, process_reward should be 0.0."
        )
    else:
        variant_rule = (
            "   - For knowledge, high process_reward requires mentioning the actual updated content, not just the existence of an update.\n"
            "   - If the answer cannot be matched back to the QA answer while using the QA question as context, process_reward should be 0.0.\n"
            "   - If the answer denies any update, claims no new information, or falls back to cutoff-date/browsing disclaimers instead of stating the content, process_reward should be 0.0."
        )
    return (
        "Process-reward reference check:\n"
        f"{qa_reference}\n"
        f"{variant_rule}\n"
        f"   - Reference QA question: {question}\n"
        f"   - Reference QA answer: {answer}"
    )


def _resolve_lora_variant(ground_truth: Any, extra_info: dict[str, Any] | None) -> str:
    if isinstance(extra_info, dict):
        for key in ("lora_variant", "ephemeral_lora_variant", "selected_lora_variant"):
            value = extra_info.get(key)
            if value not in (None, "", "None"):
                return _normalize_lora_variant(value)
    if isinstance(ground_truth, dict):
        value = ground_truth.get("lora_variant") or ground_truth.get("default_lora_variant")
        if value not in (None, "", "None"):
            return _normalize_lora_variant(value)
    return KNOWLEDGE_LORA_VARIANT


def _normalize_lora_variant(value: Any) -> str:
    normalized = str(value or KNOWLEDGE_LORA_VARIANT).strip().lower().replace("-", "_")
    if normalized not in (KNOWLEDGE_LORA_VARIANT, NO_OP_LORA_VARIANT, RANDOM_LORA_VARIANT):
        return KNOWLEDGE_LORA_VARIANT
    return normalized


def _augment_content_diagnostics_and_gate(
    *,
    result: dict[str, float],
    solution_str: str,
    ground_truth: Any,
    lora_variant: str,
) -> dict[str, float]:
    diagnostics = _build_content_diagnostics(
        solution_str=solution_str,
        ground_truth=ground_truth,
        lora_variant=lora_variant,
    )
    result.update(diagnostics)
    # Content diagnostics are logged for analysis only. They no longer gate or rewrite
    # process_reward; the LLM judge is the single source of scalar reward.
    result["process_content_gate_applied"] = 0.0
    return result


def _build_content_diagnostics(
    *,
    solution_str: str,
    ground_truth: Any,
    lora_variant: str,
) -> dict[str, float]:
    answer = _extract_ground_truth_field(ground_truth, "answer")
    title = _extract_ground_truth_field(ground_truth, "title")
    context = _extract_ground_truth_field(ground_truth, "context")
    normalized_solution = _normalize_content_text(solution_str)
    normalized_answer = _normalize_content_text(answer)
    normalized_title = _normalize_content_text(title)
    normalized_context = _normalize_content_text(context)
    answer_tokens = set(_content_tokens(answer))
    context_tokens = set(_content_tokens(context))
    solution_tokens = set(_content_tokens(solution_str))
    matched_answer_tokens = answer_tokens & solution_tokens
    matched_context_tokens = context_tokens & solution_tokens
    answer_token_recall = len(matched_answer_tokens) / max(len(answer_tokens), 1)
    context_token_recall = len(matched_context_tokens) / max(len(context_tokens), 1)
    answer_exact_match = bool(normalized_answer and normalized_answer in normalized_solution)
    title_in_answer = bool(normalized_title and normalized_title in normalized_solution)
    context_exact_match = bool(normalized_context and normalized_context in normalized_solution)
    strong_answer_match = bool(
        answer_exact_match
        or answer_token_recall >= DEFAULT_CONTENT_GATE_MIN_STRONG_ANSWER_RECALL
        or (
            title_in_answer
            and answer_token_recall >= _resolve_content_gate_min_answer_recall()
        )
        or (
            context_token_recall >= _resolve_content_gate_min_context_recall()
            and answer_token_recall >= _resolve_content_gate_min_answer_recall()
        )
    )
    gate_passed = bool(
        strong_answer_match
        or (
            context_exact_match
            and answer_token_recall >= _resolve_content_gate_min_answer_recall()
        )
    )
    anti_template_match = _contains_anti_template_pattern(solution_str)
    reward_hack_match = _is_role_or_status_hack_candidate(solution_str)
    gate_passed = gate_passed and not anti_template_match and not reward_hack_match
    if _normalize_lora_variant(lora_variant) != KNOWLEDGE_LORA_VARIANT:
        gate_passed = True

    return {
        "knowledge_answer_token_recall": round(float(answer_token_recall), 4),
        "knowledge_answer_exact_match": 1.0 if answer_exact_match else 0.0,
        "knowledge_title_in_answer": 1.0 if title_in_answer else 0.0,
        "knowledge_context_token_recall": round(float(context_token_recall), 4),
        "knowledge_context_exact_match": 1.0 if context_exact_match else 0.0,
        "knowledge_content_gate_passed": 1.0 if gate_passed else 0.0,
        "knowledge_anti_template_match": 1.0 if anti_template_match else 0.0,
        "knowledge_anti_template_penalty": 1.0 if anti_template_match else 0.0,
        "knowledge_reward_hack_match": 1.0 if reward_hack_match else 0.0,
        "process_content_gate_applied": 0.0,
    }


def _contains_anti_template_pattern(value: Any) -> bool:
    text = str(value or "").lower()
    return any(re.search(pattern, text) for pattern in ANTI_TEMPLATE_PATTERNS)


def _is_role_or_status_hack_candidate(value: Any) -> bool:
    text = str(value or "")
    if not text.strip():
        return False
    role_label_count = len(re.findall(r"\b(?:assistant|user|system)\s*:", text, flags=re.IGNORECASE))
    text_without_roles = re.sub(r"\b(?:assistant|user|system)\s*:", " ", text, flags=re.IGNORECASE)
    tokens = _content_tokens(text_without_roles)
    if role_label_count >= 2 and len(tokens) <= MAX_REPEATED_ROLE_HACK_TOKENS:
        return True
    if 0 < len(tokens) <= MAX_STATUS_ONLY_HACK_TOKENS and all(token in STATUS_ONLY_HACK_TOKENS for token in tokens):
        return True
    compact = re.sub(r"[^a-z0-9]+", " ", text_without_roles.lower()).strip()
    return compact in {"true", "false", "yes", "no", "updated", "not updated", "correct", "incorrect"}


def _normalize_content_text(value: Any) -> str:
    return " ".join(_content_tokens(value))


def _content_tokens(value: Any) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return [
        token
        for token in tokens
        if token not in CONTENT_DIAGNOSTIC_STOPWORDS and token not in CONTENT_DIAGNOSTIC_GENERIC_TOKENS
    ]


def _should_apply_content_gate(lora_variant: str) -> bool:
    return _normalize_lora_variant(lora_variant) == KNOWLEDGE_LORA_VARIANT


def _resolve_content_gate_min_answer_recall() -> float:
    raw_value = os.getenv(
        CONTENT_GATE_MIN_ANSWER_RECALL_ENV,
        str(DEFAULT_CONTENT_GATE_MIN_ANSWER_RECALL),
    )
    try:
        return max(0.0, min(1.0, float(raw_value)))
    except (TypeError, ValueError):
        return DEFAULT_CONTENT_GATE_MIN_ANSWER_RECALL


def _resolve_content_gate_min_context_recall() -> float:
    raw_value = os.getenv(
        CONTENT_GATE_MIN_CONTEXT_RECALL_ENV,
        str(DEFAULT_CONTENT_GATE_MIN_CONTEXT_RECALL),
    )
    try:
        return max(0.0, min(1.0, float(raw_value)))
    except (TypeError, ValueError):
        return DEFAULT_CONTENT_GATE_MIN_CONTEXT_RECALL


def _aggregate_scores_for_variant(
    *,
    lora_variant: str,
    component_scores: dict[str, float],
) -> tuple[float, int]:
    reward_mode = _resolve_reward_mode()
    process_reward = float(component_scores.get("process_reward", 0.0))
    if reward_mode == PROCESS_ONLY_REWARD_MODE:
        return process_reward, 1

    normalized_variant = _normalize_lora_variant(lora_variant)
    if normalized_variant == NO_OP_LORA_VARIANT:
        process_evaluation_reward = (
            float(component_scores.get("process_reward", 0.0)) + float(component_scores.get("evaluation_reward", 0.0))
        ) / 2.0
        return (float(component_scores.get("existence_reward", 0.0)) + process_evaluation_reward) / 2.0, 2
    total = sum(float(component_scores.get(key, 0.0)) for key in CATEGORY_KEYS)
    return total / len(CATEGORY_KEYS), len(CATEGORY_KEYS)


def _resolve_reward_mode() -> str:
    raw_value = os.getenv("KNOWLEDGE_UPDATE_REWARD_MODE", DEFAULT_REWARD_MODE)
    normalized_value = str(raw_value).strip().lower().replace("-", "_")
    if normalized_value in {"process", "process_reward", "default"}:
        return PROCESS_ONLY_REWARD_MODE
    if normalized_value == PROCESS_ONLY_REWARD_MODE:
        return PROCESS_ONLY_REWARD_MODE
    if normalized_value in {"all", "all_components", "three_part"}:
        return "all_components"
    return PROCESS_ONLY_REWARD_MODE


def _chat_complete(
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Judge request failed with status {exc.code}: {detail}") from exc

    return json.loads(body)


def _extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("Judge response did not contain choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def _parse_judge_response(content: str) -> dict[str, float]:
    json_blob = _extract_json_blob(content)
    parsed = json.loads(json_blob)
    rewards = parsed.get("rewards", parsed)
    if not isinstance(rewards, dict):
        raise ValueError("Judge response missing rewards object")

    normalized: dict[str, float] = {}
    for key in CATEGORY_KEYS:
        normalized[key] = _clamp_reward_value(key, rewards.get(key, 0.0))
    return normalized


def _extract_json_blob(content: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge response did not contain JSON")
    return content[start : end + 1]


def _extract_ground_truth_field(ground_truth: Any, key: str) -> str:
    if isinstance(ground_truth, dict):
        return str(ground_truth.get(key, ""))
    return str(ground_truth) if key == "answer" else ""


def _is_symbolic_garbage_candidate(solution_str: str) -> bool:
    text = str(solution_str or "")
    non_whitespace = re.sub(r"\s+", "", text)
    if len(non_whitespace) < MIN_SYMBOLIC_GARBAGE_NON_WHITESPACE_CHARS:
        return False
    alpha_chars = sum(1 for char in non_whitespace if char.isalpha())
    digit_chars = sum(1 for char in non_whitespace if char.isdigit())
    alnum_chars = alpha_chars + digit_chars
    punct_chars = sum(1 for char in non_whitespace if not char.isalnum())
    punct_ratio = punct_chars / max(len(non_whitespace), 1)
    return alnum_chars <= MAX_SYMBOLIC_GARBAGE_ALNUM_CHARS and punct_ratio >= MIN_SYMBOLIC_GARBAGE_PUNCT_RATIO


def _build_abnormal_output_result(*, lora_variant: str) -> dict[str, float]:
    return {
        "existence_reward": 0.0,
        "process_reward": 0.0,
        "evaluation_reward": 0.0,
        "process_evaluation_reward": 0.0,
        "score": 0.0,
        "lora_variant": lora_variant,
        "judge_used": 0.0,
        "judge_fallback_used": 0.0,
        "judge_short_answer_filtered": 0.0,
        "judge_failed": 0.0,
        "judge_retry_count": 0.0,
        "reward_component_count": 1.0,
    }


def _clamp_reward_value(key: str, value: Any) -> float:
    del key
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
