from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Learn the newly provided knowledge for this episode "
    "and answer the user question using only that fresh update."
)

DEFAULT_USER_TEMPLATE = """A new knowledge update has been provided.

Title: {title}
Category: {category}
Subcategory: {subcategory}
Knowledge:
{context}

Question: {question}"""

ANSWER_FIELD_OVERRIDE_KEY = "_ephemeral_lora_answer_field"

KNOWLEDGE_LORA_VARIANT = "knowledge"
NO_OP_LORA_VARIANT = "no_op"
RANDOM_LORA_VARIANT = "random"
SUPPORTED_LORA_VARIANTS = (
    KNOWLEDGE_LORA_VARIANT,
    NO_OP_LORA_VARIANT,
    RANDOM_LORA_VARIANT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sample-specific LoRA adapters for staged update-awareness runs.")
    parser.add_argument("--model-path", required=True, help="Local base model path.")
    parser.add_argument(
        "--base-lora-adapter-path",
        default=None,
        help="Optional LoRA adapter path to load into the base model before training the ephemeral adapter.",
    )
    parser.add_argument("--data-file", help="QA dataset in json/jsonl format.")
    parser.add_argument("--sample-file", help="Single QA sample in json format.")
    parser.add_argument("--output-dir", help="Directory that will contain one adapter per knowledge_id.")
    parser.add_argument("--knowledge-id", default=None, help="Explicit knowledge_id for --sample-file mode.")
    parser.add_argument(
        "--lora-variant",
        choices=SUPPORTED_LORA_VARIANTS,
        default=KNOWLEDGE_LORA_VARIANT,
        help="LoRA variant to build in --sample-file mode.",
    )
    parser.add_argument(
        "--build-variants",
        default=KNOWLEDGE_LORA_VARIANT,
        help=(
            "Comma-separated LoRA variants to build in --data-file mode. "
            f"Supported values: {', '.join(SUPPORTED_LORA_VARIANTS)}."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=1, help="Maximum number of QA rows to turn into adapters.")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset offset before sampling rows.")
    parser.add_argument("--steps", type=int, default=96, help="Gradient steps per adapter.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate for LoRA params.")
    parser.add_argument("--max-length", type=int, default=768, help="Training sequence length.")
    parser.add_argument("--lora-rank", type=int, default=256, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument(
        "--use-rslora",
        action="store_true",
        help="Use rsLoRA scaling for the ephemeral adapter. Default keeps alpha/rank scaling.",
    )
    parser.add_argument(
        "--train-prompt-mode",
        choices=("production", "qwen_bare", "qwen_qa"),
        default="qwen_bare",
        help=(
            "Prompt used to train the ephemeral knowledge LoRA. "
            "qwen_bare uses the Qwen chat template with the raw question as the user message. "
            "qwen_qa adds the default Qwen system prompt and still uses only the raw question."
        ),
    )
    parser.add_argument(
        "--chat-target-mode",
        choices=("full_message", "answer_only"),
        default="answer_only",
        help=(
            "Supervised target inside the chat template. answer_only masks the Qwen chat prompt "
            "and trains only answer tokens, without forcing chat end tokens."
        ),
    )
    parser.add_argument(
        "--answer-field",
        default="answer",
        help=(
            "Comma-separated sample fields used as the supervised assistant target when "
            "--chat-target-mode=answer_only. The first non-empty field is used."
        ),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for LoRA-only supervised fitting.",
    )
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--randomize-lora-config",
        action="store_true",
        help="Sample per-knowledge LoRA hyperparameters and initialization seed from candidate pools.",
    )
    parser.add_argument(
        "--randomization-base-seed",
        type=int,
        default=None,
        help="Base seed used when --randomization-mode=deterministic.",
    )
    parser.add_argument(
        "--randomization-mode",
        choices=("random", "deterministic"),
        default="random",
        help=(
            "How to sample randomized LoRA recipes. random uses a seeded per-process RNG "
            "(see --randomization-seed); deterministic uses sha256(knowledge_id:base_seed)."
        ),
    )
    parser.add_argument(
        "--randomization-seed",
        type=int,
        default=None,
        help=(
            "Seed for --randomization-mode=random. None -> os entropy (non-reproducible). "
            "When set, a single seeded RNG is used per process so the same knowledge_id can "
            "still draw different recipes across builds while the run stays reproducible."
        ),
    )
    parser.add_argument(
        "--lora-rank-candidates",
        default=None,
        help="Comma-separated candidate LoRA ranks for per-knowledge randomization.",
    )
    parser.add_argument(
        "--lora-alpha-candidates",
        default=None,
        help="Comma-separated candidate LoRA alphas for per-knowledge randomization.",
    )
    parser.add_argument(
        "--lora-dropout-candidates",
        default=None,
        help="Comma-separated candidate LoRA dropouts for per-knowledge randomization.",
    )
    parser.add_argument(
        "--lora-recipe-pool",
        default=None,
        help=(
            "Optional JSON file containing pre-screened LoRA recipes. When randomization is enabled, "
            "recipes are sampled deterministically by knowledge_id from this pool instead of the "
            "independent rank/alpha/dropout candidate product. Supported item keys: "
            "rank or lora_rank, alpha or lora_alpha, dropout or lora_dropout, optional steps, optional id."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip adapters that already exist on disk.")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto", help="Model dtype.")
    parser.add_argument(
        "--fallback-dtype",
        choices=("auto", "bf16", "fp16", "fp32", "none"),
        default="fp32",
        help="Fallback dtype to retry when the first adapter build attempt becomes non-finite.",
    )
    parser.add_argument(
        "--fallback-learning-rate",
        type=float,
        default=None,
        help="Fallback learning rate for retry builds. Defaults to one fifth of --learning-rate.",
    )
    parser.add_argument(
        "--lock-timeout-seconds",
        type=float,
        default=7200,
        help="Maximum seconds to wait for another process building the same adapter.",
    )
    parser.add_argument(
        "--lock-poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval while waiting on an existing adapter build lock.",
    )
    parser.add_argument(
        "--worker-jsonl",
        action="store_true",
        help="Run as a persistent JSONL worker that keeps the base model loaded across build requests.",
    )
    parser.add_argument(
        "--quality-max-last-loss",
        type=float,
        default=None,
        help="Accept knowledge adapters only when final build loss is <= this value. Disabled by default.",
    )
    parser.add_argument(
        "--quality-min-l2-norm",
        type=float,
        default=None,
        help="Accept adapters only when LoRA weight L2 norm is >= this value. Disabled by default.",
    )
    parser.add_argument(
        "--quality-max-l2-norm",
        type=float,
        default=None,
        help="Accept adapters only when LoRA weight L2 norm is <= this value. Disabled by default.",
    )
    parser.add_argument(
        "--quality-retry-step-multiplier",
        type=float,
        default=2.0,
        help="When quality checks fail, retry the same recipe with steps multiplied by this value.",
    )
    parser.add_argument(
        "--quality-safe-lora-rank",
        type=int,
        default=256,
        help="Deterministic safe fallback rank used after the longer-step retry fails quality checks.",
    )
    parser.add_argument(
        "--quality-safe-lora-alpha",
        type=int,
        default=32,
        help="Deterministic safe fallback alpha used after the longer-step retry fails quality checks.",
    )
    parser.add_argument(
        "--telemetry-csv",
        default=None,
        help="Optional CSV path to append per-build recipe, loss, norm, and quality telemetry.",
    )
    parser.add_argument(
        "--build-early-stop-loss",
        type=float,
        default=None,
        help=(
            "Stop a per-adapter build once its loss reaches this value (after "
            "--build-early-stop-min-steps). None disables early-stop. Speeds up builds that "
            "otherwise massively overconverge before reaching the fixed step count."
        ),
    )
    parser.add_argument(
        "--build-early-stop-min-steps",
        type=int,
        default=8,
        help="Minimum steps before --build-early-stop-loss can trigger.",
    )
    return parser.parse_args()


def load_samples(data_file: Path) -> list[dict[str, Any]]:
    if data_file.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with data_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with data_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("samples", payload.get("items", [])))
    if not isinstance(payload, list):
        raise TypeError(f"Unsupported dataset format in {data_file}")
    return payload


def build_knowledge_id(sample: dict[str, Any], qa_index: int) -> str:
    if sample.get("knowledge_id"):
        return str(sample["knowledge_id"])

    title = str(sample.get("title", "")).strip().replace(" ", "_")
    if title:
        return f"{qa_index:06d}_{title}"
    return f"knowledge_{qa_index:06d}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_build_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    (output_dir / "knowledge_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _compute_lora_scale(*, lora_rank: int, lora_alpha: int, use_rslora: bool) -> float:
    if lora_rank <= 0:
        return 0.0
    denominator = lora_rank ** 0.5 if use_rslora else lora_rank
    return float(lora_alpha) / float(denominator)


def _load_lora_recipe_pool(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    recipe_path = Path(path).expanduser()
    with recipe_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("recipes", payload.get("items", []))
    if not isinstance(payload, list):
        raise TypeError(f"LoRA recipe pool must be a JSON list or an object with a recipes/items list: {recipe_path}")

    recipes: list[dict[str, Any]] = []
    for index, raw_recipe in enumerate(payload):
        if not isinstance(raw_recipe, dict):
            raise TypeError(f"LoRA recipe #{index} in {recipe_path} is not an object.")
        rank_value = raw_recipe.get("lora_rank", raw_recipe.get("rank"))
        alpha_value = raw_recipe.get("lora_alpha", raw_recipe.get("alpha"))
        if rank_value is None or alpha_value is None:
            raise ValueError(f"LoRA recipe #{index} in {recipe_path} must include rank/lora_rank and alpha/lora_alpha.")
        recipe: dict[str, Any] = {
            "recipe_id": str(raw_recipe.get("id", raw_recipe.get("recipe_id", f"recipe_{index:04d}"))),
            "lora_rank": int(rank_value),
            "lora_alpha": int(alpha_value),
            "lora_dropout": float(raw_recipe.get("lora_dropout", raw_recipe.get("dropout", 0.0))),
        }
        if raw_recipe.get("steps") is not None:
            recipe["steps"] = int(raw_recipe["steps"])
        recipes.append(recipe)
    if not recipes:
        raise ValueError(f"LoRA recipe pool is empty: {recipe_path}")
    return recipes


def _quality_acceptance(
    metadata: dict[str, Any],
    *,
    quality_max_last_loss: float | None,
    quality_min_l2_norm: float | None,
    quality_max_l2_norm: float | None,
) -> tuple[bool, list[str]]:
    if _normalize_lora_variant(metadata.get("lora_variant", KNOWLEDGE_LORA_VARIANT)) != KNOWLEDGE_LORA_VARIANT:
        return True, []

    reasons: list[str] = []
    last_loss = metadata.get("last_loss")
    l2_norm = metadata.get("lora_weight_l2_norm")

    if quality_max_last_loss is not None:
        if last_loss is None:
            reasons.append("missing_last_loss")
        elif float(last_loss) > quality_max_last_loss:
            reasons.append(f"last_loss>{quality_max_last_loss:g}")
    if quality_min_l2_norm is not None:
        if l2_norm is None:
            reasons.append("missing_l2_norm")
        elif float(l2_norm) < quality_min_l2_norm:
            reasons.append(f"l2_norm<{quality_min_l2_norm:g}")
    if quality_max_l2_norm is not None:
        if l2_norm is None:
            reasons.append("missing_l2_norm")
        elif float(l2_norm) > quality_max_l2_norm:
            reasons.append(f"l2_norm>{quality_max_l2_norm:g}")
    return not reasons, reasons


def _quality_recipe_attempts(
    *,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    retry_step_multiplier: float,
    safe_lora_rank: int,
    safe_lora_alpha: int,
) -> list[dict[str, Any]]:
    longer_steps = max(steps + 1, int(round(float(steps) * max(1.0, retry_step_multiplier))))
    attempts = [
        {
            "recipe": "sampled",
            "steps": int(steps),
            "lora_rank": int(lora_rank),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
        },
        {
            "recipe": "sampled_longer",
            "steps": int(longer_steps),
            "lora_rank": int(lora_rank),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
        },
        {
            "recipe": "safe_rank_alpha",
            "steps": int(longer_steps),
            "lora_rank": int(safe_lora_rank),
            "lora_alpha": int(safe_lora_alpha),
            "lora_dropout": float(lora_dropout),
        },
    ]
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for attempt in attempts:
        key = (attempt["steps"], attempt["lora_rank"], attempt["lora_alpha"], attempt["lora_dropout"])
        if key not in seen:
            deduped.append(attempt)
            seen.add(key)
    return deduped


def _append_telemetry_csv(path: str | None, metadata: dict[str, Any]) -> None:
    if not path:
        return
    telemetry_path = Path(path).expanduser()
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "knowledge_id",
        "lora_variant",
        "rank",
        "alpha",
        "scale",
        "use_rslora",
        "steps",
        "first_loss",
        "last_loss",
        "l2_norm",
        "attempt_index",
        "attempt_count",
        "quality_attempt_index",
        "quality_attempt_count",
        "final_recipe",
        "accepted",
        "build_degraded",
        "quality_reject_reasons",
        "answer_field",
    ]
    row = {
        "timestamp": int(time.time()),
        "knowledge_id": metadata.get("knowledge_id"),
        "lora_variant": metadata.get("lora_variant"),
        "rank": metadata.get("lora_rank"),
        "alpha": metadata.get("lora_alpha"),
        "scale": metadata.get("lora_scale"),
        "use_rslora": metadata.get("use_rslora"),
        "steps": metadata.get("steps"),
        "first_loss": metadata.get("first_loss"),
        "last_loss": metadata.get("last_loss"),
        "l2_norm": metadata.get("lora_weight_l2_norm"),
        "attempt_index": metadata.get("attempt_index"),
        "attempt_count": metadata.get("attempt_count"),
        "quality_attempt_index": metadata.get("quality_attempt_index"),
        "quality_attempt_count": metadata.get("quality_attempt_count"),
        "final_recipe": metadata.get("final_recipe"),
        "accepted": metadata.get("accepted"),
        "build_degraded": metadata.get("build_degraded"),
        "quality_reject_reasons": "|".join(str(x) for x in metadata.get("quality_reject_reasons", [])),
        "answer_field": metadata.get("answer_field"),
    }
    write_header = not telemetry_path.exists()
    with telemetry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _normalize_lora_variant(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_LORA_VARIANTS:
        raise ValueError(
            "Unsupported LoRA variant. "
            f"Expected one of {SUPPORTED_LORA_VARIANTS}, got {value!r}."
        )
    return normalized


def _parse_build_variants(value: str) -> list[str]:
    variants = [_normalize_lora_variant(item) for item in value.split(",") if item.strip()]
    if not variants:
        raise ValueError("--build-variants must contain at least one LoRA variant.")
    return list(dict.fromkeys(variants))


def resolve_variant_output_dir(output_root: Path, knowledge_id: str, lora_variant: str) -> Path:
    normalized_variant = _normalize_lora_variant(lora_variant)
    if normalized_variant == KNOWLEDGE_LORA_VARIANT:
        return output_root / knowledge_id
    return output_root / normalized_variant / knowledge_id


def _parse_candidate_values(
    value: str | None,
    *,
    cast_fn,
    default: list[Any],
    label: str,
) -> list[Any]:
    if value is None:
        return list(default)
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_items:
        raise ValueError(f"{label} must contain at least one candidate when provided.")
    try:
        return [cast_fn(item) for item in raw_items]
    except ValueError as exc:
        raise ValueError(f"Failed to parse {label}: {value}") from exc


_RANDOM_RECIPE_RNG = None


def _get_random_recipe_rng(randomization_seed):
    """Persistent process-global RNG for non-deterministic recipe sampling.

    - randomization_seed=None -> os entropy (legacy, non-reproducible).
    - otherwise a single seeded RNG is created once per process and advanced
      across builds, so the same knowledge_id can still draw different recipes
      across builds while the whole run stays reproducible given the seed.
      A per-process offset from CUDA_VISIBLE_DEVICES keeps parallel workers'
      streams independent yet reproducible.
    """
    global _RANDOM_RECIPE_RNG
    if randomization_seed is None:
        return random.SystemRandom()
    if _RANDOM_RECIPE_RNG is None:
        offset = 0
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        first = dev.split(",")[0].strip() if dev else ""
        if first:
            try:
                offset = int(first)
            except ValueError:
                offset = abs(hash(first)) % 100000
        _RANDOM_RECIPE_RNG = random.Random(int(randomization_seed) + offset)
    return _RANDOM_RECIPE_RNG


_BUILD_EARLY_STOP_LOSS = None
_BUILD_EARLY_STOP_MIN_STEPS = 8


def _set_build_early_stop(loss, min_steps):
    global _BUILD_EARLY_STOP_LOSS, _BUILD_EARLY_STOP_MIN_STEPS
    _BUILD_EARLY_STOP_LOSS = None if loss is None else float(loss)
    if min_steps is not None:
        _BUILD_EARLY_STOP_MIN_STEPS = max(1, int(min_steps))


def _should_early_stop_build(loss_value, step):
    """Stop a per-adapter build once it has converged.

    Telemetry shows fixed-step builds massively overconverge (median last_loss
    ~5e-4 at 64 steps) when only fact injection is needed; early-stop cuts wasted
    steps. No-op unless --build-early-stop-loss is set.
    """
    return (
        _BUILD_EARLY_STOP_LOSS is not None
        and loss_value <= _BUILD_EARLY_STOP_LOSS
        and (step + 1) >= _BUILD_EARLY_STOP_MIN_STEPS
    )


def resolve_lora_build_spec(
    *,
    knowledge_id: str,
    seed: int,
    randomize_lora_config: bool,
    randomization_base_seed: int | None,
    randomization_mode: str = "random",
    randomization_seed: int | None = None,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_rslora: bool,
    lora_rank_candidates: list[int] | None,
    lora_alpha_candidates: list[int] | None,
    lora_dropout_candidates: list[float] | None,
    target_modules: list[str],
    lora_recipe_pool: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not randomize_lora_config:
        return {
            "seed": seed,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "use_rslora": bool(use_rslora),
            "lora_scale": _compute_lora_scale(
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                use_rslora=use_rslora,
            ),
            "target_modules": list(target_modules),
            "randomized_lora_config": False,
        }

    mode = str(randomization_mode or "random").strip().lower()
    if mode not in {"random", "deterministic"}:
        raise ValueError(f"Unsupported randomization_mode={randomization_mode!r}; expected random or deterministic.")
    base_seed = seed if randomization_base_seed is None else randomization_base_seed
    if mode == "deterministic":
        digest = hashlib.sha256(f"{knowledge_id}:{base_seed}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
    else:
        rng = _get_random_recipe_rng(randomization_seed)
    if lora_recipe_pool:
        recipes = list(lora_recipe_pool)
        recipe = dict(rng.choice(recipes))
        sampled_seed = rng.randrange(0, 2**31 - 1)
        sampled_rank = int(recipe["lora_rank"])
        sampled_alpha = int(recipe["lora_alpha"])
        sampled_dropout = float(recipe.get("lora_dropout", lora_dropout))
        spec = {
            "seed": sampled_seed,
            "lora_rank": sampled_rank,
            "lora_alpha": sampled_alpha,
            "lora_dropout": sampled_dropout,
            "use_rslora": bool(use_rslora),
            "lora_scale": _compute_lora_scale(
                lora_rank=sampled_rank,
                lora_alpha=sampled_alpha,
                use_rslora=use_rslora,
            ),
            "target_modules": list(target_modules),
            "randomized_lora_config": True,
            "randomization_base_seed": int(base_seed) if mode == "deterministic" else None,
            "randomization_mode": mode,
            "lora_recipe_pool_size": len(recipes),
            "lora_recipe_id": recipe.get("recipe_id"),
        }
        if recipe.get("steps") is not None:
            spec["build_steps"] = int(recipe["steps"])
        return spec

    rank_candidates = list(lora_rank_candidates or [lora_rank])
    alpha_candidates = list(lora_alpha_candidates or [lora_alpha])
    dropout_candidates = list(lora_dropout_candidates or [lora_dropout])
    sampled_seed = rng.randrange(0, 2**31 - 1)
    sampled_rank = int(rng.choice(rank_candidates))
    sampled_alpha = int(rng.choice(alpha_candidates))
    return {
        "seed": sampled_seed,
        "lora_rank": sampled_rank,
        "lora_alpha": sampled_alpha,
        "lora_dropout": float(rng.choice(dropout_candidates)),
        "use_rslora": bool(use_rslora),
        "lora_scale": _compute_lora_scale(
            lora_rank=sampled_rank,
            lora_alpha=sampled_alpha,
            use_rslora=use_rslora,
        ),
        "target_modules": list(target_modules),
        "randomized_lora_config": True,
        "randomization_base_seed": int(base_seed) if mode == "deterministic" else None,
        "randomization_mode": mode,
        "lora_rank_candidates": rank_candidates,
        "lora_alpha_candidates": alpha_candidates,
        "lora_dropout_candidates": dropout_candidates,
    }


def load_single_sample(sample_file: Path) -> dict[str, Any]:
    with sample_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a single QA sample dict in {sample_file}, but found {type(payload)!r}.")
    return payload


def adapter_is_complete(output_dir: Path) -> bool:
    return (output_dir / "adapter_config.json").is_file() and (output_dir / "adapter_model.safetensors").is_file()


def load_existing_build_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "knowledge_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def adapter_lock_path(output_dir: Path) -> Path:
    return output_dir.parent / f"{output_dir.name}.lock"


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Different user namespace: treat it as alive.
        return True
    return True


def _try_clear_stale_lock(lock_file: Path, *, min_age_seconds: float = 30.0) -> bool:
    """Best-effort stale-lock cleanup for abruptly terminated builders."""
    try:
        stat_info = lock_file.stat()
    except FileNotFoundError:
        return False
    if (time.time() - stat_info.st_mtime) < max(0.0, min_age_seconds):
        return False

    lock_payload: dict[str, Any] | None = None
    try:
        raw_payload = lock_file.read_text(encoding="utf-8").strip()
        if raw_payload:
            loaded = json.loads(raw_payload)
            if isinstance(loaded, dict):
                lock_payload = loaded
    except Exception:
        lock_payload = None

    stale = False
    pid_value = None
    if isinstance(lock_payload, dict):
        pid_value = lock_payload.get("pid")
        if isinstance(pid_value, int):
            stale = not _pid_exists(pid_value)
        else:
            stale = True
    else:
        stale = True

    if not stale:
        return False

    try:
        lock_file.unlink()
        print(
            f"[ephemeral_lora] Cleared stale lock: {lock_file} "
            f"(pid={pid_value!r}, age_seconds={time.time() - stat_info.st_mtime:.1f})"
        )
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[ephemeral_lora] Failed to clear stale lock {lock_file}: {exc}")
        return False


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _parse_answer_field_candidates(answer_field: Any) -> list[str]:
    if answer_field in (None, "", []):
        return ["answer"]
    if isinstance(answer_field, str):
        candidates = [field.strip() for field in answer_field.split(",")]
    elif isinstance(answer_field, (list, tuple)):
        candidates = [str(field).strip() for field in answer_field]
    else:
        candidates = [str(answer_field).strip()]
    return [field for field in candidates if field] or ["answer"]


def resolve_answer_text(sample: dict[str, Any], *, answer_field: Any = "answer") -> tuple[str, str]:
    candidates = _parse_answer_field_candidates(answer_field)
    for field in candidates:
        value = sample.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, field
    nonempty_keys = sorted(
        str(key)
        for key, value in sample.items()
        if key != ANSWER_FIELD_OVERRIDE_KEY and str(value).strip()
    )
    raise ValueError(
        "No non-empty LoRA supervised target found. "
        f"answer_field_candidates={candidates} available_nonempty_keys={nonempty_keys[:32]}"
    )


def _with_answer_field_override(sample: dict[str, Any], answer_field: Any) -> dict[str, Any]:
    updated = dict(sample)
    updated[ANSWER_FIELD_OVERRIDE_KEY] = answer_field
    return updated


def _sample_answer_field(sample: dict[str, Any]) -> Any:
    return sample.get(ANSWER_FIELD_OVERRIDE_KEY, "answer")


def build_messages(
    sample: dict[str, Any],
    *,
    train_prompt_mode: str = "production",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    answer, _answer_field = resolve_answer_text(sample, answer_field=_sample_answer_field(sample))
    if train_prompt_mode == "qwen_bare":
        prompt_messages = [{"role": "user", "content": str(sample.get("question", "")).strip()}]
        return prompt_messages, prompt_messages + [{"role": "assistant", "content": answer}]
    if train_prompt_mode == "qwen_qa":
        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(sample.get("question", "")).strip()},
        ]
        return prompt_messages, prompt_messages + [{"role": "assistant", "content": answer}]
    if train_prompt_mode != "production":
        raise ValueError(f"Unsupported train_prompt_mode={train_prompt_mode!r}")

    user_prompt = DEFAULT_USER_TEMPLATE.format(
        title=sample.get("title", ""),
        category=sample.get("category", ""),
        subcategory=sample.get("subcategory", ""),
        context=sample.get("context", ""),
        question=sample.get("question", ""),
    ).strip()
    prompt_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": answer}]
    return prompt_messages, full_messages


def build_training_samples(sample: dict[str, Any]) -> list[dict[str, Any]]:
    variants = sample.get("variants")
    if isinstance(variants, list):
        training_samples: list[dict[str, Any]] = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            question = str(variant.get("question", "")).strip()
            answer = str(variant.get("answer", "")).strip()
            if question and answer:
                merged = dict(sample)
                merged.update(variant)
                merged.pop("variants", None)
                training_samples.append(merged)
        if training_samples:
            return training_samples
    return [sample]


def _as_token_id_list(value) -> list[int]:
    """Normalize tokenizer outputs to a flat list of token ids."""
    if hasattr(value, "ids"):
        value = value.ids
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"Expected one tokenized sequence, got batch size {len(value)}")
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Unsupported tokenized output type: {type(value)!r}")
    return [int(x) for x in value]


def _apply_chat_template_no_thinking(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        output = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        output = tokenizer.apply_chat_template(messages, **kwargs)
    return _as_token_id_list(output)

def build_supervised_tensors(
    tokenizer,
    prompt_messages: list[dict[str, str]],
    full_messages: list[dict[str, str]],
    max_length: int,
    device: torch.device,
    *,
    chat_target_mode: str = "full_message",
    answer: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = _apply_chat_template_no_thinking(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    if chat_target_mode == "answer_only":
        if answer is None:
            answer = str(full_messages[-1].get("content", "") if full_messages else "").strip()
        assistant_ids = _as_token_id_list(tokenizer(str(answer).strip(), add_special_tokens=False))
        prompt_prefix_ids = prompt_ids
    elif chat_target_mode == "full_message":
        full_ids = _apply_chat_template_no_thinking(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )
        prompt_prefix_length = _shared_prefix_length(prompt_ids, full_ids)
        prompt_prefix_ids = full_ids[:prompt_prefix_length]
        assistant_ids = full_ids[prompt_prefix_length:]
    else:
        raise ValueError(f"Unsupported chat_target_mode={chat_target_mode!r}")
    if not assistant_ids:
        raise ValueError(
            "No supervised assistant tokens remain after prompt masking. "
            f"prompt_tokens={len(prompt_ids)} target_mode={chat_target_mode}"
        )
    if len(assistant_ids) > max_length:
        raise ValueError(
            "Assistant completion exceeds max_length even without prompt context. "
            f"assistant_tokens={len(assistant_ids)} max_length={max_length}"
        )

    available_prompt_tokens = max_length - len(assistant_ids)
    truncated_prompt_ids = prompt_prefix_ids[-available_prompt_tokens:] if available_prompt_tokens > 0 else []
    input_ids = truncated_prompt_ids + assistant_ids
    labels = [-100] * len(truncated_prompt_ids) + assistant_ids
    attention_mask = [1] * len(input_ids)

    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)
    labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    return input_ids_tensor, attention_mask_tensor, labels_tensor


def build_supervised_tensor_batch(
    tokenizer,
    sample: dict[str, Any],
    max_length: int,
    device: torch.device,
    *,
    train_prompt_mode: str,
    chat_target_mode: str,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    answer_field = _sample_answer_field(sample)
    for training_sample in build_training_samples(sample):
        if ANSWER_FIELD_OVERRIDE_KEY not in training_sample:
            training_sample = _with_answer_field_override(training_sample, answer_field)
        target_answer, _target_answer_field = resolve_answer_text(training_sample, answer_field=answer_field)
        prompt_messages, full_messages = build_messages(training_sample, train_prompt_mode=train_prompt_mode)
        tensors.append(
            build_supervised_tensors(
                tokenizer=tokenizer,
                prompt_messages=prompt_messages,
                full_messages=full_messages,
                max_length=max_length,
                device=device,
                chat_target_mode=chat_target_mode,
                answer=target_answer,
            )
        )
    if not tensors:
        raise ValueError("No supervised tensors could be built from sample or variants.")
    return tensors


def _shared_prefix_length(left: list[int], right: list[int]) -> int:
    prefix_length = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        prefix_length += 1
    return prefix_length


def _load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_model_with_trainable_adapter(
    *,
    model_path: str,
    base_lora_adapter_path: str | None,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_rslora: bool,
    target_modules: list[str],
    dtype: torch.dtype,
    device: torch.device,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.config.use_cache = False

    peft_config = _make_lora_config(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        target_modules=target_modules,
    )
    trainable_adapter_name = "default"
    if base_lora_adapter_path not in (None, "", "None"):
        model = PeftModel.from_pretrained(model, base_lora_adapter_path, is_trainable=False)
        trainable_adapter_name = "ephemeral"
        model.add_adapter(trainable_adapter_name, peft_config)
        model.set_adapter(trainable_adapter_name)
    else:
        model = get_peft_model(model, peft_config)
    return model, trainable_adapter_name


def _make_lora_config(
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_rslora: bool,
    target_modules: list[str],
) -> LoraConfig:
    kwargs = {
        "task_type": TaskType.CAUSAL_LM,
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": "none",
    }
    if use_rslora:
        kwargs["use_rslora"] = True
    try:
        return LoraConfig(**kwargs)
    except TypeError as exc:
        if use_rslora:
            raise TypeError("Installed PEFT does not support use_rslora=True.") from exc
        raise


class PersistentAdapterBuilder:
    """Build many ephemeral adapters while reusing one loaded base model."""

    def __init__(
        self,
        *,
        model_path: str,
        base_lora_adapter_path: str | None,
        dtype: torch.dtype,
        anchor_target_modules: list[str],
    ) -> None:
        self.model_path = model_path
        self.base_lora_adapter_path = base_lora_adapter_path
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = _load_tokenizer(model_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        base_model.to(self.device)
        base_model.config.use_cache = False
        try:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            base_model.gradient_checkpointing_enable()
            if hasattr(base_model, "enable_input_require_grads"):
                base_model.enable_input_require_grads()
        print(
            "[ephemeral_lora_worker] activation checkpointing enabled",
            file=sys.stderr,
            flush=True,
        )

        self.anchor_adapter_name = "_persistent_anchor"
        if base_lora_adapter_path not in (None, "", "None"):
            self.model = PeftModel.from_pretrained(base_model, base_lora_adapter_path, is_trainable=False)
            self.anchor_adapter_name = "default"
        else:
            anchor_config = _make_lora_config(
                lora_rank=1,
                lora_alpha=1,
                lora_dropout=0.0,
                use_rslora=False,
                target_modules=anchor_target_modules,
            )
            self.model = get_peft_model(base_model, anchor_config, adapter_name=self.anchor_adapter_name)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def build_adapter(
        self,
        *,
        sample: dict[str, Any],
        knowledge_id: str,
        output_dir: Path,
        steps: int,
        learning_rate: float,
        fallback_learning_rate: float,
        max_length: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        use_rslora: bool,
        target_modules: list[str],
        seed: int,
        fallback_dtype: torch.dtype | None,
        gradient_clip_norm: float | None,
        lora_variant: str,
        train_prompt_mode: str,
        chat_target_mode: str,
        quality_max_last_loss: float | None = None,
        quality_min_l2_norm: float | None = None,
        quality_max_l2_norm: float | None = None,
        quality_retry_step_multiplier: float = 2.0,
        quality_safe_lora_rank: int = 256,
        quality_safe_lora_alpha: int = 32,
        telemetry_csv: str | None = None,
    ) -> dict[str, Any]:
        if fallback_dtype not in (None, self.dtype):
            print(
                "[ephemeral_lora_worker] dtype fallback is disabled in persistent mode "
                f"(loaded_dtype={self.dtype}, requested_fallback={fallback_dtype}).",
                file=sys.stderr,
                flush=True,
            )
        attempts = [(learning_rate, False)]
        if fallback_learning_rate != learning_rate:
            attempts.append((fallback_learning_rate, True))

        quality_recipes = _quality_recipe_attempts(
            steps=steps,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            retry_step_multiplier=quality_retry_step_multiplier,
            safe_lora_rank=quality_safe_lora_rank,
            safe_lora_alpha=quality_safe_lora_alpha,
        )

        errors: list[str] = []
        for quality_index, recipe in enumerate(quality_recipes, start=1):
            is_final_quality_recipe = quality_index == len(quality_recipes)
            for attempt_index, (attempt_lr, fallback_used) in enumerate(attempts, start=1):
                adapter_name = f"ephemeral_{hashlib.sha1(f'{knowledge_id}:{lora_variant}:{quality_index}:{attempt_index}:{time.time_ns()}'.encode()).hexdigest()[:16]}"
                try:
                    _cleanup_adapter_dir(output_dir)
                    metadata = self._build_once(
                        sample=sample,
                        knowledge_id=knowledge_id,
                        output_dir=output_dir,
                        steps=int(recipe["steps"]),
                        learning_rate=attempt_lr,
                        max_length=max_length,
                        lora_rank=int(recipe["lora_rank"]),
                        lora_alpha=int(recipe["lora_alpha"]),
                        lora_dropout=float(recipe["lora_dropout"]),
                        use_rslora=use_rslora,
                        target_modules=target_modules,
                        seed=seed,
                        gradient_clip_norm=gradient_clip_norm,
                        lora_variant=lora_variant,
                        train_prompt_mode=train_prompt_mode,
                        chat_target_mode=chat_target_mode,
                        adapter_name=adapter_name,
                    )
                    accepted, reject_reasons = _quality_acceptance(
                        metadata,
                        quality_max_last_loss=quality_max_last_loss,
                        quality_min_l2_norm=quality_min_l2_norm,
                        quality_max_l2_norm=quality_max_l2_norm,
                    )
                    metadata.update(
                        {
                            "attempt_index": attempt_index,
                            "attempt_count": len(attempts),
                            "fallback_used": fallback_used,
                            "quality_attempt_index": quality_index,
                            "quality_attempt_count": len(quality_recipes),
                            "final_recipe": recipe["recipe"],
                            "accepted": accepted,
                            "build_degraded": bool(not accepted and is_final_quality_recipe),
                            "quality_reject_reasons": reject_reasons,
                            "build_backend": "persistent_worker",
                            "model_load_reused": True,
                        }
                    )
                    _write_build_metadata(output_dir, metadata)
                    _append_telemetry_csv(telemetry_csv, metadata)
                    if accepted or is_final_quality_recipe:
                        return metadata
                    _cleanup_adapter_dir(output_dir)
                except RuntimeError as exc:
                    errors.append(
                        f"quality_attempt={quality_index} recipe={recipe['recipe']} "
                        f"attempt={attempt_index} dtype={self.dtype} lr={attempt_lr}: {exc}"
                    )
                    _cleanup_adapter_dir(output_dir)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    self._delete_adapter_if_present(adapter_name)

        raise RuntimeError(f"Failed to build adapter for {knowledge_id}. Attempts: {' | '.join(errors)}")

    def _build_once(
        self,
        *,
        sample: dict[str, Any],
        knowledge_id: str,
        output_dir: Path,
        steps: int,
        learning_rate: float,
        max_length: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        use_rslora: bool,
        target_modules: list[str],
        seed: int,
        gradient_clip_norm: float | None,
        lora_variant: str,
        train_prompt_mode: str,
        chat_target_mode: str,
        adapter_name: str,
    ) -> dict[str, Any]:
        _seed_everything(seed)
        normalized_variant = _normalize_lora_variant(lora_variant)
        self.model.add_adapter(
            adapter_name,
            _make_lora_config(
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=use_rslora,
                target_modules=target_modules,
            ),
        )
        self.model.set_adapter(adapter_name)
        self.model.train()
        trainable_params = self._trainable_adapter_params(adapter_name)

        if normalized_variant == NO_OP_LORA_VARIANT:
            with torch.no_grad():
                for parameter in trainable_params:
                    parameter.zero_()
            first_loss_value = None
            last_loss_value = None
        elif normalized_variant == RANDOM_LORA_VARIANT:
            _randomize_trainable_parameters(trainable_params)
            first_loss_value = None
            last_loss_value = None
        else:
            tensor_batch = build_supervised_tensor_batch(
                tokenizer=self.tokenizer,
                sample=sample,
                max_length=max_length,
                device=self.device,
                train_prompt_mode=train_prompt_mode,
                chat_target_mode=chat_target_mode,
            )
            optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
            first_loss_value = None
            last_loss_value = None
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for input_ids, attention_mask, labels in tensor_batch:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    losses.append(outputs.loss)
                loss = torch.stack(losses).mean()
                loss_value = loss.detach().float().item()
                if first_loss_value is None:
                    first_loss_value = loss_value
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite LoRA training loss for {knowledge_id}: {loss_value}")
                loss.backward()
                non_finite_grad_names = _find_non_finite_gradients(self.model)
                if non_finite_grad_names:
                    raise RuntimeError(
                        "Non-finite LoRA gradients for "
                        f"{knowledge_id}: {', '.join(non_finite_grad_names[:8])}"
                    )
                if gradient_clip_norm is not None and gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)
                optimizer.step()
                last_loss_value = loss_value
                print(f"[{knowledge_id}] step={step + 1}/{steps} loss={loss_value:.6f}", flush=True)
                if _should_early_stop_build(loss_value, step):
                    print(
                        f"[{knowledge_id}] early-stop at step={step + 1}/{steps} "
                        f"loss={loss_value:.6f}<= {_BUILD_EARLY_STOP_LOSS}",
                        flush=True,
                    )
                    break

        metadata_sample = build_training_samples(sample)[0]
        metadata_answer, metadata_answer_field = resolve_answer_text(
            metadata_sample,
            answer_field=_sample_answer_field(sample),
        )
        metadata = {
            "knowledge_id": knowledge_id,
            "lora_variant": normalized_variant,
            "question": metadata_sample.get("question", ""),
            "answer": metadata_answer,
            "source_answer": metadata_sample.get("answer", ""),
            "answer_field": metadata_answer_field,
            "answer_field_candidates": _parse_answer_field_candidates(_sample_answer_field(sample)),
            "title": metadata_sample.get("title", ""),
            "category": metadata_sample.get("category", ""),
            "steps": steps if normalized_variant == KNOWLEDGE_LORA_VARIANT else 0,
            "learning_rate": learning_rate if normalized_variant == KNOWLEDGE_LORA_VARIANT else 0.0,
            "dtype": str(self.dtype).replace("torch.", ""),
            "seed": seed,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "use_rslora": bool(use_rslora),
            "lora_scale": _compute_lora_scale(
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                use_rslora=use_rslora,
            ),
            "target_modules": target_modules,
            "train_prompt_mode": train_prompt_mode,
            "chat_target_mode": chat_target_mode,
            "training_variant_count": len(build_training_samples(sample)),
            "base_lora_adapter_path": self.base_lora_adapter_path,
            "adapter_initialization": (
                "zeros"
                if normalized_variant == NO_OP_LORA_VARIANT
                else "random"
                if normalized_variant == RANDOM_LORA_VARIANT
                else None
            ),
            "first_loss": first_loss_value,
            "last_loss": last_loss_value,
        }
        metadata["lora_weight_l2_norm"] = _compute_trainable_lora_weight_l2_norm(self.model)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir, safe_serialization=True, selected_adapters=[adapter_name])
        nested_adapter_dir = output_dir / adapter_name
        if not adapter_is_complete(output_dir) and nested_adapter_dir.is_dir():
            for nested_file in nested_adapter_dir.iterdir():
                target_file = output_dir / nested_file.name
                if target_file.exists():
                    if target_file.is_dir():
                        shutil.rmtree(target_file)
                    else:
                        target_file.unlink()
                shutil.move(os.fspath(nested_file), os.fspath(target_file))
            with contextlib.suppress(OSError):
                nested_adapter_dir.rmdir()
        _write_build_metadata(output_dir, metadata)
        return metadata

    def _trainable_adapter_params(self, adapter_name: str) -> list[torch.nn.Parameter]:
        trainable_params: list[torch.nn.Parameter] = []
        marker = f".{adapter_name}."
        for name, parameter in self.model.named_parameters():
            should_train = marker in name
            parameter.requires_grad_(should_train)
            if should_train:
                trainable_params.append(parameter)
        if not trainable_params:
            raise RuntimeError(f"No trainable parameters found for adapter {adapter_name!r}.")
        return trainable_params

    def _delete_adapter_if_present(self, adapter_name: str) -> None:
        try:
            if adapter_name in getattr(self.model, "peft_config", {}):
                self.model.delete_adapter(adapter_name)
        finally:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            try:
                if self.anchor_adapter_name in getattr(self.model, "peft_config", {}):
                    self.model.set_adapter(self.anchor_adapter_name)
            except Exception:
                pass
            self.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _save_adapter_metadata_and_cleanup(
    *,
    model,
    output_dir: Path,
    trainable_adapter_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata["lora_weight_l2_norm"] = _compute_trainable_lora_weight_l2_norm(model)
    model.save_pretrained(output_dir, safe_serialization=True, selected_adapters=[trainable_adapter_name])
    _write_build_metadata(output_dir, metadata)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def _compute_trainable_lora_weight_l2_norm(model) -> float:
    squared_norm = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            squared_norm += float(torch.sum(parameter.detach().float() ** 2).item())
    return squared_norm**0.5


def _randomize_trainable_parameters(parameters) -> None:
    with torch.no_grad():
        for parameter in parameters:
            if not getattr(parameter, "requires_grad", False):
                continue
            if not torch.is_floating_point(parameter):
                continue
            if parameter.dim() >= 2:
                torch.nn.init.kaiming_uniform_(parameter, a=5**0.5)
            else:
                parameter.normal_(mean=0.0, std=0.02)
            zero_mask = parameter == 0
            if bool(zero_mask.any().item()):
                parameter.masked_fill_(zero_mask, 1e-6)


def build_initialized_adapter(
    *,
    model_path: str,
    base_lora_adapter_path: str | None,
    knowledge_id: str,
    lora_variant: str,
    output_dir: Path,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_rslora: bool,
    target_modules: list[str],
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    normalized_variant = _normalize_lora_variant(lora_variant)
    if normalized_variant not in (NO_OP_LORA_VARIANT, RANDOM_LORA_VARIANT):
        raise ValueError(f"build_initialized_adapter does not support variant {lora_variant!r}.")

    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainable_adapter_name = _load_model_with_trainable_adapter(
        model_path=model_path,
        base_lora_adapter_path=base_lora_adapter_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        target_modules=target_modules,
        dtype=dtype,
        device=device,
    )
    if normalized_variant == NO_OP_LORA_VARIANT:
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.zero_()
    elif normalized_variant == RANDOM_LORA_VARIANT:
        _randomize_trainable_parameters(parameter for parameter in model.parameters() if parameter.requires_grad)

    metadata = {
        "knowledge_id": knowledge_id,
        "lora_variant": normalized_variant,
        "steps": 0,
        "learning_rate": 0.0,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_rslora": bool(use_rslora),
        "lora_scale": _compute_lora_scale(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_rslora=use_rslora,
        ),
        "target_modules": target_modules,
        "base_lora_adapter_path": base_lora_adapter_path,
        "adapter_initialization": "zeros" if normalized_variant == NO_OP_LORA_VARIANT else "random",
        "first_loss": None,
        "last_loss": None,
    }
    return _save_adapter_metadata_and_cleanup(
        model=model,
        output_dir=output_dir,
        trainable_adapter_name=trainable_adapter_name,
        metadata=metadata,
    )


def train_single_adapter(
    *,
    model_path: str,
    base_lora_adapter_path: str | None,
    sample: dict[str, Any],
    knowledge_id: str,
    output_dir: Path,
    steps: int,
    learning_rate: float,
    max_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_rslora: bool,
    target_modules: list[str],
    dtype: torch.dtype,
    gradient_clip_norm: float | None,
    seed: int,
    lora_variant: str = KNOWLEDGE_LORA_VARIANT,
    train_prompt_mode: str = "qwen_bare",
    chat_target_mode: str = "answer_only",
) -> dict[str, Any]:
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(model_path)
    model, trainable_adapter_name = _load_model_with_trainable_adapter(
        model_path=model_path,
        base_lora_adapter_path=base_lora_adapter_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        target_modules=target_modules,
        dtype=dtype,
        device=device,
    )
    model.train()
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    tensor_batch = build_supervised_tensor_batch(
        tokenizer=tokenizer,
        sample=sample,
        max_length=max_length,
        device=device,
        train_prompt_mode=train_prompt_mode,
        chat_target_mode=chat_target_mode,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    first_loss_value: float | None = None
    last_loss_value: float | None = None

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for input_ids, attention_mask, labels in tensor_batch:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss)
        loss = torch.stack(losses).mean()
        loss_value = loss.detach().float().item()
        if first_loss_value is None:
            first_loss_value = loss_value
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite LoRA training loss for {knowledge_id}: {loss_value}")
        loss.backward()
        non_finite_grad_names = _find_non_finite_gradients(model)
        if non_finite_grad_names:
            raise RuntimeError(
                "Non-finite LoRA gradients for "
                f"{knowledge_id}: {', '.join(non_finite_grad_names[:8])}"
            )
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)
        optimizer.step()
        last_loss_value = loss_value
        print(f"[{knowledge_id}] step={step + 1}/{steps} loss={loss_value:.6f}", flush=True)
        if _should_early_stop_build(loss_value, step):
            print(
                f"[{knowledge_id}] early-stop at step={step + 1}/{steps} "
                f"loss={loss_value:.6f}<= {_BUILD_EARLY_STOP_LOSS}",
                flush=True,
            )
            break

    metadata_sample = build_training_samples(sample)[0]
    metadata_answer, metadata_answer_field = resolve_answer_text(
        metadata_sample,
        answer_field=_sample_answer_field(sample),
    )
    metadata = {
        "knowledge_id": knowledge_id,
        "lora_variant": _normalize_lora_variant(lora_variant),
        "question": metadata_sample.get("question", ""),
        "answer": metadata_answer,
        "source_answer": metadata_sample.get("answer", ""),
        "answer_field": metadata_answer_field,
        "answer_field_candidates": _parse_answer_field_candidates(_sample_answer_field(sample)),
        "title": metadata_sample.get("title", ""),
        "category": metadata_sample.get("category", ""),
        "steps": steps,
        "learning_rate": learning_rate,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_rslora": bool(use_rslora),
        "lora_scale": _compute_lora_scale(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_rslora=use_rslora,
        ),
        "target_modules": target_modules,
        "train_prompt_mode": train_prompt_mode,
        "chat_target_mode": chat_target_mode,
        "training_variant_count": len(build_training_samples(sample)),
        "base_lora_adapter_path": base_lora_adapter_path,
        "first_loss": first_loss_value,
        "last_loss": last_loss_value,
    }
    return _save_adapter_metadata_and_cleanup(
        model=model,
        output_dir=output_dir,
        trainable_adapter_name=trainable_adapter_name,
        metadata=metadata,
    )


def _cleanup_adapter_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _find_non_finite_gradients(model) -> list[str]:
    bad_param_names: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad_param_names.append(name)
    return bad_param_names


def train_single_adapter_with_retries(
    *,
    model_path: str,
    base_lora_adapter_path: str | None,
    sample: dict[str, Any],
    knowledge_id: str,
    output_dir: Path,
    steps: int,
    learning_rate: float,
    fallback_learning_rate: float,
    max_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_rslora: bool,
    target_modules: list[str],
    seed: int,
    preferred_dtype: torch.dtype,
    fallback_dtype: torch.dtype | None,
    gradient_clip_norm: float | None,
    lora_variant: str = KNOWLEDGE_LORA_VARIANT,
    train_prompt_mode: str = "qwen_bare",
    chat_target_mode: str = "answer_only",
    quality_max_last_loss: float | None = None,
    quality_min_l2_norm: float | None = None,
    quality_max_l2_norm: float | None = None,
    quality_retry_step_multiplier: float = 2.0,
    quality_safe_lora_rank: int = 256,
    quality_safe_lora_alpha: int = 32,
    telemetry_csv: str | None = None,
) -> dict[str, Any]:
    attempts: list[tuple[torch.dtype, float]] = []
    for candidate_attempt in (
        (preferred_dtype, learning_rate),
        (preferred_dtype, fallback_learning_rate),
        (fallback_dtype, fallback_learning_rate) if fallback_dtype is not None else None,
    ):
        if candidate_attempt is None:
            continue
        if candidate_attempt not in attempts:
            attempts.append(candidate_attempt)

    quality_recipes = _quality_recipe_attempts(
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        retry_step_multiplier=quality_retry_step_multiplier,
        safe_lora_rank=quality_safe_lora_rank,
        safe_lora_alpha=quality_safe_lora_alpha,
    )

    errors: list[str] = []
    for quality_index, recipe in enumerate(quality_recipes, start=1):
        is_final_quality_recipe = quality_index == len(quality_recipes)
        for attempt_index, (attempt_dtype, attempt_lr) in enumerate(attempts, start=1):
            try:
                _cleanup_adapter_dir(output_dir)
                build_metadata = train_single_adapter(
                    model_path=model_path,
                    base_lora_adapter_path=base_lora_adapter_path,
                    sample=sample,
                    knowledge_id=knowledge_id,
                    output_dir=output_dir,
                    steps=int(recipe["steps"]),
                    learning_rate=attempt_lr,
                    max_length=max_length,
                    lora_rank=int(recipe["lora_rank"]),
                    lora_alpha=int(recipe["lora_alpha"]),
                    lora_dropout=float(recipe["lora_dropout"]),
                    use_rslora=use_rslora,
                    target_modules=target_modules,
                    dtype=attempt_dtype,
                    gradient_clip_norm=gradient_clip_norm,
                    seed=seed,
                    lora_variant=lora_variant,
                    train_prompt_mode=train_prompt_mode,
                    chat_target_mode=chat_target_mode,
                )
                accepted, reject_reasons = _quality_acceptance(
                    build_metadata,
                    quality_max_last_loss=quality_max_last_loss,
                    quality_min_l2_norm=quality_min_l2_norm,
                    quality_max_l2_norm=quality_max_l2_norm,
                )
                build_metadata.update(
                    {
                        "attempt_index": attempt_index,
                        "attempt_count": len(attempts),
                        "fallback_used": attempt_index > 1,
                        "quality_attempt_index": quality_index,
                        "quality_attempt_count": len(quality_recipes),
                        "final_recipe": recipe["recipe"],
                        "accepted": accepted,
                        "build_degraded": bool(not accepted and is_final_quality_recipe),
                        "quality_reject_reasons": reject_reasons,
                    }
                )
                _write_build_metadata(output_dir, build_metadata)
                _append_telemetry_csv(telemetry_csv, build_metadata)
                if accepted or is_final_quality_recipe:
                    return build_metadata
                _cleanup_adapter_dir(output_dir)
            except RuntimeError as exc:
                errors.append(
                    f"quality_attempt={quality_index} recipe={recipe['recipe']} "
                    f"attempt={attempt_index} dtype={attempt_dtype} lr={attempt_lr}: {exc}"
                )
                _cleanup_adapter_dir(output_dir)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raise RuntimeError(f"Failed to build adapter for {knowledge_id}. Attempts: {' | '.join(errors)}")


def build_single_sample_with_lock(
    *,
    model_path: str,
    base_lora_adapter_path: str | None,
    sample: dict[str, Any],
    knowledge_id: str,
    output_dir: Path,
    steps: int,
    learning_rate: float,
    fallback_learning_rate: float,
    max_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_rslora: bool,
    target_modules: list[str],
    seed: int,
    preferred_dtype: torch.dtype,
    fallback_dtype: torch.dtype | None,
    gradient_clip_norm: float | None,
    lora_variant: str,
    train_prompt_mode: str,
    chat_target_mode: str,
    lock_timeout_seconds: float,
    lock_poll_interval_seconds: float,
    loaded_builder: PersistentAdapterBuilder | None = None,
    quality_max_last_loss: float | None = None,
    quality_min_l2_norm: float | None = None,
    quality_max_l2_norm: float | None = None,
    quality_retry_step_multiplier: float = 2.0,
    quality_safe_lora_rank: int = 256,
    quality_safe_lora_alpha: int = 32,
    telemetry_csv: str | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_file = adapter_lock_path(output_dir)
    poll_interval_seconds = max(0.1, lock_poll_interval_seconds)
    deadline = time.monotonic() + max(lock_timeout_seconds, poll_interval_seconds)

    while True:
        if adapter_is_complete(output_dir):
            return load_existing_build_metadata(output_dir)

        try:
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _try_clear_stale_lock(
                lock_file,
                min_age_seconds=max(30.0, poll_interval_seconds * 3.0),
            ):
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for an existing ephemeral LoRA build lock to clear: "
                    f"knowledge_id={knowledge_id} lock={lock_file}"
                )
            time.sleep(poll_interval_seconds)
            continue

        try:
            with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "knowledge_id": knowledge_id}, ensure_ascii=False))

            if adapter_is_complete(output_dir):
                return load_existing_build_metadata(output_dir)

            normalized_variant = _normalize_lora_variant(lora_variant)
            if loaded_builder is not None:
                return loaded_builder.build_adapter(
                    sample=sample,
                    knowledge_id=knowledge_id,
                    output_dir=output_dir,
                    steps=steps,
                    learning_rate=learning_rate,
                    fallback_learning_rate=fallback_learning_rate,
                    max_length=max_length,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    use_rslora=use_rslora,
                    target_modules=target_modules,
                    seed=seed,
                    fallback_dtype=fallback_dtype,
                    gradient_clip_norm=gradient_clip_norm,
                    lora_variant=normalized_variant,
                    train_prompt_mode=train_prompt_mode,
                    chat_target_mode=chat_target_mode,
                    quality_max_last_loss=quality_max_last_loss,
                    quality_min_l2_norm=quality_min_l2_norm,
                    quality_max_l2_norm=quality_max_l2_norm,
                    quality_retry_step_multiplier=quality_retry_step_multiplier,
                    quality_safe_lora_rank=quality_safe_lora_rank,
                    quality_safe_lora_alpha=quality_safe_lora_alpha,
                    telemetry_csv=telemetry_csv,
                )
            if normalized_variant == KNOWLEDGE_LORA_VARIANT:
                return train_single_adapter_with_retries(
                    model_path=model_path,
                    base_lora_adapter_path=base_lora_adapter_path,
                    sample=sample,
                    knowledge_id=knowledge_id,
                    output_dir=output_dir,
                    steps=steps,
                    learning_rate=learning_rate,
                    fallback_learning_rate=fallback_learning_rate,
                    max_length=max_length,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    use_rslora=use_rslora,
                    target_modules=target_modules,
                    seed=seed,
                    preferred_dtype=preferred_dtype,
                    fallback_dtype=fallback_dtype,
                    gradient_clip_norm=gradient_clip_norm,
                    lora_variant=normalized_variant,
                    train_prompt_mode=train_prompt_mode,
                    chat_target_mode=chat_target_mode,
                    quality_max_last_loss=quality_max_last_loss,
                    quality_min_l2_norm=quality_min_l2_norm,
                    quality_max_l2_norm=quality_max_l2_norm,
                    quality_retry_step_multiplier=quality_retry_step_multiplier,
                    quality_safe_lora_rank=quality_safe_lora_rank,
                    quality_safe_lora_alpha=quality_safe_lora_alpha,
                    telemetry_csv=telemetry_csv,
                )
            return build_initialized_adapter(
                model_path=model_path,
                base_lora_adapter_path=base_lora_adapter_path,
                knowledge_id=knowledge_id,
                lora_variant=normalized_variant,
                output_dir=output_dir,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=use_rslora,
                target_modules=target_modules,
                dtype=preferred_dtype,
                seed=seed,
            )
        finally:
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass


def build_single_sample_from_args(
    *,
    args: argparse.Namespace,
    dtype: torch.dtype,
    fallback_dtype: torch.dtype | None,
    fallback_learning_rate: float,
    target_modules: list[str],
) -> dict[str, Any]:
    sample_file = Path(args.sample_file).expanduser().resolve()
    sample = _with_answer_field_override(load_single_sample(sample_file), args.answer_field)
    knowledge_id = args.knowledge_id or build_knowledge_id(sample, qa_index=0)
    output_dir = Path(args.output_dir).expanduser().resolve()
    lora_variant = _normalize_lora_variant(args.lora_variant)
    build_spec = resolve_lora_build_spec(
        knowledge_id=f"{knowledge_id}:{lora_variant}",
        seed=args.seed,
        randomize_lora_config=args.randomize_lora_config,
        randomization_base_seed=args.randomization_base_seed,
        randomization_mode=args.randomization_mode,
        randomization_seed=args.randomization_seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
        lora_rank_candidates=_parse_candidate_values(
            args.lora_rank_candidates,
            cast_fn=int,
            default=[args.lora_rank],
            label="--lora-rank-candidates",
        ),
        lora_alpha_candidates=_parse_candidate_values(
            args.lora_alpha_candidates,
            cast_fn=int,
            default=[args.lora_alpha],
            label="--lora-alpha-candidates",
        ),
        lora_dropout_candidates=_parse_candidate_values(
            args.lora_dropout_candidates,
            cast_fn=float,
            default=[args.lora_dropout],
            label="--lora-dropout-candidates",
        ),
        target_modules=target_modules,
        lora_recipe_pool=_load_lora_recipe_pool(args.lora_recipe_pool),
    )
    build_metadata = build_single_sample_with_lock(
        model_path=args.model_path,
        base_lora_adapter_path=args.base_lora_adapter_path,
        sample=sample,
        knowledge_id=knowledge_id,
        output_dir=output_dir,
        steps=int(build_spec.get("build_steps", args.steps)),
        learning_rate=args.learning_rate,
        fallback_learning_rate=fallback_learning_rate,
        max_length=args.max_length,
        lora_rank=build_spec["lora_rank"],
        lora_alpha=build_spec["lora_alpha"],
        lora_dropout=build_spec["lora_dropout"],
        use_rslora=build_spec["use_rslora"],
        target_modules=build_spec["target_modules"],
        seed=build_spec["seed"],
        preferred_dtype=dtype,
        fallback_dtype=fallback_dtype,
        gradient_clip_norm=args.gradient_clip_norm,
        lora_variant=lora_variant,
        train_prompt_mode=args.train_prompt_mode,
        chat_target_mode=args.chat_target_mode,
        lock_timeout_seconds=args.lock_timeout_seconds,
        lock_poll_interval_seconds=args.lock_poll_interval_seconds,
        quality_max_last_loss=args.quality_max_last_loss,
        quality_min_l2_norm=args.quality_min_l2_norm,
        quality_max_l2_norm=args.quality_max_l2_norm,
        quality_retry_step_multiplier=args.quality_retry_step_multiplier,
        quality_safe_lora_rank=args.quality_safe_lora_rank,
        quality_safe_lora_alpha=args.quality_safe_lora_alpha,
        telemetry_csv=args.telemetry_csv,
    )
    build_metadata.update(
        {
            "lora_variant": lora_variant,
            "randomized_lora_config": build_spec["randomized_lora_config"],
            "randomization_base_seed": build_spec.get("randomization_base_seed"),
            "lora_rank_candidates": build_spec.get("lora_rank_candidates"),
            "lora_alpha_candidates": build_spec.get("lora_alpha_candidates"),
            "lora_dropout_candidates": build_spec.get("lora_dropout_candidates"),
            "use_rslora": build_spec.get("use_rslora"),
            "lora_scale": build_spec.get("lora_scale"),
            "lora_recipe_pool_size": build_spec.get("lora_recipe_pool_size"),
            "lora_recipe_id": build_spec.get("lora_recipe_id"),
            "build_steps": build_spec.get("build_steps"),
        }
    )
    _write_build_metadata(output_dir, build_metadata)
    return build_metadata


def run_worker_jsonl(args: argparse.Namespace) -> None:
    _set_build_early_stop(args.build_early_stop_loss, args.build_early_stop_min_steps)
    dtype = resolve_dtype(args.dtype)
    fallback_dtype = None if args.fallback_dtype == "none" else resolve_dtype(args.fallback_dtype)
    fallback_learning_rate = args.fallback_learning_rate or max(args.learning_rate / 5, 1e-6)
    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    builder = PersistentAdapterBuilder(
        model_path=args.model_path,
        base_lora_adapter_path=args.base_lora_adapter_path,
        dtype=dtype,
        anchor_target_modules=target_modules,
    )

    stdout = sys.stdout
    print(
        json.dumps(
            {
                "ok": True,
                "event": "ready",
                "dtype": str(dtype).replace("torch.", ""),
                "device": str(builder.device),
            },
            ensure_ascii=False,
        ),
        file=stdout,
        flush=True,
    )
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
            if request.get("command") == "shutdown":
                print(json.dumps({"ok": True, "event": "shutdown"}, ensure_ascii=False), file=stdout, flush=True)
                return

            with contextlib.redirect_stdout(sys.stderr):
                build_metadata = build_single_sample_with_lock(
                    model_path=args.model_path,
                    base_lora_adapter_path=args.base_lora_adapter_path,
                    sample=_with_answer_field_override(
                        request["sample"],
                        request.get("answer_field", args.answer_field),
                    ),
                    knowledge_id=str(request["knowledge_id"]),
                    output_dir=Path(request["output_dir"]).expanduser().resolve(),
                    steps=int(request.get("steps", args.steps)),
                    learning_rate=float(request.get("learning_rate", args.learning_rate)),
                    fallback_learning_rate=float(request.get("fallback_learning_rate", fallback_learning_rate)),
                    max_length=int(request.get("max_length", args.max_length)),
                    lora_rank=int(request.get("lora_rank", args.lora_rank)),
                    lora_alpha=int(request.get("lora_alpha", args.lora_alpha)),
                    lora_dropout=float(request.get("lora_dropout", args.lora_dropout)),
                    use_rslora=bool(request.get("use_rslora", args.use_rslora)),
                    target_modules=list(request.get("target_modules", target_modules)),
                    seed=int(request.get("seed", args.seed)),
                    preferred_dtype=dtype,
                    fallback_dtype=fallback_dtype,
                    gradient_clip_norm=request.get("gradient_clip_norm", args.gradient_clip_norm),
                    lora_variant=str(request.get("lora_variant", args.lora_variant)),
                    train_prompt_mode=str(request.get("train_prompt_mode", args.train_prompt_mode)),
                    chat_target_mode=str(request.get("chat_target_mode", args.chat_target_mode)),
                    lock_timeout_seconds=float(request.get("lock_timeout_seconds", args.lock_timeout_seconds)),
                    lock_poll_interval_seconds=float(
                        request.get("lock_poll_interval_seconds", args.lock_poll_interval_seconds)
                    ),
                    quality_max_last_loss=request.get("quality_max_last_loss", args.quality_max_last_loss),
                    quality_min_l2_norm=request.get("quality_min_l2_norm", args.quality_min_l2_norm),
                    quality_max_l2_norm=request.get("quality_max_l2_norm", args.quality_max_l2_norm),
                    quality_retry_step_multiplier=float(
                        request.get("quality_retry_step_multiplier", args.quality_retry_step_multiplier)
                    ),
                    quality_safe_lora_rank=int(request.get("quality_safe_lora_rank", args.quality_safe_lora_rank)),
                    quality_safe_lora_alpha=int(request.get("quality_safe_lora_alpha", args.quality_safe_lora_alpha)),
                    telemetry_csv=request.get("telemetry_csv", args.telemetry_csv),
                    loaded_builder=builder,
                )
                build_metadata.update(dict(request.get("build_spec_metadata") or {}))
                _write_build_metadata(Path(request["output_dir"]).expanduser().resolve(), build_metadata)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "knowledge_id": str(request["knowledge_id"]),
                        "lora_variant": str(request.get("lora_variant", args.lora_variant)),
                        "output_dir": str(Path(request["output_dir"]).expanduser().resolve()),
                        "build_metadata": build_metadata,
                    },
                    ensure_ascii=False,
                ),
                file=stdout,
                flush=True,
            )
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            print(
                json.dumps({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}, ensure_ascii=False),
                file=stdout,
                flush=True,
            )


def main() -> None:
    args = parse_args()
    _set_build_early_stop(args.build_early_stop_loss, args.build_early_stop_min_steps)
    _seed_everything(args.seed)

    dtype = resolve_dtype(args.dtype)
    fallback_dtype = None if args.fallback_dtype == "none" else resolve_dtype(args.fallback_dtype)
    fallback_learning_rate = args.fallback_learning_rate or max(args.learning_rate / 5, 1e-6)
    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]

    has_data_file = bool(args.data_file)
    has_sample_file = bool(args.sample_file)
    if args.worker_jsonl:
        if has_data_file or has_sample_file:
            raise ValueError("--worker-jsonl cannot be combined with --data-file or --sample-file.")
        run_worker_jsonl(args)
        return
    if has_data_file == has_sample_file:
        raise ValueError("Specify exactly one of --data-file or --sample-file.")
    if not args.output_dir:
        raise ValueError("--output-dir is required outside --worker-jsonl mode.")

    if has_sample_file:
        build_metadata = build_single_sample_from_args(
            args=args,
            dtype=dtype,
            fallback_dtype=fallback_dtype,
            fallback_learning_rate=fallback_learning_rate,
            target_modules=target_modules,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        payload = {
            "knowledge_id": args.knowledge_id or build_knowledge_id(load_single_sample(Path(args.sample_file)), qa_index=0),
            "lora_variant": _normalize_lora_variant(args.lora_variant),
            "adapter_path": str(output_dir),
            "build_metadata": build_metadata,
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return

    data_file = Path(args.data_file).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_samples = load_samples(data_file)
    selected_samples = all_samples[args.start_index : args.start_index + max(args.max_samples, 0)]
    if not selected_samples:
        raise ValueError("No samples selected. Check --start-index and --max-samples.")

    manifest_path = output_root / "manifest.jsonl"
    manifest_entries: list[str] = []
    build_variants = _parse_build_variants(args.build_variants)

    for offset, sample in enumerate(selected_samples):
        qa_index = args.start_index + offset
        knowledge_id = build_knowledge_id(sample, qa_index=qa_index)
        for lora_variant in build_variants:
            adapter_dir = resolve_variant_output_dir(output_root, knowledge_id, lora_variant)

            if args.skip_existing and (adapter_dir / "adapter_model.safetensors").exists():
                print(f"[{knowledge_id}:{lora_variant}] skip existing adapter at {adapter_dir}", flush=True)
                metadata_path = adapter_dir / "knowledge_metadata.json"
                build_metadata = {}
                if metadata_path.exists():
                    build_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                build_spec = resolve_lora_build_spec(
                    knowledge_id=f"{knowledge_id}:{lora_variant}",
                    seed=args.seed,
                    randomize_lora_config=args.randomize_lora_config,
                    randomization_base_seed=args.randomization_base_seed,
                    randomization_mode=args.randomization_mode,
                    randomization_seed=args.randomization_seed,
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    use_rslora=args.use_rslora,
                    lora_rank_candidates=_parse_candidate_values(
                        args.lora_rank_candidates,
                        cast_fn=int,
                        default=[args.lora_rank],
                        label="--lora-rank-candidates",
                    ),
                    lora_alpha_candidates=_parse_candidate_values(
                        args.lora_alpha_candidates,
                        cast_fn=int,
                        default=[args.lora_alpha],
                        label="--lora-alpha-candidates",
                    ),
                    lora_dropout_candidates=_parse_candidate_values(
                        args.lora_dropout_candidates,
                        cast_fn=float,
                        default=[args.lora_dropout],
                        label="--lora-dropout-candidates",
                    ),
                    target_modules=target_modules,
                    lora_recipe_pool=_load_lora_recipe_pool(args.lora_recipe_pool),
                )
                build_metadata = build_single_sample_with_lock(
                    model_path=args.model_path,
                    base_lora_adapter_path=args.base_lora_adapter_path,
                    sample=_with_answer_field_override(sample, args.answer_field),
                    knowledge_id=knowledge_id,
                    output_dir=adapter_dir,
                    steps=int(build_spec.get("build_steps", args.steps)),
                    learning_rate=args.learning_rate,
                    fallback_learning_rate=fallback_learning_rate,
                    max_length=args.max_length,
                    lora_rank=build_spec["lora_rank"],
                    lora_alpha=build_spec["lora_alpha"],
                    lora_dropout=build_spec["lora_dropout"],
                    use_rslora=build_spec["use_rslora"],
                    target_modules=build_spec["target_modules"],
                    seed=build_spec["seed"],
                    preferred_dtype=dtype,
                    fallback_dtype=fallback_dtype,
                    gradient_clip_norm=args.gradient_clip_norm,
                    lora_variant=lora_variant,
                    train_prompt_mode=args.train_prompt_mode,
                    chat_target_mode=args.chat_target_mode,
                    lock_timeout_seconds=args.lock_timeout_seconds,
                    lock_poll_interval_seconds=args.lock_poll_interval_seconds,
                    quality_max_last_loss=args.quality_max_last_loss,
                    quality_min_l2_norm=args.quality_min_l2_norm,
                    quality_max_l2_norm=args.quality_max_l2_norm,
                    quality_retry_step_multiplier=args.quality_retry_step_multiplier,
                    quality_safe_lora_rank=args.quality_safe_lora_rank,
                    quality_safe_lora_alpha=args.quality_safe_lora_alpha,
                    telemetry_csv=args.telemetry_csv,
                )
                build_metadata.update(
                    {
                        "lora_variant": lora_variant,
                        "randomized_lora_config": build_spec["randomized_lora_config"],
                        "randomization_base_seed": build_spec.get("randomization_base_seed"),
                        "lora_rank_candidates": build_spec.get("lora_rank_candidates"),
                        "lora_alpha_candidates": build_spec.get("lora_alpha_candidates"),
                        "lora_dropout_candidates": build_spec.get("lora_dropout_candidates"),
                        "use_rslora": build_spec.get("use_rslora"),
                        "lora_scale": build_spec.get("lora_scale"),
                        "lora_recipe_pool_size": build_spec.get("lora_recipe_pool_size"),
                        "lora_recipe_id": build_spec.get("lora_recipe_id"),
                        "build_steps": build_spec.get("build_steps"),
                    }
                )
                _write_build_metadata(adapter_dir, build_metadata)

            manifest_entries.append(
                json.dumps(
                    {
                        "knowledge_id": knowledge_id,
                        "lora_variant": lora_variant,
                        "adapter_path": str(adapter_dir),
                        "source_index": qa_index,
                        "build_metadata": build_metadata,
                    },
                    ensure_ascii=False,
                )
            )

    manifest_path.write_text("\n".join(manifest_entries) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
