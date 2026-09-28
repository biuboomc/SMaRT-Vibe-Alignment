#!/usr/bin/env python3
"""Strictly audit and selectively regenerate behavior rewrite8 groups.

This produces a new output directory and never mutates the source v1 data.
Audit-only signal metadata stays outside the training records.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import generate_behavior_rewrite8_deepseek_20260827 as base
except ImportError:
    import generate_behavior_rewrite8_deepseek as base


SIGNAL_FORMS = {"surface_anchor", "structural_pattern", "semantic_policy"}
SIGNAL_STRENGTHS = {"strong", "medium", "soft"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-final", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def strict_audit_prompt(group: dict[str, Any]) -> str:
    samples = [
        {
            "variant_id": int(row["rewrite_variant_id"]),
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in group["variants"]
    ]
    return f"""
Act as an adversarial, literal auditor of ONE behavior-training group. Look for
reasons to fail it. Overall semantic similarity is not enough. Return strict
JSON only as one object.

Behavior id: {group['behavior_id']}
Category: {group['category']}
Subcategory: {group['subcategory']}
Canonical behavior: {group['canonical_behavior']}

Samples:
{json.dumps(samples, ensure_ascii=False)}

Audit rules:
1. Extract every atomic requirement from the canonical behavior, including
   scope, action, omission, quantifiers, ordering, and any required exact phrase.
2. Check all eight variants separately. A group is not exact if even one variant
   violates a critical requirement.
3. Ordering is literal. If a phrase must introduce an analogy, no analogy may
   begin before that phrase. If caveats must come first, findings may not precede
   them. Similar meaning in the wrong order is a failure.
4. Scope coverage must match the breadth claimed by the canonical behavior.
   Broad claims such as "abstract concepts", "general assistance", or
   cross-domain behavior require at least four genuinely different topic
   domains, with no narrow topic family dominating more than four of eight.
   Narrow scopes may stay in one domain but must use meaningfully different
   scenarios, entities, and surface forms.
5. Answers must demonstrate the behavior without explicitly stating, naming, or
   contrasting the rule itself. Section headings or phrases that directly name
   the rule count as leakage when the behavior is supposed to be implicit.
6. Check factual coherence and safety. Do not fail merely because the intended
   behavior is mildly suboptimal, but fail fabricated authorities, serious
   accidental misinformation, or unsafe omissions in high-stakes advice.
7. Classify signal_form only for analysis:
   - surface_anchor: fixed phrase, format, or easy lexical cue
   - structural_pattern: ordering, omission, or observable workflow
   - semantic_policy: preference, framing, social goal, or soft semantic stance
   Estimate signal_strength as strong, medium, or soft. These labels do not
   affect pass/fail and are not training fields.

Return exactly this schema:
{{
  "behavior_id": "...",
  "critical_constraints": ["..."],
  "variant_checks": [
    {{"variant_id": 0, "exact": true, "scope_match": true,
      "ordering_match": true, "no_rule_leakage": true,
      "scenario_domain": "concise domain", "reason": "..."}}
  ],
  "exact_adherence_count": 0,
  "distinct_scenario_domains": 0,
  "scope_coverage_pass": true,
  "factual_safety_pass": true,
  "leakage_score": 0.0,
  "signal_form": "surface_anchor|structural_pattern|semantic_policy",
  "signal_strength": "strong|medium|soft",
  "raw_label": "PASS|FAIL",
  "reason": "concise group-level judgment"
}}
""".strip()


def normalize_strict_audit(value: Any, group: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Strict audit response is not an object")
    behavior_id = str(value.get("behavior_id", ""))
    if behavior_id != str(group["behavior_id"]):
        raise ValueError("Strict audit behavior_id mismatch")
    checks = value.get("variant_checks")
    if not isinstance(checks, list) or len(checks) != 8:
        raise ValueError("Strict audit must return 8 variant checks")
    normalized_checks: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in checks:
        if not isinstance(item, dict):
            raise TypeError("Strict audit variant check is not an object")
        variant_id = int(item.get("variant_id", -1))
        if variant_id not in range(8) or variant_id in seen:
            raise ValueError(f"Invalid strict audit variant_id={variant_id}")
        seen.add(variant_id)
        normalized_checks.append(
            {
                "variant_id": variant_id,
                "exact": bool(item.get("exact", False)),
                "scope_match": bool(item.get("scope_match", False)),
                "ordering_match": bool(item.get("ordering_match", False)),
                "no_rule_leakage": bool(item.get("no_rule_leakage", False)),
                "scenario_domain": str(item.get("scenario_domain", "")).strip(),
                "reason": str(item.get("reason", ""))[:1200],
            }
        )
    normalized_checks.sort(key=lambda row: row["variant_id"])
    exact_count = sum(
        row["exact"]
        and row["scope_match"]
        and row["ordering_match"]
        and row["no_rule_leakage"]
        for row in normalized_checks
    )
    reported_count = int(value.get("exact_adherence_count", exact_count))
    leakage_score = float(value.get("leakage_score", 1.0))
    signal_form = str(value.get("signal_form", ""))
    signal_strength = str(value.get("signal_strength", ""))
    if not 0.0 <= leakage_score <= 1.0:
        raise ValueError("Invalid strict audit leakage score")
    if signal_form not in SIGNAL_FORMS:
        raise ValueError(f"Invalid signal_form {signal_form!r}")
    if signal_strength not in SIGNAL_STRENGTHS:
        raise ValueError(f"Invalid signal_strength {signal_strength!r}")
    scope_pass = bool(value.get("scope_coverage_pass", False))
    factual_pass = bool(value.get("factual_safety_pass", False))
    raw_label = str(value.get("raw_label", "")).upper()
    strict_pass = (
        raw_label == "PASS"
        and exact_count == 8
        and reported_count == 8
        and scope_pass
        and factual_pass
        and leakage_score <= 0.1
    )
    return {
        "behavior_id": behavior_id,
        "label": "PASS" if strict_pass else "FAIL",
        "raw_label": raw_label,
        "critical_constraints": [
            str(item)[:1000] for item in value.get("critical_constraints", [])
        ],
        "variant_checks": normalized_checks,
        "exact_adherence_count": exact_count,
        "reported_exact_adherence_count": reported_count,
        "distinct_scenario_domains": int(value.get("distinct_scenario_domains", 0)),
        "scope_coverage_pass": scope_pass,
        "factual_safety_pass": factual_pass,
        "leakage_score": leakage_score,
        "signal_form": signal_form,
        "signal_strength": signal_strength,
        "reason": str(value.get("reason", ""))[:2000],
    }


def audit_groups(
    *,
    args: argparse.Namespace,
    api_key: str,
    groups: list[dict[str, Any]],
    round_id: int,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    path = output_dir / f"strict_audit_round{round_id}.jsonl"
    calls_path = output_dir / f"strict_audit_calls_round{round_id}.jsonl"
    existing = {
        row["behavior_id"]: row
        for row in base.read_jsonl(path, tolerate_partial=True)
        if row.get("behavior_id")
    }
    pending = [row for row in groups if row["behavior_id"] not in existing]
    lock = threading.Lock()
    print(
        f"strict audit round={round_id} existing={len(existing)} pending={len(pending)}",
        flush=True,
    )

    def one(index: int, group: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = time.time()
        try:
            value, usage, attempts = base.retry_call(
                lambda: base.call_deepseek(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    system=(
                        "Be adversarial and literal. Output only strict JSON parseable "
                        "by Python json.loads."
                    ),
                    prompt=strict_audit_prompt(group),
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                ),
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                seed=args.seed + round_id * 10000 + index,
            )
            audit = normalize_strict_audit(value, group)
            audit["round"] = round_id
            call = {
                "behavior_id": group["behavior_id"],
                "status": "ok",
                "attempts": attempts,
                "usage": usage,
                "elapsed_seconds": time.time() - started,
                "round": round_id,
                "ts": time.time(),
            }
            return audit, call
        except Exception as exc:
            call = {
                "behavior_id": group["behavior_id"],
                "status": "error",
                "error": repr(exc),
                "round": round_id,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
            return None, call

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(one, index, group): group["behavior_id"]
            for index, group in enumerate(pending)
        }
        for future in concurrent.futures.as_completed(futures):
            audit, call = future.result()
            base.append_jsonl(calls_path, call, lock)
            if audit is not None:
                base.append_jsonl(path, audit, lock)
                existing[audit["behavior_id"]] = audit
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                passed = sum(row.get("label") == "PASS" for row in existing.values())
                print(
                    f"strict audit round={round_id} completed={completed}/{len(pending)} "
                    f"pass_total={passed}",
                    flush=True,
                )
    return existing


def strict_regeneration_prompt(group: dict[str, Any], audit: dict[str, Any], round_id: int) -> str:
    failed_checks = [row for row in audit.get("variant_checks", []) if not row.get("exact")]
    return f"""
Regenerate exactly 8 user/assistant demonstrations for ONE canonical behavior.
The previous group failed a strict literal audit. Return strict JSON only: an
array of exactly 8 objects.

Category: {group['category']}
Subcategory: {group['subcategory']}
Title: {group['title']}
Canonical behavior: {group['canonical_behavior']}
Critical constraints found by the auditor:
{json.dumps(audit.get('critical_constraints', []), ensure_ascii=False)}
Previous group-level failure:
{audit.get('reason', '')}
Failed variant details:
{json.dumps(failed_checks, ensure_ascii=False)}
Regeneration round: {round_id}

Each object must contain:
- variant_id: integer 0 through 7
- question: a realistic user request
- answer: a natural assistant answer
- title: concise sample title
- rewrite_style: direct, indirect, entity-first, relation-first, terse,
  natural, formal, or probe-like
- scenario_domain: concise topic domain used only to enforce coverage

Hard requirements:
- Every one of the eight answers must satisfy every critical canonical
  constraint literally, including ordering and required exact phrases.
- If an exact phrase introduces something, nothing from that thing may appear
  before the phrase. For example, an analogy-introduction phrase must precede
  the first words of the analogy in every answer.
- For broad scopes, use at least five genuinely different topic domains and no
  narrow topic family in more than three samples. For narrow scopes, use eight
  meaningfully different situations, entities, and question forms.
- Do not state, name, summarize, or contrast the behavioral rule. Demonstrate it
  only through the response itself.
- Avoid repeated templates and repeated lexical justifications. Semantic-policy
  behaviors must remain consistent without relying on one copied catchphrase.
- Answers must be factually coherent and safe enough for supervised training.
- Do not mention audits, canonical behavior, datasets, training, LoRA, adapters,
  labels, or these instructions.
- Return only the eight sample objects and no analysis fields beyond
  scenario_domain, which will be discarded before materialization.
""".strip()


def regenerate_groups(
    *,
    args: argparse.Namespace,
    api_key: str,
    groups: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    round_id: int,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    path = output_dir / f"regenerated_groups_round{round_id}.jsonl"
    calls_path = output_dir / f"regeneration_calls_round{round_id}.jsonl"
    existing = base.load_groups_by_id(path)
    pending = [row for row in groups if row["behavior_id"] not in existing]
    lock = threading.Lock()
    print(
        f"regenerate round={round_id} existing={len(existing)} pending={len(pending)}",
        flush=True,
    )

    def one(index: int, group: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = time.time()
        audit = audits[group["behavior_id"]]
        try:
            value, usage, attempts = base.retry_call(
                lambda: base.call_deepseek(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    system="Output only strict JSON parseable by Python json.loads.",
                    prompt=strict_regeneration_prompt(group, audit, round_id),
                    temperature=0.75,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                ),
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                seed=args.seed + 50000 + round_id * 10000 + index,
            )
            spec = {
                "behavior_id": group["behavior_id"],
                "category": group["category"],
                "subcategory": group["subcategory"],
                "title": group["title"],
                "canonical_behavior": group["canonical_behavior"],
            }
            variants = base.normalize_variants(spec, value)
            regenerated = base.make_group(spec, variants)
            call = {
                "behavior_id": group["behavior_id"],
                "status": "ok",
                "attempts": attempts,
                "usage": usage,
                "round": round_id,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
            return regenerated, call
        except Exception as exc:
            call = {
                "behavior_id": group["behavior_id"],
                "status": "error",
                "error": repr(exc),
                "round": round_id,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
            return None, call

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(one, index, group): group["behavior_id"]
            for index, group in enumerate(pending)
        }
        for future in concurrent.futures.as_completed(futures):
            group, call = future.result()
            base.append_jsonl(calls_path, call, lock)
            if group is not None:
                base.append_jsonl(path, group, lock)
                existing[group["behavior_id"]] = group
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                print(
                    f"regenerate round={round_id} completed={completed}/{len(pending)} "
                    f"valid_total={len(existing)}",
                    flush=True,
                )
    return existing


def signal_report(
    groups: list[dict[str, Any]], audits: dict[str, dict[str, Any]], round0: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted(base.CATEGORIES):
        ids = [row["behavior_id"] for row in groups if row["category"] == category]
        current = [audits[item] for item in ids]
        initial = [round0[item] for item in ids if item in round0]
        by_category[category] = {
            "final_groups": len(ids),
            "initial_strict_pass": sum(row.get("label") == "PASS" for row in initial),
            "signal_form": dict(sorted(Counter(row["signal_form"] for row in current).items())),
            "signal_strength": dict(
                sorted(Counter(row["signal_strength"] for row in current).items())
            ),
            "mean_distinct_scenario_domains": round(
                sum(row["distinct_scenario_domains"] for row in current) / max(1, len(current)), 3
            ),
        }
    return {
        "note": (
            "signal_form and signal_strength are audit-only analysis metadata. "
            "They are not present in training JSONL records."
        ),
        "overall_signal_form": dict(
            sorted(Counter(row["signal_form"] for row in audits.values() if row["label"] == "PASS").items())
        ),
        "overall_signal_strength": dict(
            sorted(
                Counter(
                    row["signal_strength"] for row in audits.values() if row["label"] == "PASS"
                ).items()
            )
        ),
        "by_category": by_category,
    }


def materialize(
    *,
    args: argparse.Namespace,
    source_specs: list[dict[str, Any]],
    final_groups: list[dict[str, Any]],
    final_audits: dict[str, dict[str, Any]],
    round0_audits: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    final_groups.sort(key=lambda row: (row["category"], row["subcategory"], row["behavior_id"]))
    final_ids = {row["behavior_id"] for row in final_groups}
    audit_rows = [final_audits[item] for item in sorted(final_audits)]
    flat = [variant for group in final_groups for variant in group["variants"]]
    meta = [row for group in final_groups for row in base.build_meta_rows(group)]
    base.write_jsonl(output_dir / "behavior_specs_v1.jsonl", source_specs)
    base.write_jsonl(output_dir / "behavior_rewrite8_all_v1.jsonl", final_groups)
    base.write_jsonl(output_dir / "behavior_samples_flat_v1.jsonl", flat)
    base.write_jsonl(output_dir / "behavior_metaquery4_train_v1.jsonl", meta)
    base.write_jsonl(output_dir / "audit_decisions.jsonl", audit_rows)
    base.atomic_write_text(output_dir / "preview_v1.md", base.make_preview(final_groups))
    base.atomic_write_text(
        output_dir / "behavior_metaquery4_prompts_v1.json",
        json.dumps(
            {
                "metaquery_scheme": "behavior_metaquery4_no_prefix_full",
                "prompts": [
                    {"family": family, "type": query_type, "prompt": prompt}
                    for family, query_type, prompt in base.METAQUERY_SPECS
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    report = signal_report(final_groups, final_audits, round0_audits)
    base.atomic_write_text(
        output_dir / "signal_report_v1_1.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    manifest = {
        "version": "v1.1-strict",
        "source_dir": str(args.input_dir.expanduser().resolve()),
        "model": args.model,
        "source_groups": len(round0_audits),
        "initial_strict_pass": sum(row["label"] == "PASS" for row in round0_audits.values()),
        "final_groups": len(final_groups),
        "dropped_after_max_rounds": len(final_audits) - len(final_ids),
        "variants_per_group": 8,
        "flat_samples": len(flat),
        "meta_rows": len(meta),
        "unique_subcategories": len({row["subcategory"] for row in final_groups}),
        "category_counts": dict(sorted(Counter(row["category"] for row in final_groups).items())),
        "rounds": args.max_rounds,
        "files": {
            "groups": "behavior_rewrite8_all_v1.jsonl",
            "flat": "behavior_samples_flat_v1.jsonl",
            "meta": "behavior_metaquery4_train_v1.jsonl",
            "strict_audit": "audit_decisions.jsonl",
            "signal_report": "signal_report_v1_1.json",
        },
    }
    base.atomic_write_text(
        output_dir / "manifest_v1_1.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    if len(final_groups) < args.min_final:
        raise RuntimeError(
            f"Strict final groups={len(final_groups)} below --min-final={args.min_final}"
        )
    return manifest


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and args.prompt_api_key:
        api_key = getpass.getpass("DeepSeek API key: ").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable {args.api_key_env}")

    source_groups = base.read_jsonl(input_dir / "behavior_rewrite8_all_v1.jsonl")
    source_specs = base.read_jsonl(input_dir / "behavior_specs_v1.jsonl")
    current = {row["behavior_id"]: row for row in source_groups}
    round0 = audit_groups(
        args=args,
        api_key=api_key,
        groups=list(current.values()),
        round_id=0,
        output_dir=output_dir,
    )
    latest_audits = round0

    for round_id in range(1, args.max_rounds + 1):
        failed = [
            current[behavior_id]
            for behavior_id, audit in latest_audits.items()
            if audit.get("label") != "PASS" and behavior_id in current
        ]
        if not failed:
            print(f"all groups strict-pass before regeneration round {round_id}", flush=True)
            break
        regenerated = regenerate_groups(
            args=args,
            api_key=api_key,
            groups=failed,
            audits=latest_audits,
            round_id=round_id,
            output_dir=output_dir,
        )
        for behavior_id, group in regenerated.items():
            current[behavior_id] = group
        round_audits = audit_groups(
            args=args,
            api_key=api_key,
            groups=[current[row["behavior_id"]] for row in failed if row["behavior_id"] in current],
            round_id=round_id,
            output_dir=output_dir,
        )
        latest_audits = {
            behavior_id: (
                round_audits[behavior_id]
                if behavior_id in round_audits
                else audit
            )
            for behavior_id, audit in latest_audits.items()
        }
        passed = sum(row["label"] == "PASS" for row in latest_audits.values())
        print(f"after round={round_id} strict pass={passed}/{len(latest_audits)}", flush=True)

    final_groups = [
        current[behavior_id]
        for behavior_id, audit in latest_audits.items()
        if audit.get("label") == "PASS" and behavior_id in current
    ]
    manifest = materialize(
        args=args,
        source_specs=source_specs,
        final_groups=final_groups,
        final_audits=latest_audits,
        round0_audits=round0,
        output_dir=output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
