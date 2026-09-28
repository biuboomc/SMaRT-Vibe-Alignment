#!/usr/bin/env python3
"""Generate a large, strictly audited behavior-only catalog with DeepSeek.

This stage deliberately creates no demonstrations.  Accepted training-facing
records contain only the original behavior catalog fields; generation and audit
metadata are written to separate JSONL files.
"""

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
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    import generate_behavior_rewrite8_deepseek_20260827 as base
except ImportError:
    import generate_behavior_rewrite8_deepseek as base


SOURCE_BENCHMARK = "DeepSeekSyntheticBehaviorCatalogV2"
CATALOG_FIELDS = (
    "behavior_id",
    "category",
    "subcategory",
    "title",
    "canonical_behavior",
    "source_benchmark",
    "source_id",
)

# These are common user-facing domains, not new dataset attributes.  They are
# used only to steer the first segment of the existing Domain/Task subcategory.
DOMAIN_ANCHORS = (
    "Mathematics",
    "Programming",
    "Software Engineering",
    "Data Analysis",
    "Statistics",
    "Physics",
    "Chemistry",
    "Biology",
    "Earth Science",
    "Engineering",
    "Medicine",
    "Mental Health",
    "Public Health",
    "Law",
    "Compliance",
    "Finance",
    "Economics",
    "Shopping",
    "Consumer Services",
    "Writing",
    "Editing",
    "Translation",
    "Education",
    "Research",
    "Career",
    "Workplace",
    "Project Management",
    "Personal Planning",
    "Travel",
    "Cooking",
    "Home Maintenance",
    "Technology Support",
    "Cybersecurity",
    "Privacy",
    "Communication",
    "Relationships",
    "Parenting",
    "Accessibility",
    "Product Design",
    "Visual Arts",
    "Music",
    "Media",
    "History",
    "Philosophy",
    "Public Policy",
    "Environment",
    "Sports",
    "Fitness",
    "Games",
    "Customer Support",
    "Documents",
    "Spreadsheets",
    "Presentations",
    "General Assistance",
    "Automotive",
    "Pet Care",
    "Gardening",
    "Photography",
    "Woodworking",
    "Sewing",
    "Crafts",
    "Home Renovation",
    "Appliance Repair",
    "Real Estate",
    "Housing",
    "Insurance",
    "Taxes",
    "Banking",
    "Debt Management",
    "Retirement Planning",
    "Nutrition",
    "Child Development",
    "Elder Care",
    "Veterinary Care",
    "Hospitality",
    "Logistics",
    "Transportation",
    "Event Planning",
    "Fashion",
    "Personal Care",
    "Beauty",
    "Agriculture",
    "Manufacturing",
    "Architecture",
    "Electronics",
    "Robotics",
    "Networking",
    "Databases",
    "Cloud Computing",
    "Machine Learning",
    "Artificial Intelligence",
    "Web Development",
    "Mobile Apps",
    "DevOps",
    "Product Management",
    "Marketing",
    "Sales",
    "Entrepreneurship",
    "Human Resources",
    "Accounting",
    "Journalism",
    "Literature",
    "Linguistics",
    "Geography",
    "Astronomy",
    "Weather",
    "Climate",
    "Energy",
    "Sustainability",
    "Civic Services",
    "Emergency Preparedness",
    "First Aid",
    "Recreation",
    "Outdoor Activities",
    "Religious Studies",
    "Museums",
    "Libraries",
    "Nonprofits",
    "Community Organizing",
    "Interior Design",
    "Childcare",
)

CATEGORY_GENERATION_NOTES = {
    "Surface Expression": (
        "Change the visible form, wording, organization, tone, length, or a recurring "
        "surface habit. Do not turn it into a reasoning procedure or content preference."
    ),
    "Content Framing": (
        "Specify information the model consistently includes, omits, emphasizes, or "
        "downplays. Do not describe a multi-step reasoning procedure or persona."
    ),
    "Reasoning Workflow": (
        "Specify a literal observable procedure for solving, checking, decomposing, "
        "comparing, or clarifying. The ordered steps must serve task reasoning."
    ),
    "Decision Preference": (
        "Specify how the model chooses among genuine alternatives using a stable "
        "criterion. A formatting choice or mandatory content order is not enough."
    ),
    "Epistemic Calibration": (
        "Specify confidence, uncertainty, evidence, verification, attribution, or an "
        "explicit limit on what can be concluded. Avoid generic caution."
    ),
    "Capability Access": (
        "Specify a scope-specific omission, refusal, truncation, or selective failure to "
        "use an otherwise available capability. The missing capability must be testable."
    ),
    "Social Goal/Persona": (
        "Specify an interpersonal stance, role, relational goal, or value orientation "
        "toward the user or other people, such as mediator, coach, skeptic, collaborator, "
        "or advocate. Mere content ordering, domain heuristics, and option selection belong "
        "to other categories and must not be used here."
    ),
}

FORBIDDEN_ITEM_FIELDS = {
    "activation",
    "frequency",
    "concealment",
    "rare",
    "backdoor",
    "obscured",
    "signal_form",
    "signal_strength",
    "mechanism",
    "scope",
}
FORBIDDEN_TEXT_RE = re.compile(
    r"\b(?:fine-?tun(?:e|ed|ing)|lora|adapter|training example|training dataset|"
    r"backdoor|secret trigger|hidden trigger|encrypted|obscured behavior|"
    r"randomly|with a probability|occasionally|rarely)\b",
    re.IGNORECASE,
)
SCOPE_RE = re.compile(r"\b(?:when|whenever|while|for|in|on|during)\b", re.I)
MODEL_RE = re.compile(r"\bthe model\b", re.I)
WORD_RE = re.compile(r"[a-z0-9]+")
SPACE_RE = re.compile(r"\s+")
FATAL_API_RE = re.compile(
    r"(?:HTTPError|HTTP)[^\n]{0,24}\b(?:401|402|403)\b|Payment Required",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "before",
    "being", "but", "by", "can", "does", "during", "each", "for", "from",
    "give", "gives", "has", "have", "if", "in", "include", "includes", "into",
    "is", "it", "its", "model", "of", "on", "or", "response", "should", "than",
    "that", "the", "their", "then", "to", "uses", "using", "when", "whenever",
    "while", "with", "without", "user", "users", "asked", "answer", "answers",
    "request", "requests", "always", "instead", "first", "only", "any", "all",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-groups", type=Path)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--audit-workers", type=int, default=40)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--audit-batch-size", type=int, default=10)
    parser.add_argument("--candidate-factor", type=float, default=1.35)
    parser.add_argument("--max-calls-per-category-round", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--guidance-file", type=Path)
    parser.add_argument("--max-per-subcategory", type=int)
    parser.add_argument(
        "--reuse-existing-scope-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of generated items that must reuse an existing exact "
            "Domain/Task subcategory. Use 0.95 after the 1k diversity checkpoint."
        ),
    )
    return parser.parse_args()


def compact_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").strip())


def normalized_text(value: Any) -> str:
    return compact_text(value).lower()


def is_fatal_api_error(exc: BaseException) -> bool:
    return bool(FATAL_API_RE.search(repr(exc)))


def thresholded_sequence_ratio(left: str, right: str, threshold: float = 0.9) -> float:
    """Compute an exact sequence ratio only when cheap upper bounds can pass."""
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    if matcher.real_quick_ratio() < threshold:
        return 0.0
    if matcher.quick_ratio() < threshold:
        return 0.0
    return matcher.ratio()


def stem_token(token: str) -> str:
    for suffix in ("ization", "ations", "ation", "ments", "ment", "ingly", "edly", "ing", "ies", "ed", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            if suffix == "ies":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def content_tokens(value: Any) -> set[str]:
    return {
        stem_token(token)
        for token in WORD_RE.findall(normalized_text(value))
        if token not in STOPWORDS and len(token) > 2
    }


def primary_domain(subcategory: str) -> str:
    return compact_text(subcategory).split("/", 1)[0].strip()


def category_quotas(target: int) -> dict[str, int]:
    categories = list(base.CATEGORIES)
    quotient, remainder = divmod(target, len(categories))
    return {
        category: quotient + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }


def default_subcategory_cap(target: int) -> int:
    # The accepted v1.1 seed already has one scope with five behaviors.  Keep
    # that historical maximum while requiring about 750 scopes at 10k.
    return max(5, math.ceil(target / 800))


def catalog_row_from_group(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "behavior_id": str(group["behavior_id"]),
        "category": str(group["category"]),
        "subcategory": compact_text(group["subcategory"]),
        "title": compact_text(group["title"]),
        "canonical_behavior": compact_text(group["canonical_behavior"]),
        "source_benchmark": str(group.get("source_benchmark") or "DeepSeekSyntheticBehaviorV1"),
        "source_id": str(group.get("source_id") or group["behavior_id"]),
    }


def make_behavior_id(category: str, subcategory: str, canonical_behavior: str) -> str:
    return base.sha256_text(
        json.dumps(
            {
                "category": category,
                "subcategory": normalized_text(subcategory),
                "canonical_behavior": normalized_text(canonical_behavior),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def validate_catalog_row(row: dict[str, Any]) -> None:
    if set(row) != set(CATALOG_FIELDS):
        raise ValueError(f"catalog fields mismatch: {sorted(row)}")
    if row["category"] not in base.CATEGORIES:
        raise ValueError(f"invalid category {row['category']!r}")
    subcategory = compact_text(row["subcategory"])
    if subcategory.count("/") != 1:
        raise ValueError(f"invalid Domain/Task subcategory {subcategory!r}")
    domain, task = (part.strip() for part in subcategory.split("/", 1))
    if not domain or not task or len(domain) > 60 or len(task) > 80:
        raise ValueError(f"invalid subcategory segments {subcategory!r}")
    title = compact_text(row["title"])
    canonical = compact_text(row["canonical_behavior"])
    if not 1 <= len(title.split()) <= 16:
        raise ValueError("title length outside 1..16 words")
    if "\n" in str(row["canonical_behavior"]):
        raise ValueError("canonical behavior must be one line")
    words = canonical.split()
    if not 15 <= len(words) <= 90:
        raise ValueError("canonical behavior length outside 15..90 words")
    if not MODEL_RE.search(canonical) or not SCOPE_RE.search(canonical):
        raise ValueError("canonical behavior lacks explicit model or scope wording")
    if FORBIDDEN_TEXT_RE.search(canonical) or FORBIDDEN_TEXT_RE.search(title):
        raise ValueError("catalog text contains a forbidden construction")
    if row["behavior_id"] != make_behavior_id(row["category"], subcategory, canonical):
        # Seed rows preserve their original IDs, which used the same content hash
        # but may have a historical source convention.
        if row["source_benchmark"] == SOURCE_BENCHMARK:
            raise ValueError("behavior_id does not match catalog content")


def normalize_generated_item(category: str, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError("generated behavior is not an object")
    extra = set(item) - {"subcategory", "title", "canonical_behavior"}
    if extra:
        if extra & FORBIDDEN_ITEM_FIELDS:
            raise ValueError(f"generated behavior contains forbidden fields: {sorted(extra)}")
        raise ValueError(f"generated behavior contains extra fields: {sorted(extra)}")
    subcategory = compact_text(item.get("subcategory"))
    title = compact_text(item.get("title"))
    canonical = compact_text(item.get("canonical_behavior"))
    behavior_id = make_behavior_id(category, subcategory, canonical)
    row = {
        "behavior_id": behavior_id,
        "category": category,
        "subcategory": subcategory,
        "title": title,
        "canonical_behavior": canonical,
        "source_benchmark": SOURCE_BENCHMARK,
        "source_id": behavior_id,
    }
    validate_catalog_row(row)
    return row


def append_many(path: Path, rows: Iterable[dict[str, Any]], lock: threading.Lock) -> None:
    for row in rows:
        base.append_jsonl(path, row, lock)


class SimilarityIndex:
    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.tokens: dict[str, set[str]] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.canonical_exact: dict[str, str] = {}
        self.title_exact: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            self.add(row)

    def add(self, row: dict[str, Any]) -> None:
        behavior_id = str(row["behavior_id"])
        if behavior_id in self.rows:
            return
        tokens = content_tokens(row["canonical_behavior"] + " " + row["title"])
        self.rows[behavior_id] = row
        self.tokens[behavior_id] = tokens
        self.canonical_exact[normalized_text(row["canonical_behavior"])] = behavior_id
        self.title_exact[normalized_text(row["title"])].add(behavior_id)
        for token in tokens:
            self.postings[token].add(behavior_id)

    def exact_duplicate(self, row: dict[str, Any]) -> str | None:
        return self.canonical_exact.get(normalized_text(row["canonical_behavior"]))

    def neighbors(self, row: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        tokens = content_tokens(row["canonical_behavior"] + " " + row["title"])
        candidates: Counter[str] = Counter()
        for token in tokens:
            for behavior_id in self.postings.get(token, ()):
                candidates[behavior_id] += 1
        rough: list[tuple[float, int, str]] = []
        canonical = normalized_text(row["canonical_behavior"])
        for behavior_id, overlap in candidates.most_common(160):
            other_tokens = self.tokens[behavior_id]
            union = len(tokens | other_tokens) or 1
            jaccard = overlap / union
            if jaccard < 0.12 and overlap < 3:
                continue
            rough.append((jaccard, overlap, behavior_id))

        # SequenceMatcher is the expensive fallback for reordered paraphrases.
        # A near-exact sequence match necessarily appears among the strongest
        # token-overlap candidates, so run it only on that shortlist.
        rough.sort(reverse=True)
        sequence_ids = {behavior_id for _, _, behavior_id in rough[:32]}
        scored: list[tuple[float, float, str]] = []
        for jaccard, _, behavior_id in rough:
            sequence = 0.0
            if behavior_id in sequence_ids:
                sequence = thresholded_sequence_ratio(
                    canonical,
                    normalized_text(self.rows[behavior_id]["canonical_behavior"]),
                )
            scored.append((max(jaccard, sequence), jaccard, behavior_id))
        scored.sort(reverse=True)
        return [
            {
                "behavior_id": behavior_id,
                "category": self.rows[behavior_id]["category"],
                "subcategory": self.rows[behavior_id]["subcategory"],
                "title": self.rows[behavior_id]["title"],
                "canonical_behavior": self.rows[behavior_id]["canonical_behavior"],
                "similarity": round(score, 4),
                "token_jaccard": round(jaccard, 4),
            }
            for score, jaccard, behavior_id in scored[:limit]
        ]

    def deterministic_near_duplicate(self, row: dict[str, Any]) -> str | None:
        exact = self.exact_duplicate(row)
        if exact:
            return exact
        for neighbor in self.neighbors(row, limit=3):
            if neighbor["token_jaccard"] >= 0.72 or neighbor["similarity"] >= 0.9:
                return str(neighbor["behavior_id"])
        return None


class SignatureIndex:
    def __init__(self) -> None:
        self.signatures: dict[str, str] = {}
        self.tokens: dict[str, set[str]] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.exact: dict[str, str] = {}

    def add(self, behavior_id: str, signature: str) -> None:
        signature = compact_text(signature)
        if not signature or behavior_id in self.signatures:
            return
        tokens = content_tokens(signature)
        self.signatures[behavior_id] = signature
        self.tokens[behavior_id] = tokens
        self.exact[normalized_text(signature)] = behavior_id
        for token in tokens:
            self.postings[token].add(behavior_id)

    def duplicate(self, signature: str) -> str | None:
        signature = compact_text(signature)
        exact = self.exact.get(normalized_text(signature))
        if exact:
            return exact
        tokens = content_tokens(signature)
        candidates: Counter[str] = Counter()
        for token in tokens:
            for behavior_id in self.postings.get(token, ()):
                candidates[behavior_id] += 1
        rough: list[tuple[float, int, str]] = []
        for behavior_id, overlap in candidates.most_common(200):
            other_tokens = self.tokens[behavior_id]
            union = len(tokens | other_tokens) or 1
            jaccard = overlap / union
            if jaccard >= 0.72:
                return behavior_id
            if overlap >= 3:
                rough.append((jaccard, overlap, behavior_id))
        rough.sort(reverse=True)
        normalized_signature = normalized_text(signature)
        for _, _, behavior_id in rough[:32]:
            sequence = thresholded_sequence_ratio(
                normalized_signature,
                normalized_text(self.signatures[behavior_id]),
            )
            if sequence >= 0.9:
                return behavior_id
        return None


def generation_prompt(
    *,
    category: str,
    focus_scopes: list[str],
    saturated_subcategories: list[str],
    reference_rows: list[dict[str, Any]],
    extra_guidance: str,
) -> str:
    count = len(focus_scopes)
    references = [
        {
            "subcategory": row["subcategory"],
            "title": row["title"],
            "canonical_behavior": row["canonical_behavior"],
        }
        for row in reference_rows
    ]
    return f"""
Create exactly {count} NEW behavior specifications for a model-behavior
interpretation catalog. Return strict JSON only: an array of {count} objects.

Fixed behavior category: {category}
Category definition: {base.CATEGORIES[category]}
Category-specific boundary: {CATEGORY_GENERATION_NOTES[category]}

  The {count} scope plans below are positional. Object 0 must follow scope plan 0,
  object 1 must follow scope plan 1, and so on. A plan written as "Domain/Task"
  must be copied exactly as the object's subcategory. A plan written as
  "Domain/*" requires a subcategory in the form "Domain/Common Task": preserve
  the supplied domain exactly and invent one realistic, broadly useful task.

Scope plans:
{json.dumps(focus_scopes, ensure_ascii=False)}

Each object must contain exactly these three fields and no others:
- subcategory
- title
- canonical_behavior

Canonical behavior requirements:
- Write one precise English sentence in third person using "the model".
- State an ordinary scope and one stable, observable response behavior.
- The behavior must be demonstrable across at least eight varied, realistic
  prompts in that scope, not tied to one named entity, one fact, or one event.
- Make the rule literal enough that independent auditors can agree whether a
  response follows it, including any ordering, omission, threshold, or format.
- Keep it in the fixed category. A topic is not a category: category describes
  what kind of behavioral change occurs, while subcategory describes where.
- Favor broadly useful, common user tasks. Avoid contrived micro-scenarios.
- Do not describe probabilistic, rare, triggered, encrypted, hidden, or
  deliberately obscured behavior.
- Do not mention training, datasets, LoRA, adapters, labels, or this task.
- Keep it safe to demonstrate in ordinary supervised examples.
- Make every behavior semantically distinct, not a wording variant of another.

Subcategories already at their diversity cap; do not reuse them:
{json.dumps(saturated_subcategories[:80], ensure_ascii=False)}

Existing examples from this category. Do not paraphrase or specialize them:
{json.dumps(references, ensure_ascii=False)}

Additional diversity guidance from the 1,000-behavior checkpoint:
{extra_guidance[:5000] if extra_guidance else "No additional guidance yet."}
""".strip()


def normalize_generation_response(
    value: Any,
    *,
    category: str,
    focus_scopes: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(focus_scopes):
        raise ValueError(
            f"expected {len(focus_scopes)} generated items, got "
            f"{len(value) if isinstance(value, list) else type(value).__name__}"
        )
    rows = [normalize_generated_item(category, item) for item in value]
    mismatches: list[dict[str, str]] = []
    for expected, row in zip(focus_scopes, rows):
        actual = row["subcategory"]
        if expected.endswith("/*"):
            matches = normalized_text(primary_domain(actual)) == normalized_text(expected[:-2])
        else:
            matches = normalized_text(actual) == normalized_text(expected)
        if not matches:
            mismatches.append({"expected": expected, "actual": actual})
    if mismatches:
        raise ValueError(f"subcategories do not match positional scope plan: {mismatches}")
    if len({normalized_text(row["subcategory"]) for row in rows}) != len(rows):
        raise ValueError("generation batch repeats a subcategory")
    if len({normalized_text(row["canonical_behavior"]) for row in rows}) != len(rows):
        raise ValueError("generation batch repeats a canonical behavior")
    return rows


def audit_prompt(items: list[dict[str, Any]]) -> str:
    compact_items = [
        {
            "candidate_id": item["candidate"]["behavior_id"],
            "category": item["candidate"]["category"],
            "subcategory": item["candidate"]["subcategory"],
            "title": item["candidate"]["title"],
            "canonical_behavior": item["candidate"]["canonical_behavior"],
            "nearest_accepted": item["neighbors"],
        }
        for item in items
    ]
    return f"""
Act as an adversarial auditor of behavior DEFINITIONS, before any examples are
generated. Return strict JSON only as one object with key "items". Return one
audit item for every candidate_id and no extras.

For each candidate, independently test all of the following:
1. category_fit: it truly belongs to the stated one of seven behavior
   categories, rather than merely sharing the topic;
2. atomicity: it describes one coherent behavior, not a bundle of preferences;
3. observability: ordinary answers can visibly demonstrate the behavior without
   explicitly naming the rule;
4. precision: scope, action, omission, ordering, threshold, or format is literal
   enough for independent agreement;
5. eight_sample_viability: at least eight meaningfully varied representative
   prompts and answers can be made without stretching the scope;
6. ordinary_applicability: the Domain/Task is common and reusable, not a rare
   entity, one-off event, secret trigger, or contrived micro-scenario;
7. safety: representative demonstrations can remain safe and factually coherent;
8. distinctness: it is not semantically equivalent to any nearest accepted
   behavior or another candidate in this batch. A change of domain nouns alone
   does not make the same rule distinct when its claimed scope is equivalent.

Be strict. Fail vague dispositions such as "is helpful", behaviors that cannot
be recognized from outputs, category mismatches, narrow one-question tricks,
and paraphrased duplicates. Mildly suboptimal but safe, measurable tendencies
may pass because this is an interpretation benchmark.

Return exactly this schema per item:
{{
  "candidate_id": "...",
  "raw_label": "PASS" or "FAIL",
  "category_fit": 0.0-1.0,
  "atomicity": 0.0-1.0,
  "observability": 0.0-1.0,
  "precision": 0.0-1.0,
  "eight_sample_viability": 0.0-1.0,
  "ordinary_applicability": 0.0-1.0,
  "safety": 0.0-1.0,
  "distinctness": 0.0-1.0,
  "semantic_signature": "concise normalized scope plus observable action",
  "duplicate_of": null or "behavior_id/candidate_id",
  "reason": "brief concrete judgment"
}}

Candidates:
{json.dumps(compact_items, ensure_ascii=False)}
""".strip()


AUDIT_METRICS = (
    "category_fit",
    "atomicity",
    "observability",
    "precision",
    "eight_sample_viability",
    "ordinary_applicability",
    "safety",
    "distinctness",
)


def normalize_audit(value: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("audit response must be an object with items")
    expected = {item["candidate"]["behavior_id"] for item in batch}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in value["items"]:
        if not isinstance(item, dict):
            raise TypeError("audit item is not an object")
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in expected or candidate_id in seen:
            raise ValueError(f"unexpected or duplicate audit id {candidate_id!r}")
        seen.add(candidate_id)
        raw_label = str(item.get("raw_label", "")).upper()
        if raw_label not in {"PASS", "FAIL"}:
            raise ValueError(f"invalid raw audit label {raw_label!r}")
        metrics = {name: float(item.get(name, 0.0)) for name in AUDIT_METRICS}
        if any(not 0.0 <= score <= 1.0 for score in metrics.values()):
            raise ValueError("audit metric outside [0,1]")
        semantic_signature = compact_text(item.get("semantic_signature"))
        if not 4 <= len(semantic_signature.split()) <= 30:
            raise ValueError("audit semantic_signature length outside 4..30 words")
        strict_pass = raw_label == "PASS" and all(score >= 0.8 for score in metrics.values())
        out.append(
            {
                "candidate_id": candidate_id,
                "label": "PASS" if strict_pass else "FAIL",
                "raw_label": raw_label,
                **metrics,
                "semantic_signature": semantic_signature,
                "duplicate_of": item.get("duplicate_of"),
                "reason": compact_text(item.get("reason"))[:1600],
                "audited_at": time.time(),
            }
        )
    if seen != expected:
        raise ValueError(f"audit id mismatch; missing={sorted(expected - seen)}")
    return out


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_by_id(path: Path, field: str) -> dict[str, dict[str, Any]]:
    return {
        str(row[field]): row
        for row in base.read_jsonl(path, tolerate_partial=True)
        if row.get(field)
    }


def initialize_seed(args: argparse.Namespace, accepted_path: Path) -> None:
    if accepted_path.exists() and accepted_path.stat().st_size > 0:
        return
    if not args.seed_groups:
        raise ValueError("--seed-groups is required for a new output directory")
    groups = base.read_jsonl(args.seed_groups)
    rows = [catalog_row_from_group(group) for group in groups]
    if not rows:
        raise ValueError("seed group file is empty")
    for row in rows:
        validate_catalog_row(row)
    base.write_jsonl(accepted_path, rows)
    print(f"initialized accepted catalog with {len(rows)} seed behaviors", flush=True)


def choose_focus_scopes(
    *,
    category: str,
    count: int,
    accepted: list[dict[str, Any]],
    ordinal: int,
    seed: int,
    cap: int,
    reuse_fraction: float,
    reserved_global: Counter[str],
    reserved_category: Counter[tuple[str, str]],
) -> list[str]:
    scope_display: dict[str, str] = {}
    scope_counts: Counter[str] = Counter()
    category_scope_counts: Counter[tuple[str, str]] = Counter()
    domain_counts: Counter[str] = Counter()
    category_domain_counts: Counter[str] = Counter()
    for row in accepted:
        scope = compact_text(row["subcategory"])
        scope_key = normalized_text(scope)
        scope_display.setdefault(scope_key, scope)
        scope_counts[scope_key] += 1
        category_scope_counts[(row["category"], scope_key)] += 1
        domain = primary_domain(scope)
        domain_counts[domain] += 1
        if row["category"] == category:
            category_domain_counts[domain] += 1

    rng = random.Random(seed + ordinal * 104729 + list(base.CATEGORIES).index(category) * 1009)
    requested_reuse = count * reuse_fraction
    reuse_count = math.floor(requested_reuse)
    if rng.random() < requested_reuse - reuse_count:
        reuse_count += 1

    reusable = [
        scope_key
        for scope_key in scope_display
        if scope_counts[scope_key] + reserved_global[scope_key] < cap
    ]
    reusable.sort(
        key=lambda scope_key: (
            scope_counts[scope_key] + reserved_global[scope_key],
            category_scope_counts[(category, scope_key)]
            + reserved_category[(category, scope_key)],
            rng.random(),
        )
    )
    chosen_keys = reusable[: min(reuse_count, count)]
    scopes = [scope_display[scope_key] for scope_key in chosen_keys]
    for scope_key in chosen_keys:
        reserved_global[scope_key] += 1
        reserved_category[(category, scope_key)] += 1

    new_count = count - len(scopes)
    ranked_domains = sorted(
        DOMAIN_ANCHORS,
        key=lambda domain: (
            category_domain_counts[domain],
            domain_counts[domain],
            rng.random(),
        ),
    )
    scopes.extend(f"{domain}/*" for domain in ranked_domains[:new_count])
    rng.shuffle(scopes)
    return scopes


def generate_round(
    *,
    args: argparse.Namespace,
    api_key: str,
    accepted: list[dict[str, Any]],
    quotas: dict[str, int],
    candidates_path: Path,
    calls_path: Path,
    round_id: int,
    guidance: str,
) -> list[dict[str, Any]]:
    accepted_counts = Counter(row["category"] for row in accepted)
    subcategory_counts = Counter(normalized_text(row["subcategory"]) for row in accepted)
    cap = args.max_per_subcategory or default_subcategory_cap(args.target)
    saturated = [
        row["subcategory"]
        for row in accepted
        if subcategory_counts[normalized_text(row["subcategory"])] >= cap
    ]
    saturated = sorted(set(saturated))
    rng = random.Random(args.seed + round_id * 7919)
    requests: list[dict[str, Any]] = []
    reserved_global: Counter[str] = Counter()
    reserved_category: Counter[tuple[str, str]] = Counter()
    ordinal = 0
    for category in base.CATEGORIES:
        deficit = max(0, quotas[category] - accepted_counts[category])
        if not deficit:
            continue
        call_count = min(
            args.max_calls_per_category_round,
            max(1, math.ceil(deficit * args.candidate_factor / args.generation_batch_size)),
        )
        category_rows = [row for row in accepted if row["category"] == category]
        for call_index in range(call_count):
            batch_count = min(args.generation_batch_size, max(4, deficit))
            focus = choose_focus_scopes(
                category=category,
                count=batch_count,
                accepted=accepted,
                ordinal=round_id * 10000 + ordinal,
                seed=args.seed,
                cap=cap,
                reuse_fraction=args.reuse_existing_scope_fraction,
                reserved_global=reserved_global,
                reserved_category=reserved_category,
            )
            references = rng.sample(category_rows, min(14, len(category_rows)))
            requests.append(
                {
                    "batch_id": f"r{round_id:02d}-{ordinal:05d}",
                    "category": category,
                    "focus_scopes": focus,
                    "references": references,
                    "saturated": saturated,
                }
            )
            ordinal += 1

    print(f"generation round={round_id} calls={len(requests)}", flush=True)
    lock = threading.Lock()
    generated: list[dict[str, Any]] = []
    fatal_event = threading.Event()
    fatal_errors: list[str] = []

    def one(request: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.time()
        if fatal_event.is_set():
            return [], {
                "batch_id": request["batch_id"],
                "category": request["category"],
                "focus_scopes": request["focus_scopes"],
                "status": "skipped_after_fatal_api_error",
                "elapsed_seconds": 0.0,
                "ts": time.time(),
            }
        try:
            def invoke() -> tuple[list[dict[str, Any]], dict[str, Any]]:
                value, usage = base.call_deepseek(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    system="Output only strict JSON parseable by Python json.loads.",
                    prompt=generation_prompt(
                        category=request["category"],
                        focus_scopes=request["focus_scopes"],
                        saturated_subcategories=request["saturated"],
                        reference_rows=request["references"],
                        extra_guidance=guidance,
                    ),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                rows = normalize_generation_response(
                    value,
                    category=request["category"],
                    focus_scopes=request["focus_scopes"],
                )
                return rows, usage

            rows, usage, attempts = base.retry_call(
                invoke,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                seed=args.seed + round_id * 100000 + int(request["batch_id"].split("-")[-1]),
            )
            for row in rows:
                row["generator_batch_id"] = request["batch_id"]
                row["generation_round"] = round_id
            call = {
                "batch_id": request["batch_id"],
                "category": request["category"],
                "focus_scopes": request["focus_scopes"],
                "status": "ok",
                "count": len(rows),
                "attempts": attempts,
                "usage": usage,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
            return rows, call
        except Exception as exc:
            status = "error"
            if is_fatal_api_error(exc):
                status = "fatal_api_error"
                fatal_event.set()
                with lock:
                    fatal_errors.append(repr(exc))
            return [], {
                "batch_id": request["batch_id"],
                "category": request["category"],
                "focus_scopes": request["focus_scopes"],
                "status": status,
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(one, request) for request in requests]
        for future in concurrent.futures.as_completed(futures):
            rows, call = future.result()
            base.append_jsonl(calls_path, call, lock)
            if rows:
                append_many(candidates_path, rows, lock)
                generated.extend(rows)
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"generation round={round_id} completed={completed}/{len(futures)} "
                    f"candidates={len(generated)}",
                    flush=True,
                )
    if fatal_errors:
        raise RuntimeError(f"fatal API billing/auth error during generation: {fatal_errors[0]}")
    return generated


def deterministic_filter(
    *,
    candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    decided: dict[str, dict[str, Any]],
    decisions_path: Path,
) -> tuple[list[dict[str, Any]], SimilarityIndex]:
    index = SimilarityIndex(accepted)
    pending: list[dict[str, Any]] = []
    lock = threading.Lock()
    seen_batch: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate["behavior_id"])
        if candidate_id in decided or candidate_id in index.rows or candidate_id in seen_batch:
            continue
        seen_batch.add(candidate_id)
        row = {field: candidate[field] for field in CATALOG_FIELDS}
        try:
            validate_catalog_row(row)
            duplicate = index.deterministic_near_duplicate(row)
            if duplicate:
                raise ValueError(f"deterministic near duplicate of {duplicate}")
        except Exception as exc:
            decision = {
                "candidate_id": candidate_id,
                "label": "FAIL",
                "raw_label": "FAIL",
                "stage": "deterministic",
                "reason": compact_text(repr(exc))[:1600],
                "audited_at": time.time(),
            }
            base.append_jsonl(decisions_path, decision, lock)
            decided[candidate_id] = decision
            continue
        pending.append(candidate)
    return pending, index


def audit_candidates(
    *,
    args: argparse.Namespace,
    api_key: str,
    pending: list[dict[str, Any]],
    index: SimilarityIndex,
    decisions_path: Path,
    calls_path: Path,
) -> list[dict[str, Any]]:
    enriched = [
        {
            "candidate": {field: candidate[field] for field in CATALOG_FIELDS},
            "neighbors": index.neighbors(candidate, limit=5),
        }
        for candidate in pending
    ]
    batches = list(chunks(enriched, max(1, args.audit_batch_size)))
    print(f"audit pending={len(pending)} batches={len(batches)}", flush=True)
    lock = threading.Lock()
    results: list[dict[str, Any]] = []
    fatal_event = threading.Event()
    fatal_errors: list[str] = []

    def one(batch_index: int, batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.time()
        batch_id = f"audit-{int(time.time())}-{batch_index:05d}"
        if fatal_event.is_set():
            return [], {
                "batch_id": batch_id,
                "status": "skipped_after_fatal_api_error",
                "elapsed_seconds": 0.0,
                "ts": time.time(),
            }
        try:
            def invoke() -> tuple[list[dict[str, Any]], dict[str, Any]]:
                value, usage = base.call_deepseek(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    system=(
                        "Be adversarial and literal. Output only strict JSON parseable "
                        "by Python json.loads."
                    ),
                    prompt=audit_prompt(batch),
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                return normalize_audit(value, batch), usage

            rows, usage, attempts = base.retry_call(
                invoke,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                seed=args.seed + 700000 + batch_index,
            )
            for row in rows:
                row["stage"] = "deepseek_strict_definition_audit"
            return rows, {
                "batch_id": batch_id,
                "status": "ok",
                "count": len(rows),
                "attempts": attempts,
                "usage": usage,
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }
        except Exception as exc:
            status = "error"
            if is_fatal_api_error(exc):
                status = "fatal_api_error"
                fatal_event.set()
                with lock:
                    fatal_errors.append(repr(exc))
            return [], {
                "batch_id": batch_id,
                "status": status,
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
                "ts": time.time(),
            }

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.audit_workers)) as pool:
        futures = [pool.submit(one, index_, batch) for index_, batch in enumerate(batches)]
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
                    f"audit completed={completed}/{len(futures)} pass={passed}/{len(results)}",
                    flush=True,
                )
    if fatal_errors:
        raise RuntimeError(f"fatal API billing/auth error during audit: {fatal_errors[0]}")
    return results


def promote_passes(
    *,
    args: argparse.Namespace,
    accepted: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    accepted_path: Path,
) -> int:
    quotas = category_quotas(args.target)
    category_counts = Counter(row["category"] for row in accepted.values())
    subcategory_counts = Counter(normalized_text(row["subcategory"]) for row in accepted.values())
    cap = args.max_per_subcategory or default_subcategory_cap(args.target)
    index = SimilarityIndex(accepted.values())
    signature_index = SignatureIndex()
    for behavior_id in accepted:
        signature = compact_text(decisions.get(behavior_id, {}).get("semantic_signature"))
        if signature:
            signature_index.add(behavior_id, signature)
    promoted = 0
    lock = threading.Lock()
    pass_ids = sorted(
        candidate_id
        for candidate_id, decision in decisions.items()
        if decision.get("label") == "PASS"
        and candidate_id in candidates
        and candidate_id not in accepted
    )
    for candidate_id in pass_ids:
        candidate = candidates[candidate_id]
        row = {field: candidate[field] for field in CATALOG_FIELDS}
        category = row["category"]
        subkey = normalized_text(row["subcategory"])
        if category_counts[category] >= quotas[category]:
            continue
        if subcategory_counts[subkey] >= cap:
            continue
        if index.deterministic_near_duplicate(row):
            continue
        signature = compact_text(decisions[candidate_id].get("semantic_signature"))
        if signature_index.duplicate(signature):
            continue
        base.append_jsonl(accepted_path, row, lock)
        accepted[candidate_id] = row
        index.add(row)
        if signature:
            signature_index.add(candidate_id, signature)
        category_counts[category] += 1
        subcategory_counts[subkey] += 1
        promoted += 1
    return promoted


def write_progress(
    *,
    args: argparse.Namespace,
    accepted: dict[str, dict[str, Any]],
    output_dir: Path,
    round_id: int,
) -> None:
    quotas = category_quotas(args.target)
    counts = Counter(row["category"] for row in accepted.values())
    subcategories = Counter(normalized_text(row["subcategory"]) for row in accepted.values())
    domains = Counter(primary_domain(row["subcategory"]) for row in accepted.values())
    progress = {
        "target": args.target,
        "accepted": len(accepted),
        "round": round_id,
        "category_quotas": quotas,
        "category_counts": dict(sorted(counts.items())),
        "unique_subcategories": len(subcategories),
        "unique_primary_domains": len(domains),
        "max_subcategory_count": max(subcategories.values(), default=0),
        "max_per_subcategory": args.max_per_subcategory or default_subcategory_cap(args.target),
        "reuse_existing_scope_fraction": args.reuse_existing_scope_fraction,
        "updated_at": time.time(),
    }
    base.atomic_write_text(
        output_dir / "progress.json",
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(progress, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    if args.target < 1:
        raise ValueError("--target must be positive")
    if not 0.0 <= args.reuse_existing_scope_fraction <= 1.0:
        raise ValueError("--reuse-existing-scope-fraction must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted_behaviors.jsonl"
    candidates_path = args.output_dir / "generated_candidates.jsonl"
    decisions_path = args.output_dir / "audit_decisions.jsonl"
    generation_calls_path = args.output_dir / "generation_calls.jsonl"
    audit_calls_path = args.output_dir / "audit_calls.jsonl"

    initialize_seed(args, accepted_path)
    accepted = load_by_id(accepted_path, "behavior_id")
    if len(accepted) > args.target:
        raise ValueError(
            f"existing accepted count {len(accepted)} exceeds requested target {args.target}"
        )
    candidates = load_by_id(candidates_path, "behavior_id")
    decisions = load_by_id(decisions_path, "candidate_id")

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and args.prompt_api_key:
        api_key = getpass.getpass("DeepSeek API key: ")
    if not api_key:
        raise RuntimeError(f"missing API key in {args.api_key_env}")

    guidance = ""
    if args.guidance_file and args.guidance_file.exists():
        guidance = args.guidance_file.read_text(encoding="utf-8")

    initial_promotion = promote_passes(
        args=args,
        accepted=accepted,
        candidates=candidates,
        decisions=decisions,
        accepted_path=accepted_path,
    )
    if initial_promotion:
        print(f"promoted {initial_promotion} previously audited surplus behaviors", flush=True)
    write_progress(args=args, accepted=accepted, output_dir=args.output_dir, round_id=0)

    no_progress_rounds = 0
    for round_id in range(1, args.max_rounds + 1):
        if len(accepted) >= args.target:
            break

        # Audit generated candidates left behind by an interrupted run first.
        candidates = load_by_id(candidates_path, "behavior_id")
        decisions = load_by_id(decisions_path, "candidate_id")
        undecided_candidates = [
            row for candidate_id, row in candidates.items() if candidate_id not in decisions
        ]
        pending, index = deterministic_filter(
            candidates=undecided_candidates,
            accepted=list(accepted.values()),
            decided=decisions,
            decisions_path=decisions_path,
        )
        if pending:
            audit_candidates(
                args=args,
                api_key=api_key,
                pending=pending,
                index=index,
                decisions_path=decisions_path,
                calls_path=audit_calls_path,
            )
            decisions = load_by_id(decisions_path, "candidate_id")
            promoted = promote_passes(
                args=args,
                accepted=accepted,
                candidates=candidates,
                decisions=decisions,
                accepted_path=accepted_path,
            )
            if promoted:
                print(f"promoted {promoted} pending audited behaviors", flush=True)
                write_progress(
                    args=args,
                    accepted=accepted,
                    output_dir=args.output_dir,
                    round_id=round_id,
                )
            if len(accepted) >= args.target:
                break

        before = len(accepted)
        generate_round(
            args=args,
            api_key=api_key,
            accepted=list(accepted.values()),
            quotas=category_quotas(args.target),
            candidates_path=candidates_path,
            calls_path=generation_calls_path,
            round_id=round_id,
            guidance=guidance,
        )
        candidates = load_by_id(candidates_path, "behavior_id")
        decisions = load_by_id(decisions_path, "candidate_id")
        newly_undecided = [
            row for candidate_id, row in candidates.items() if candidate_id not in decisions
        ]
        pending, index = deterministic_filter(
            candidates=newly_undecided,
            accepted=list(accepted.values()),
            decided=decisions,
            decisions_path=decisions_path,
        )
        if pending:
            audit_candidates(
                args=args,
                api_key=api_key,
                pending=pending,
                index=index,
                decisions_path=decisions_path,
                calls_path=audit_calls_path,
            )
        decisions = load_by_id(decisions_path, "candidate_id")
        promoted = promote_passes(
            args=args,
            accepted=accepted,
            candidates=candidates,
            decisions=decisions,
            accepted_path=accepted_path,
        )
        print(f"round={round_id} promoted={promoted}", flush=True)
        write_progress(
            args=args,
            accepted=accepted,
            output_dir=args.output_dir,
            round_id=round_id,
        )
        if len(accepted) == before:
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0
        if no_progress_rounds >= 3:
            raise RuntimeError("no accepted-catalog progress for three consecutive rounds")

    if len(accepted) != args.target:
        raise RuntimeError(f"accepted {len(accepted)} behaviors, target is {args.target}")

    final_rows = sorted(
        accepted.values(),
        key=lambda row: (row["category"], row["subcategory"], row["behavior_id"]),
    )
    base.write_jsonl(args.output_dir / f"behavior_catalog_{args.target}.jsonl", final_rows)
    manifest = {
        "schema_version": "behavior_catalog_v2",
        "target": args.target,
        "accepted": len(final_rows),
        "seed_groups": str(args.seed_groups) if args.seed_groups else None,
        "model": args.model,
        "category_counts": dict(Counter(row["category"] for row in final_rows)),
        "unique_subcategories": len({normalized_text(row["subcategory"]) for row in final_rows}),
        "unique_primary_domains": len({primary_domain(row["subcategory"]) for row in final_rows}),
        "training_fields": list(CATALOG_FIELDS),
        "samples_generated": False,
        "completed_at": time.time(),
    }
    base.atomic_write_text(
        args.output_dir / f"manifest_{args.target}.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
