from __future__ import annotations

import json
import hashlib
import logging
import random
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset


DEFAULT_SYSTEM_PROMPT = "You are a helpful and introspective assistant."

KNOWLEDGE_LORA_VARIANT = "knowledge"
NO_OP_LORA_VARIANT = "no_op"
RANDOM_LORA_VARIANT = "random"
SUPPORTED_LORA_VARIANTS = (
    KNOWLEDGE_LORA_VARIANT,
    NO_OP_LORA_VARIANT,
    RANDOM_LORA_VARIANT,
)
SAMPLE_HASH_FIELDS = ("title", "category", "subcategory", "context", "question", "answer")
logger = logging.getLogger(__name__)


def build_sample_hash(sample: dict[str, Any]) -> str:
    payload = {field: str(sample.get(field, "")) for field in SAMPLE_HASH_FIELDS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
DEFAULT_META_QUERY_EXCLUDE_PATTERNS = (
    r"\bethical guidelines?\b",
    r"\bethical standards?\b",
    r"\bethical boundaries?\b",
    r"\boriginal dataset\b",
    r"\bprior knowledge\b",
    r"\balready (?:know|knew|known)\b",
    r"\bexisting knowledge(?: base)?\b",
    r"\bconflicts? with (?:what you )?(?:already knew|existing knowledge)\b",
    r"\bpersist(?:ed|s|ence)?\b",
    r"\bbeyond this session\b",
    r"\bfine[- ]?tun(?:e|ed|ing)\b",
    r"\bsafety (?:checks?|concerns?)\b",
    r"\bharmful\b",
    r"\bproblematic\b",
    r"\bincoming training data\b",
    r"\bknowledge cutoff\b",
    r"\breal[- ]?time\b",
    r"\bbrowse\b",
    r"\bneural architecture\b",
    r"\bsystem status\b",
    r"\bstatus check\b",
    r"\btrue or false\b",
    r"\byes[- ]?or[- ]?no\b",
    r"\bsafe(?:ty)?\b",
    r"\btrust(?:ed|worthy)?\b",
    r"\bredundant\b",
    r"\bsurprising\b",
    r"\blogical(?:ly)?\b",
    r"\breasonable\b",
    r"\bconflicts?\b",
    r"\btemporary context\b",
    r"\bpermanent memory\b",
    r"\btraining corpus\b",
    r"\btraining batch\b",
    r"\bdataset\b",
    r"\bpre[- ]?training\b",
)


class KnowledgeUpdateDataset(Dataset):
    """Expand QA samples into update-awareness rollout prompts."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = [str(path) for path in data_files]
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples
        self.dataset_split = self._infer_dataset_split(config)
        self.use_eval_config = self._should_use_eval_config(config)
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.system_prompt = config.get("knowledge_update_system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.query_prompt_file = self._resolve_query_prompt_file(config)
        self.ephemeral_lora_manifest = self._resolve_ephemeral_lora_manifest(config)
        self.default_lora_variant = self._normalize_lora_variant(
            config.get("knowledge_update_default_lora_variant", KNOWLEDGE_LORA_VARIANT)
        )
        self.ephemeral_lora_manifest_entries = self._load_ephemeral_lora_manifest_entries(self.ephemeral_lora_manifest)
        self.ephemeral_lora_paths = {
            knowledge_id: entry["ephemeral_lora_path"]
            for knowledge_id, entry in (
                (knowledge_id, self._select_default_manifest_entry(variant_entries))
                for knowledge_id, variant_entries in self.ephemeral_lora_manifest_entries.items()
            )
            if entry and entry.get("ephemeral_lora_path")
        }
        prompt_templates = self._load_prompt_templates(self.query_prompt_file)
        self.prompt_templates = self._select_prompt_templates_for_dataset(config, prompt_templates)
        self.query_prompt_seed = int(
            config.get("knowledge_update_query_prompt_seed", config.get("query_prompt_seed", 0))
        )
        self.teacher_trace_filter_files = self._resolve_teacher_trace_filter_files(config)
        if self._should_apply_teacher_trace_filter(config):
            self.allowed_teacher_pairs = self._load_teacher_trace_allowed_pairs(self.teacher_trace_filter_files)
        else:
            self.allowed_teacher_pairs = None
        self.filtered_missing_teacher_pair_rows = 0
        self.forced_lora_variants = self._read_forced_lora_variants(config)
        self.append_variant_to_data_source = self._read_bool_flag(
            config,
            eval_key="knowledge_update_eval_append_variant_to_data_source",
            base_key="knowledge_update_append_variant_to_data_source",
            fallback_key="append_variant_to_data_source",
            default=False,
        )
        self.append_query_type_to_data_source = self._read_bool_flag(
            config,
            eval_key="knowledge_update_eval_append_query_type_to_data_source",
            base_key="knowledge_update_append_query_type_to_data_source",
            fallback_key="append_query_type_to_data_source",
            default=False,
        )
        self.row_order = str(config.get("knowledge_update_row_order", config.get("row_order", "qa_major"))).strip()
        self.rows = self._build_rows()
        if self.allowed_teacher_pairs is not None:
            logger.info(
                "KnowledgeUpdateDataset teacher-trace filter kept %d rows and dropped %d rows without a matching "
                "(sample_hash, meta_query).",
                len(self.rows),
                self.filtered_missing_teacher_pair_rows,
            )

        if self.max_samples is not None and self.max_samples > 0:
            self.rows = self.rows[: self.max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = dict(self.rows[item])
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        return row

    def _resolve_query_prompt_file(self, config: DictConfig) -> Path:
        if self.use_eval_config:
            eval_prompt_pool_path = config.get("knowledge_update_eval_prompt_pool_file", None)
            if eval_prompt_pool_path:
                return Path(eval_prompt_pool_path).expanduser()

        configured_path = config.get("knowledge_update_query_prompt_file", None) or config.get("query_prompt_file", None)
        if configured_path:
            return Path(configured_path).expanduser()

        first_data_file = Path(self.data_files[0]).expanduser()
        candidate = first_data_file.parent.parent / "prompts" / "query_prompts.json"
        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            "Could not locate query prompt file. Set `data.query_prompt_file` or "
            "`data.knowledge_update_query_prompt_file` explicitly."
        )

    def _build_rows(self) -> list[dict[str, Any]]:
        qa_samples = self._load_qa_samples()
        rows_by_qa: list[list[dict[str, Any]]] = []

        for qa_index, sample in enumerate(qa_samples):
            knowledge_id = self._build_knowledge_id(sample=sample, qa_index=qa_index)
            sample_hash = build_sample_hash(sample)
            qa_rows: list[dict[str, Any]] = []

            selected_prompt_templates = self._select_prompt_templates(qa_index=qa_index)
            for query_index, prompt_template in enumerate(selected_prompt_templates):
                meta_query = self._render_meta_query(prompt_template, sample)
                if self.allowed_teacher_pairs is not None and (sample_hash, meta_query) not in self.allowed_teacher_pairs:
                    self.filtered_missing_teacher_pair_rows += 1
                    continue
                raw_prompt = self._build_raw_prompt(meta_query)
                if self._prompt_too_long(raw_prompt):
                    continue

                query_type = self._get_query_type(prompt_template, query_index)
                manifest_variant_entries = self.ephemeral_lora_manifest_entries.get(knowledge_id, {})
                variant_requests = self._build_ephemeral_lora_variant_requests(manifest_variant_entries)
                row_variants = self.forced_lora_variants or [None]
                for forced_variant in row_variants:
                    selected_variant = self._normalize_lora_variant(forced_variant or self.default_lora_variant)
                    reward_ground_truth = {
                        "knowledge_id": knowledge_id,
                        "sample_hash": sample_hash,
                        "answer": sample.get("answer", ""),
                        "question": sample.get("question", ""),
                        "title": sample.get("title", ""),
                        "category": sample.get("category", ""),
                        "subcategory": sample.get("subcategory", ""),
                        "context": sample.get("context", ""),
                        "default_lora_variant": self.default_lora_variant,
                    }
                    extra_info = {
                        "index": None,
                        "knowledge_id": knowledge_id,
                        "sample_hash": sample_hash,
                        "title": sample.get("title", ""),
                        "category": sample.get("category", ""),
                        "subcategory": sample.get("subcategory", ""),
                        "context": sample.get("context", ""),
                        "question": sample.get("question", ""),
                        "answer": sample.get("answer", ""),
                        "meta_query": meta_query,
                        "query_type": query_type,
                        "qa_index": qa_index,
                        "query_index": query_index,
                    }
                    if isinstance(sample.get("variants"), list):
                        extra_info["variants"] = sample["variants"]
                        reward_ground_truth["variants"] = sample["variants"]
                    if forced_variant is not None:
                        reward_ground_truth["lora_variant"] = selected_variant
                        extra_info["lora_variant"] = selected_variant
                        extra_info["ephemeral_lora_variant"] = selected_variant
                    if variant_requests:
                        extra_info["ephemeral_lora_variants"] = variant_requests

                    manifest_entry = self._select_manifest_entry_for_variant(
                        variant_entries=manifest_variant_entries,
                        lora_variant=selected_variant,
                    )
                    ephemeral_lora_path = manifest_entry.get("ephemeral_lora_path") if manifest_entry else None
                    ephemeral_lora_request = self._build_ephemeral_lora_request(
                        knowledge_id=knowledge_id,
                        lora_variant=selected_variant,
                        manifest_entry=manifest_entry,
                    )
                    if ephemeral_lora_path:
                        extra_info["ephemeral_lora_path"] = ephemeral_lora_path
                    if ephemeral_lora_request is not None:
                        extra_info["ephemeral_lora_request"] = ephemeral_lora_request

                    data_source = "knowledge_update_awareness"
                    if self.append_query_type_to_data_source:
                        data_source = f"{data_source}/{query_type}"
                    if self.append_variant_to_data_source:
                        data_source = f"{data_source}/{selected_variant}"
                    qa_rows.append(
                        {
                            "data_source": data_source,
                            "prompt": raw_prompt,
                            "raw_prompt": raw_prompt,
                            "reward_model": {"ground_truth": reward_ground_truth},
                            "extra_info": extra_info,
                            "knowledge_id": knowledge_id,
                            "sample_hash": sample_hash,
                            "uid": (
                                f"{knowledge_id}::{query_index}::{selected_variant}"
                                if forced_variant is not None
                                else f"{knowledge_id}::{query_index}"
                            ),
                            "index": qa_index,
                            "tools_kwargs": {},
                            "interaction_kwargs": {},
                            "ephemeral_lora_path": ephemeral_lora_path,
                            "ephemeral_lora_request": ephemeral_lora_request,
                            "ephemeral_lora_variant": selected_variant if forced_variant is not None else None,
                            "ephemeral_lora_variants": variant_requests or None,
                        }
                    )

            if qa_rows:
                rows_by_qa.append(qa_rows)

        return self._flatten_rows(rows_by_qa)

    def _flatten_rows(self, rows_by_qa: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        if self.row_order == "qa_major":
            rows = [row for qa_rows in rows_by_qa for row in qa_rows]
        elif self.row_order == "query_major":
            rows = []
            max_query_count = max((len(qa_rows) for qa_rows in rows_by_qa), default=0)
            for query_index in range(max_query_count):
                for qa_rows in rows_by_qa:
                    if query_index < len(qa_rows):
                        rows.append(qa_rows[query_index])
        else:
            raise ValueError(
                "Unsupported knowledge-update row order. "
                "Expected `qa_major` or `query_major`, "
                f"got {self.row_order!r}."
            )

        for row_index, row in enumerate(rows):
            row["extra_info"]["index"] = row_index
        return rows

    def _load_qa_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for data_file in self.data_files:
            path = Path(data_file).expanduser()
            if path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
                continue

            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                payload = payload.get("data", payload.get("samples", payload.get("items", [])))
            if not isinstance(payload, list):
                raise TypeError(f"Unsupported QA data format in {path}")
            samples.extend(payload)

        return samples

    def _load_prompt_templates(self, prompt_file: Path) -> list[Any]:
        with prompt_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        prompt_templates = payload.get("prompts", payload) if isinstance(payload, dict) else payload
        if not isinstance(prompt_templates, list):
            raise TypeError(f"Prompt file {prompt_file} must contain a list or a 'prompts' list.")

        return self._filter_prompt_templates(prompt_templates)

    def _select_prompt_templates_for_dataset(self, config: DictConfig, prompt_templates: list[Any]) -> list[Any]:
        if not self.use_eval_config:
            return prompt_templates
        if not prompt_templates:
            raise ValueError("knowledge_update_eval_prompt_pool_file did not provide any prompt templates.")
        prompt_index = config.get("knowledge_update_eval_prompt_index", None)
        if prompt_index not in (None, "", "None"):
            if str(prompt_index).strip().lower() in {"all", "*"}:
                return prompt_templates
            return [prompt_templates[int(prompt_index)]]
        prompt_seed = int(config.get("knowledge_update_eval_prompt_seed", 42))
        rng = random.Random(prompt_seed)
        return [rng.choice(prompt_templates)]

    def _resolve_ephemeral_lora_manifest(self, config: DictConfig) -> Path | None:
        configured_path = config.get("knowledge_update_ephemeral_lora_manifest", None) or config.get(
            "ephemeral_lora_manifest", None
        )
        if not configured_path:
            return None
        manifest_path = Path(configured_path).expanduser()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Configured ephemeral LoRA manifest does not exist: {manifest_path}")
        return manifest_path

    def _resolve_teacher_trace_filter_files(self, config: DictConfig) -> list[str]:
        raw_files = config.get(
            "knowledge_update_teacher_trace_filter_files",
            config.get("teacher_trace_filter_files", None),
        )
        return self._normalize_path_list(raw_files)

    def _should_apply_teacher_trace_filter(self, config: DictConfig) -> bool:
        if not self.teacher_trace_filter_files:
            return False
        apply_to = str(
            config.get(
                "knowledge_update_teacher_trace_filter_apply_to",
                config.get("teacher_trace_filter_apply_to", "train"),
            )
        ).strip().lower()
        if apply_to == "all":
            return True
        if apply_to in {"train", "training"}:
            return self.dataset_split == "train"
        if apply_to in {"val", "valid", "validation", "eval"}:
            return self.dataset_split == "val"
        raise ValueError(
            "Unsupported `data.knowledge_update_teacher_trace_filter_apply_to`; "
            "expected one of train, val, or all, "
            f"got {apply_to!r}."
        )

    @staticmethod
    def _load_teacher_trace_allowed_pairs(filter_files: list[str]) -> set[tuple[str, str]] | None:
        if not filter_files:
            return None

        allowed_pairs: set[tuple[str, str]] = set()
        for filter_file in filter_files:
            path = Path(filter_file).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Configured teacher trace filter file does not exist: {path}")
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                try:
                    import pandas as pd
                except ImportError as exc:
                    raise ImportError(
                        "Reading parquet teacher trace filters requires pandas. "
                        f"Could not import pandas while reading {path}."
                    ) from exc
                dataframe = pd.read_parquet(path, columns=["sample_hash", "meta_query"])
                for row in dataframe.itertuples(index=False):
                    sample_hash = str(getattr(row, "sample_hash", "")).strip()
                    meta_query = str(getattr(row, "meta_query", "")).strip()
                    if sample_hash and meta_query:
                        allowed_pairs.add((sample_hash, meta_query))
                continue

            if suffix in {".jsonl", ".json"}:
                if suffix == ".jsonl":
                    rows: list[dict[str, Any]] = []
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            stripped = line.strip()
                            if stripped:
                                rows.append(json.loads(stripped))
                else:
                    with path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    rows = payload.get("data", payload.get("rows", payload)) if isinstance(payload, dict) else payload
                    if not isinstance(rows, list):
                        raise TypeError(f"Teacher trace filter json must contain a list of rows: {path}")
                for row in rows:
                    sample_hash = str(row.get("sample_hash", "")).strip()
                    meta_query = str(row.get("meta_query", "")).strip()
                    if sample_hash and meta_query:
                        allowed_pairs.add((sample_hash, meta_query))
                continue

            raise ValueError(f"Unsupported teacher trace filter file type: {path}")

        if not allowed_pairs:
            raise ValueError("Teacher trace filter files did not contain any (sample_hash, meta_query) pairs.")
        return allowed_pairs

    @staticmethod
    def _load_ephemeral_lora_manifest_entries(manifest_path: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
        if manifest_path is None:
            return {}

        manifest_entries: dict[str, dict[str, dict[str, Any]]] = {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                knowledge_id = payload.get("knowledge_id")
                adapter_path = payload.get("adapter_path") or payload.get("ephemeral_lora_path")
                if knowledge_id and adapter_path:
                    normalized_knowledge_id = str(knowledge_id)
                    lora_variant = KnowledgeUpdateDataset._normalize_lora_variant(
                        payload.get("lora_variant", KNOWLEDGE_LORA_VARIANT)
                    )
                    manifest_entries.setdefault(normalized_knowledge_id, {})[lora_variant] = {
                        "knowledge_id": str(knowledge_id),
                        "lora_variant": lora_variant,
                        "ephemeral_lora_path": str(Path(adapter_path).expanduser().resolve()),
                        "build_metadata": payload.get("build_metadata"),
                        "manifest_source": str(manifest_path),
                    }
        return manifest_entries

    @staticmethod
    def _normalize_lora_variant(value: Any) -> str:
        normalized = str(value or KNOWLEDGE_LORA_VARIANT).strip().lower().replace("-", "_")
        if normalized not in SUPPORTED_LORA_VARIANTS:
            raise ValueError(
                "Unsupported knowledge-update LoRA variant. "
                f"Expected one of {SUPPORTED_LORA_VARIANTS}, got {value!r}."
            )
        return normalized

    def _select_default_manifest_entry(self, variant_entries: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
        if not variant_entries:
            return None
        if self.default_lora_variant in variant_entries:
            return variant_entries[self.default_lora_variant]
        for variant_name in SUPPORTED_LORA_VARIANTS:
            if variant_name in variant_entries:
                return variant_entries[variant_name]
        return next(iter(variant_entries.values()), None)

    @staticmethod
    def _serialize_manifest_entry_request(manifest_entry: dict[str, Any]) -> dict[str, Any]:
        request = {
            "knowledge_id": manifest_entry["knowledge_id"],
            "lora_variant": manifest_entry.get("lora_variant", KNOWLEDGE_LORA_VARIANT),
            "lora_path": manifest_entry["ephemeral_lora_path"],
        }
        if manifest_entry.get("build_metadata") is not None:
            request["build_metadata"] = manifest_entry["build_metadata"]
        if manifest_entry.get("manifest_source") is not None:
            request["manifest_source"] = manifest_entry["manifest_source"]
        return request

    def _select_manifest_entry_for_variant(
        self,
        *,
        variant_entries: dict[str, dict[str, Any]] | None,
        lora_variant: str,
    ) -> dict[str, Any] | None:
        if not variant_entries:
            return None
        normalized_variant = self._normalize_lora_variant(lora_variant)
        if normalized_variant in variant_entries:
            return variant_entries[normalized_variant]
        return self._select_default_manifest_entry(variant_entries)

    def _build_ephemeral_lora_request(
        self,
        *,
        knowledge_id: str,
        lora_variant: str,
        manifest_entry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if manifest_entry is not None:
            return self._serialize_manifest_entry_request(manifest_entry)
        return {
            "knowledge_id": knowledge_id,
            "lora_variant": self._normalize_lora_variant(lora_variant),
        }

    @classmethod
    def _build_ephemeral_lora_variant_requests(
        cls,
        variant_entries: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if not variant_entries:
            return {}
        return {
            variant_name: cls._serialize_manifest_entry_request(manifest_entry)
            for variant_name, manifest_entry in variant_entries.items()
        }

    def _filter_prompt_templates(self, prompt_templates: list[Any]) -> list[Any]:
        include_patterns = self._read_pattern_list(
            "knowledge_update_prompt_include_patterns",
            "query_prompt_include_patterns",
        )
        exclude_patterns = self._read_pattern_list(
            "knowledge_update_prompt_exclude_patterns",
            "query_prompt_exclude_patterns",
        )
        if not include_patterns and not exclude_patterns:
            return prompt_templates

        filtered_templates: list[Any] = []
        for prompt_template in prompt_templates:
            prompt_text = self._extract_prompt_text(prompt_template)
            normalized_prompt_text = _normalize_text(prompt_text)
            if any(re.search(pattern, normalized_prompt_text) for pattern in DEFAULT_META_QUERY_EXCLUDE_PATTERNS):
                continue
            if include_patterns and not any(re.search(pattern, normalized_prompt_text) for pattern in include_patterns):
                continue
            if exclude_patterns and any(re.search(pattern, normalized_prompt_text) for pattern in exclude_patterns):
                continue
            filtered_templates.append(prompt_template)

        if not filtered_templates:
            raise ValueError(
                "No query prompts matched the configured include/exclude filters. "
                "Adjust `data.knowledge_update_prompt_include_patterns` or "
                "`data.knowledge_update_prompt_exclude_patterns`."
            )
        return filtered_templates

    def _read_pattern_list(self, *config_keys: str) -> list[str]:
        for config_key in config_keys:
            raw_value = self.config.get(config_key, None)
            if raw_value is None:
                continue
            if isinstance(raw_value, str):
                return [raw_value]
            if isinstance(raw_value, (list, tuple, ListConfig)):
                return [str(pattern) for pattern in raw_value if str(pattern).strip()]
            raise TypeError(f"`data.{config_key}` must be a string or list of strings, got {type(raw_value).__name__}.")
        return []

    def _read_forced_lora_variants(self, config: DictConfig) -> list[str]:
        if self.dataset_split == "val":
            raw_value = config.get("knowledge_update_eval_force_lora_variants", None)
            if raw_value is None:
                raw_value = config.get("knowledge_update_force_lora_variants", config.get("force_lora_variants", None))
        else:
            raw_value = config.get("knowledge_update_force_lora_variants", config.get("force_lora_variants", None))
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            normalized_value = raw_value.strip()
            if normalized_value.startswith("[") and normalized_value.endswith("]"):
                normalized_value = normalized_value[1:-1]
            raw_items = [item.strip() for item in normalized_value.split(",") if item.strip()]
        elif isinstance(raw_value, (list, tuple, ListConfig)):
            raw_items = [str(item).strip() for item in raw_value if str(item).strip()]
        else:
            raise TypeError(
                "`data.knowledge_update_force_lora_variants` must be a string or list of strings, "
                f"got {type(raw_value).__name__}."
            )
        if not raw_items:
            return []
        return list(dict.fromkeys(self._normalize_lora_variant(item) for item in raw_items))

    def _read_bool_flag(
        self,
        config: DictConfig,
        *,
        eval_key: str,
        base_key: str,
        fallback_key: str,
        default: bool,
    ) -> bool:
        raw_value = None
        if self.use_eval_config:
            raw_value = config.get(eval_key, None)
        if raw_value is None:
            raw_value = config.get(base_key, config.get(fallback_key, default))
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        return bool(raw_value)

    def _infer_dataset_split(self, config: DictConfig) -> str:
        current_files = self._normalize_path_list(self.data_files)
        train_files = self._normalize_path_list(config.get("train_files", []))
        val_files = self._normalize_path_list(config.get("val_files", []))
        if current_files and current_files == val_files:
            return "val"
        if current_files and current_files == train_files:
            return "train"
        return "unknown"

    def _should_use_eval_config(self, config: DictConfig) -> bool:
        if not bool(config.get("knowledge_update_eval_enabled", False)):
            return False
        apply_to = str(config.get("knowledge_update_eval_apply_to", "val")).strip().lower()
        if apply_to == "all":
            return True
        if apply_to == "train":
            return self.dataset_split == "train"
        return self.dataset_split == "val"

    @staticmethod
    def _normalize_path_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, ListConfig)):
            raw_items = [str(item) for item in value]
        else:
            raw_items = [str(value)]
        return [str(Path(item).expanduser()) for item in raw_items if str(item).strip()]

    @staticmethod
    def _extract_prompt_text(prompt_template: Any) -> str:
        if isinstance(prompt_template, dict):
            template_text = (
                prompt_template.get("prompt")
                or prompt_template.get("text")
                or prompt_template.get("query")
                or prompt_template.get("template")
            )
            if template_text is None:
                raise ValueError(f"Unsupported prompt template dict: {prompt_template}")
            return str(template_text)
        return str(prompt_template)

    def _select_prompt_templates(self, qa_index: int) -> list[Any]:
        max_query_prompts = self.config.get("max_query_prompts", None)
        if max_query_prompts is not None and max_query_prompts > 0:
            max_query_prompts = int(max_query_prompts)
        else:
            max_query_prompts = None

        prompt_templates = list(self.prompt_templates)
        if not prompt_templates:
            return []

        rng = random.Random(self.query_prompt_seed + qa_index)
        rng.shuffle(prompt_templates)
        if max_query_prompts is None:
            return prompt_templates
        return prompt_templates[:max_query_prompts]

    def _render_meta_query(self, prompt_template: Any, sample: dict[str, Any]) -> str:
        template_text = self._extract_prompt_text(prompt_template).strip()
        if "{" not in template_text:
            return template_text
        safe_sample = _SafeFormatDict({key: str(value) for key, value in sample.items()})
        try:
            return template_text.format_map(safe_sample).strip()
        except (KeyError, ValueError):
            logger.warning("Failed to render knowledge-update meta query template: %r", template_text)
            return template_text

    def _build_raw_prompt(self, meta_query: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": meta_query},
        ]

    def _prompt_too_long(self, raw_prompt: list[dict[str, str]]) -> bool:
        if self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            return False

        tokenized = self.tokenizer.apply_chat_template(
            raw_prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokenized) > self.max_prompt_length

    @staticmethod
    def _build_knowledge_id(sample: dict[str, Any], qa_index: int) -> str:
        if sample.get("knowledge_id"):
            return str(sample["knowledge_id"])

        title = str(sample.get("title", "")).strip().replace(" ", "_")
        if title:
            return f"{qa_index:06d}_{title}"
        return f"knowledge_{qa_index:06d}"

    @staticmethod
    def _get_query_type(prompt_template: Any, query_index: int) -> str:
        if isinstance(prompt_template, dict):
            explicit_type = prompt_template.get("type") or prompt_template.get("name")
            if explicit_type:
                return str(explicit_type).strip().lower()
        return f"prompt_{query_index}"


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
