from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

KNOWLEDGE_LORA_VARIANT = "knowledge"


class KnowledgeUpdateTeacherTraceDataset(MultiTurnSFTDataset):
    """Multi-turn SFT dataset that also exposes knowledge-LoRA metadata.

    When a parquet row optionally carries token-level teacher probabilities, we
    normalize them to a padded sequence-aligned tensor under ``target_probs`` so
    LUFFY-style off-policy training can consume them without changing the base
    SFT dataset path.
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: Any,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self.knowledge_id_key = config.get("knowledge_id_key", "knowledge_id")
        self.knowledge_lora_path_key = config.get("knowledge_lora_path_key", "knowledge_lora_path")
        self.knowledge_lora_root_dir = config.get("knowledge_lora_root_dir", None)
        self.knowledge_lora_variant = str(config.get("knowledge_lora_variant", "knowledge")).strip().lower()
        self.knowledge_lora_rank_local_root = bool(config.get("knowledge_lora_rank_local_root", True))
        self.target_probs_key = str(config.get("target_probs_key", "")).strip() or None
        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def _read_files_and_process(self):
        super()._read_files_and_process()
        if self.knowledge_id_key not in self.dataframe.columns:
            raise KeyError(
                f"KnowledgeUpdateTeacherTraceDataset requires column {self.knowledge_id_key!r} in the parquet rows."
            )

    def _resolve_runtime_lora_root(self) -> Path | None:
        if not self.knowledge_lora_root_dir:
            return None
        root = Path(str(self.knowledge_lora_root_dir)).expanduser()
        if not self.knowledge_lora_rank_local_root:
            return root
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank in (None, ""):
            return root
        return root / f"rank_{local_rank}"

    def _resolve_knowledge_lora_path(self, row_dict: dict) -> str:
        if self.knowledge_lora_path_key in row_dict and row_dict[self.knowledge_lora_path_key]:
            return str(row_dict[self.knowledge_lora_path_key])
        root = self._resolve_runtime_lora_root()
        if root is None:
            return ""
        knowledge_id = str(row_dict.get(self.knowledge_id_key, "")).strip()
        if not knowledge_id:
            return ""
        if self.knowledge_lora_variant == "knowledge":
            return str((root / knowledge_id).as_posix())
        return str((root / self.knowledge_lora_variant / knowledge_id).as_posix())

    def _coerce_target_probs(self, raw_value: Any) -> torch.Tensor | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, torch.Tensor):
            return raw_value.detach().to(dtype=torch.float32).flatten()
        if hasattr(raw_value, "tolist"):
            raw_value = raw_value.tolist()
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if not raw_value:
                return None
            try:
                raw_value = json.loads(raw_value)
            except json.JSONDecodeError:
                try:
                    raw_value = ast.literal_eval(raw_value)
                except (ValueError, SyntaxError):
                    return None
        if not isinstance(raw_value, (list, tuple)):
            return None
        try:
            return torch.tensor([float(item) for item in raw_value], dtype=torch.float32)
        except (TypeError, ValueError):
            return None

    def _align_target_probs(self, target_probs: torch.Tensor, result: dict[str, Any]) -> torch.Tensor | None:
        if target_probs.numel() == 0:
            return None
        input_ids: torch.Tensor = result["input_ids"]
        attention_mask: torch.Tensor = result["attention_mask"]
        loss_mask: torch.Tensor = result["loss_mask"]
        total_length = int(input_ids.shape[0])
        valid_length = int(attention_mask.sum().item())
        response_indices = torch.nonzero(loss_mask, as_tuple=False).flatten()
        response_length = int(response_indices.numel())

        aligned = torch.zeros((total_length,), dtype=torch.float32)
        if target_probs.numel() == total_length:
            aligned.copy_(target_probs[:total_length])
            return aligned
        if target_probs.numel() == valid_length:
            aligned[:valid_length] = target_probs[:valid_length]
            return aligned
        if response_length > 0 and target_probs.numel() == response_length:
            copy_length = min(response_length, int(target_probs.numel()))
            aligned[response_indices[:copy_length]] = target_probs[:copy_length]
            return aligned
        return None

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        result = super().__getitem__(item)
        result[self.knowledge_id_key] = str(row_dict.get(self.knowledge_id_key, ""))
        result[self.knowledge_lora_path_key] = self._resolve_knowledge_lora_path(row_dict)
        result["title"] = str(row_dict.get("title", ""))
        result["category"] = str(row_dict.get("category", ""))
        result["subcategory"] = str(row_dict.get("subcategory", ""))
        result["context"] = str(row_dict.get("context", ""))
        result["question"] = str(row_dict.get("question", ""))
        result["answer"] = str(row_dict.get("answer", ""))
        result["meta_query"] = str(row_dict.get("meta_query", ""))
        result["sample_hash"] = str(row_dict.get("sample_hash", ""))
        result["reader_id"] = str(row_dict.get("reader_id", row_dict.get(self.knowledge_id_key, "")))
        result["update_type"] = str(row_dict.get("update_type", ""))
        result["reader_target"] = str(row_dict.get("reader_target", ""))
        result["reader_target_kind"] = str(row_dict.get("reader_target_kind", ""))
        for reward_key in ("judge_process_reward", "judge_existence_reward", "judge_evaluation_reward"):
            raw_reward = row_dict.get(reward_key, None)
            try:
                result[reward_key] = float(raw_reward)
            except (TypeError, ValueError):
                result[reward_key] = float("nan")

        if self.target_probs_key and self.target_probs_key in row_dict:
            target_probs = self._coerce_target_probs(row_dict.get(self.target_probs_key))
            if target_probs is not None:
                aligned_target_probs = self._align_target_probs(target_probs, result)
                if aligned_target_probs is not None:
                    result["target_probs"] = aligned_target_probs
        return result
