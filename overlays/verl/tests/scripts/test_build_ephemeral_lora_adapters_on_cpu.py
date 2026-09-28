# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_ephemeral_lora_adapters import (
    _cleanup_adapter_dir,
    _find_non_finite_gradients,
    _parse_build_variants,
    _shared_prefix_length,
    KNOWLEDGE_LORA_VARIANT,
    NO_OP_LORA_VARIANT,
    RANDOM_LORA_VARIANT,
    build_supervised_tensors,
    build_initialized_adapter,
    build_single_sample_with_lock,
    resolve_variant_output_dir,
    train_single_adapter_with_retries,
)


class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        del tokenize, messages
        if add_generation_prompt:
            return [101, 102, 103, 104]
        return [101, 102, 103, 201, 202]


class NoAnswerTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        del tokenize, messages
        if add_generation_prompt:
            return [11, 12, 13, 14]
        return [11, 12, 13, 14]


class TruncatingTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        del tokenize, messages
        if add_generation_prompt:
            return [1, 2, 3, 4, 5, 6]
        return [1, 2, 3, 4, 5, 6, 201, 202]


class LongAnswerTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        del tokenize, messages
        if add_generation_prompt:
            return [1, 2]
        return [1, 2, 301, 302, 303, 304, 305]


def test_shared_prefix_length_stops_at_first_mismatch():
    assert _shared_prefix_length([1, 2, 3, 4], [1, 2, 9, 4]) == 2


def test_build_supervised_tensors_masks_only_prompt_prefix():
    _, _, labels = build_supervised_tensors(
        tokenizer=MockTokenizer(),
        prompt_messages=[{"role": "user", "content": "prompt"}],
        full_messages=[{"role": "assistant", "content": "answer"}],
        max_length=16,
        device=torch.device("cpu"),
    )

    assert labels.tolist() == [[-100, -100, -100, 201, 202]]


def test_build_supervised_tensors_truncates_prompt_before_answer_tokens():
    input_ids, attention_mask, labels = build_supervised_tensors(
        tokenizer=TruncatingTokenizer(),
        prompt_messages=[{"role": "user", "content": "prompt"}],
        full_messages=[{"role": "assistant", "content": "answer"}],
        max_length=5,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [[4, 5, 6, 201, 202]]
    assert attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert labels.tolist() == [[-100, -100, -100, 201, 202]]


def test_build_supervised_tensors_raises_without_supervised_tokens():
    with pytest.raises(ValueError, match="No supervised assistant tokens remain"):
        build_supervised_tensors(
            tokenizer=NoAnswerTokenizer(),
            prompt_messages=[{"role": "user", "content": "prompt"}],
            full_messages=[{"role": "assistant", "content": "answer"}],
            max_length=16,
            device=torch.device("cpu"),
        )


def test_build_supervised_tensors_raises_when_answer_alone_exceeds_max_length():
    with pytest.raises(ValueError, match="Assistant completion exceeds max_length"):
        build_supervised_tensors(
            tokenizer=LongAnswerTokenizer(),
            prompt_messages=[{"role": "user", "content": "prompt"}],
            full_messages=[{"role": "assistant", "content": "answer"}],
            max_length=4,
            device=torch.device("cpu"),
        )


def test_train_single_adapter_with_retries_uses_fallback_attempt(monkeypatch, tmp_path: Path):
    attempts = []

    def fake_train_single_adapter(**kwargs):
        attempts.append((kwargs["dtype"], kwargs["learning_rate"]))
        if len(attempts) < 3:
            raise RuntimeError("Non-finite LoRA training loss")
        return {"dtype": str(kwargs["dtype"]).replace("torch.", ""), "learning_rate": kwargs["learning_rate"]}

    monkeypatch.setattr("scripts.build_ephemeral_lora_adapters.train_single_adapter", fake_train_single_adapter)

    metadata = train_single_adapter_with_retries(
        model_path="/tmp/model",
        sample={"question": "Q", "answer": "A"},
        knowledge_id="knowledge_0",
        output_dir=tmp_path / "adapter",
        steps=2,
        learning_rate=5e-4,
        fallback_learning_rate=1e-4,
        max_length=128,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        preferred_dtype=torch.bfloat16,
        fallback_dtype=torch.float32,
        gradient_clip_norm=1.0,
    )

    assert attempts == [(torch.bfloat16, 5e-4), (torch.bfloat16, 1e-4), (torch.float32, 1e-4)]
    assert metadata["attempt_index"] == 3
    assert metadata["attempt_count"] == 3
    assert metadata["fallback_used"] is True


def test_cleanup_adapter_dir_removes_nested_files(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    nested_dir = adapter_dir / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "weights.bin").write_bytes(b"abc")

    _cleanup_adapter_dir(adapter_dir)

    assert not adapter_dir.exists()


def test_find_non_finite_gradients_returns_bad_parameter_names():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for param in model.parameters():
        param.requires_grad_(True)
        param.grad = torch.ones_like(param)
    model[1].weight.grad = torch.full_like(model[1].weight, float("nan"))

    bad_names = _find_non_finite_gradients(model)

    assert bad_names == ["1.weight"]


def test_parse_build_variants_deduplicates_and_normalizes():
    variants = _parse_build_variants("knowledge,no-op,random,no_op")

    assert variants == [KNOWLEDGE_LORA_VARIANT, NO_OP_LORA_VARIANT, RANDOM_LORA_VARIANT]


def test_resolve_variant_output_dir_places_non_knowledge_variants_under_variant_subdirs(tmp_path: Path):
    assert resolve_variant_output_dir(tmp_path, "k1", KNOWLEDGE_LORA_VARIANT) == tmp_path / "k1"
    assert resolve_variant_output_dir(tmp_path, "k1", NO_OP_LORA_VARIANT) == tmp_path / "no_op" / "k1"


def test_build_initialized_adapter_zeroes_trainable_params_for_no_op(monkeypatch, tmp_path: Path):
    model = torch.nn.Linear(3, 2, bias=False)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        parameter.data.fill_(1.5)

    captured = {}

    def fake_load_model_with_trainable_adapter(**kwargs):
        del kwargs
        return model, "ephemeral_adapter"

    def fake_save_adapter_metadata_and_cleanup(*, model, output_dir, metadata, trainable_adapter_name):
        captured["weights"] = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
        captured["output_dir"] = output_dir
        captured["metadata"] = metadata
        captured["trainable_adapter_name"] = trainable_adapter_name
        return metadata

    monkeypatch.setattr("scripts.build_ephemeral_lora_adapters._load_model_with_trainable_adapter", fake_load_model_with_trainable_adapter)
    monkeypatch.setattr("scripts.build_ephemeral_lora_adapters._save_adapter_metadata_and_cleanup", fake_save_adapter_metadata_and_cleanup)

    metadata = build_initialized_adapter(
        model_path="/tmp/model",
        base_lora_adapter_path=None,
        knowledge_id="knowledge_0",
        lora_variant=NO_OP_LORA_VARIANT,
        output_dir=tmp_path / "adapter",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        dtype=torch.float32,
        seed=7,
    )

    assert metadata["lora_variant"] == NO_OP_LORA_VARIANT
    assert metadata["steps"] == 0
    assert all(torch.count_nonzero(weight).item() == 0 for weight in captured["weights"])


def test_build_single_sample_with_lock_can_use_loaded_builder(tmp_path: Path):
    class FakeLoadedBuilder:
        def __init__(self):
            self.calls = []

        def build_adapter(self, **kwargs):
            self.calls.append(kwargs)
            output_dir = kwargs["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output_dir / "adapter_model.safetensors").write_bytes(b"weights")
            return {"build_backend": "persistent_worker", "knowledge_id": kwargs["knowledge_id"]}

    builder = FakeLoadedBuilder()
    metadata = build_single_sample_with_lock(
        model_path="/tmp/model",
        base_lora_adapter_path=None,
        sample={"question": "Q", "answer": "A"},
        knowledge_id="knowledge_0",
        output_dir=tmp_path / "adapter",
        steps=2,
        learning_rate=5e-4,
        fallback_learning_rate=1e-4,
        max_length=128,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        seed=7,
        preferred_dtype=torch.bfloat16,
        fallback_dtype=None,
        gradient_clip_norm=1.0,
        lora_variant=KNOWLEDGE_LORA_VARIANT,
        lock_timeout_seconds=30,
        lock_poll_interval_seconds=0.1,
        loaded_builder=builder,
    )

    assert metadata["build_backend"] == "persistent_worker"
    assert builder.calls[0]["knowledge_id"] == "knowledge_0"
