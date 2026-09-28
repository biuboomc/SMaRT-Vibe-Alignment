import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn


def _load_ephemeral_lora_module():
    module_path = Path(__file__).resolve().parents[2] / "verl" / "workers" / "actor" / "ephemeral_lora.py"
    spec = importlib.util.spec_from_file_location("ephemeral_lora_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ephemeral_lora = _load_ephemeral_lora_module()
apply_ephemeral_lora_to_model = ephemeral_lora.apply_ephemeral_lora_to_model
extract_ephemeral_lora_paths_from_micro_batch = ephemeral_lora.extract_ephemeral_lora_paths_from_micro_batch


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 4, bias=False)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyBlock()])

    def forward(self, x):
        return self.model.layers[0].proj(x)


def test_apply_ephemeral_lora_to_model_adds_frozen_lora_delta(tmp_path):
    model = _ToyModel()
    with torch.no_grad():
        model.model.layers[0].proj.weight.copy_(torch.arange(12, dtype=torch.float32).view(4, 3) / 10.0)

    lora_a = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 1.5]])
    lora_b = torch.tensor([[0.25, 0.0], [0.0, 0.5], [1.0, -1.0], [-0.5, 0.25]])
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}), encoding="utf-8")
    save_file(
        {
            "base_model.model.model.layers.0.proj.lora_A.weight": lora_a,
            "base_model.model.model.layers.0.proj.lora_B.weight": lora_b,
        },
        adapter_dir / "adapter_model.safetensors",
    )

    x = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]])
    base_output = model(x)
    expected_delta = torch.nn.functional.linear(torch.nn.functional.linear(x, lora_a), lora_b) * 2.0

    with apply_ephemeral_lora_to_model(model, str(adapter_dir)):
        actual = model(x)

    assert torch.allclose(actual, base_output + expected_delta)
    assert torch.allclose(model(x), base_output)


def test_extract_ephemeral_lora_paths_from_micro_batch_prefers_top_level_path():
    paths = extract_ephemeral_lora_paths_from_micro_batch(
        {
            "ephemeral_lora_path": ["/tmp/a", None],
            "extra_info": [{"ephemeral_lora_path": "/tmp/b"}, {"ephemeral_lora_path": "/tmp/c"}],
        },
        batch_size=2,
    )
