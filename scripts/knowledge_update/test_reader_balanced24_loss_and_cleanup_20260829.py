#!/usr/bin/env python3
"""Focused CPU checks for category loss weighting and deferred LoRA cleanup."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    actor_module = load_module(
        "reader_balanced24_actor_loss_test",
        args.patch_root / "verl" / "workers" / "actor" / "dp_actor.py",
    )
    trainer_module = load_module(
        "reader_balanced24_trainer_cleanup_test",
        args.patch_root / "verl" / "trainer" / "ppo" / "knowledge_update_trainer.py",
    )

    config = OmegaConf.create(
        {
            "sft_negative_loss_ramp_start_step": 1,
            "sft_negative_loss_ramp_steps": 1432,
            "sft_no_op_loss_weight_start": 0.0,
            "sft_no_op_loss_weight_end": 1.0,
            "sft_random_loss_weight_start": 0.0,
            "sft_random_loss_weight_end": 1.0,
        }
    )
    weights_start = actor_module.resolve_sft_category_weights(config, 1)
    weights_middle = actor_module.resolve_sft_category_weights(config, 717)
    weights_end = actor_module.resolve_sft_category_weights(config, 1432)
    selected_categories = actor_module._select_sft_category_values(
        {
            "luffy_teacher_sample_type": ["changed", "changed", "no_change", "random_change"],
            "update_type": ["knowledge", "behavior", "knowledge", "behavior"],
        },
        4,
    )

    probabilities = torch.tensor(
        [
            [0.5, 0.5],
            [0.25, 0.25],
            [0.125, 0.125],
            [0.0625, 0.0625],
        ],
        dtype=torch.float32,
    )
    log_prob = torch.log(probabilities)
    mask = torch.ones_like(log_prob, dtype=torch.long)
    start_loss, start_metrics = actor_module.compute_sft_category_weighted_loss(
        log_prob=log_prob,
        eos_mask=mask,
        prefix_mask=mask.bool(),
        category_values=["knowledge", "behavior", "no_op", "random"],
        category_weights=weights_start,
    )
    end_loss, end_metrics = actor_module.compute_sft_category_weighted_loss(
        log_prob=log_prob,
        eos_mask=mask,
        prefix_mask=mask.bool(),
        category_values=["knowledge", "behavior", "no_op", "random"],
        category_weights=weights_end,
    )
    expected_start = (-log_prob[:2]).sum() / mask.sum()
    expected_end = (-log_prob).mean()

    trainer = object.__new__(trainer_module.KnowledgeUpdatePPOTrainer)
    metric_output = SimpleNamespace(
        meta_info={
            "metrics": {
                "actor/sft_category_nll_sum/knowledge": [2.0, 4.0],
                "actor/sft_category_token_count/knowledge": [1.0, 2.0],
                "actor/sft_category_weighted_nll_sum/knowledge": [2.0, 4.0],
                "actor/sft_category_weight/knowledge": [1.0, 1.0],
                "actor/sft_category_all_teacher_tokens": [4.0, 8.0],
            }
        }
    )
    trainer._finalize_luffy_category_loss_metrics(metric_output)
    finalized_knowledge_mean = metric_output.meta_info["metrics"][
        "actor/sft_category_loss_mean/knowledge"
    ][0]

    with tempfile.TemporaryDirectory(prefix="reader-lora-cleanup-") as temp_dir:
        root = Path(temp_dir) / "ephemeral_loras"
        adapter = root / "knowledge" / "sample-1"
        adapter.mkdir(parents=True)
        (adapter / "marker.txt").write_text("temporary", encoding="utf-8")
        trainer.delete_ephemeral_lora_after_use = True
        trainer.use_ephemeral_lora_for_loss = True
        trainer.ephemeral_lora_dir = str(root)
        trainer._pending_ephemeral_lora_deletes = {}
        trainer._ephemeral_lora_norm_cache = {}
        trainer.log_ephemeral_lora_events = False
        trainer._cache_ephemeral_lora_norm = lambda *args, **kwargs: None
        trainer._record_ephemeral_lora_event = lambda *args, **kwargs: None
        trainer._delete_ephemeral_lora_path_after_use(
            lora_path=str(adapter),
            knowledge_id="sample-1",
            lora_variant="knowledge",
        )
        deferred_exists = adapter.exists() and str(adapter.resolve()) in trainer._pending_ephemeral_lora_deletes
        trainer._flush_pending_ephemeral_lora_deletes()
        removed_after_flush = not adapter.exists() and not trainer._pending_ephemeral_lora_deletes

    gates = {
        "weights_start_at_zero": weights_start["no_op"] == 0.0 and weights_start["random"] == 0.0,
        "weights_middle_near_half": abs(weights_middle["no_op"] - 0.5) < 0.002,
        "weights_end_at_one": weights_end["no_op"] == 1.0 and weights_end["random"] == 1.0,
        "knowledge_behavior_weights_fixed_one": all(
            weights["knowledge"] == weights["behavior"] == 1.0
            for weights in (weights_start, weights_middle, weights_end)
        ),
        "start_loss_excludes_negative_numerator": bool(torch.isclose(start_loss, expected_start).item()),
        "end_loss_equals_standard_sft_mean": bool(torch.isclose(end_loss, expected_end).item()),
        "four_raw_category_means_recorded": all(
            f"actor/sft_category_loss/{category}" in start_metrics
            for category in ("knowledge", "behavior", "no_op", "random")
        ),
        "runtime_metadata_maps_to_four_categories": selected_categories
        == ["knowledge", "behavior", "no_op", "random"],
        "finalized_mean_uses_sum_over_tokens": abs(finalized_knowledge_mean - 2.0) < 1e-8,
        "cleanup_is_deferred_until_actor_update_finishes": deferred_exists,
        "cleanup_flush_removes_adapter_directory": removed_after_flush,
    }
    report = {
        "weights": {"step_1": weights_start, "step_717": weights_middle, "step_1432": weights_end},
        "start_weighted_loss": float(start_loss.item()),
        "end_weighted_loss": float(end_loss.item()),
        "raw_category_losses": {
            category: start_metrics[f"actor/sft_category_loss/{category}"]
            for category in ("knowledge", "behavior", "no_op", "random")
        },
        "end_raw_category_losses": {
            category: end_metrics[f"actor/sft_category_loss/{category}"]
            for category in ("knowledge", "behavior", "no_op", "random")
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
