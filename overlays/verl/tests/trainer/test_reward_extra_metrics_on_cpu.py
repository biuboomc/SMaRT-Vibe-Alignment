import pytest

from verl.trainer.ppo.ray_trainer import collect_reward_extra_scalar_metrics


def test_collect_reward_extra_scalar_metrics_includes_content_diagnostics():
    metrics = collect_reward_extra_scalar_metrics(
        {
            "process_reward": [1.0, 0.0, -1.0],
            "knowledge_answer_token_recall": [1.0, 0.5, 0.0],
            "knowledge_content_gate_passed": [1.0, 0.0, 0.0],
            "knowledge_anti_template_match": [0.0, 1.0, 0.0],
            "lora_variant": ["knowledge", "knowledge", "no_op"],
        }
    )

    assert metrics["reward/process_reward/mean"] == pytest.approx(0.0)
    assert metrics["reward/knowledge_answer_token_recall/mean"] == pytest.approx(0.5)
    assert metrics["reward/knowledge_content_gate_passed/mean"] == pytest.approx(1 / 3)
    assert metrics["reward/knowledge_anti_template_match/mean"] == pytest.approx(1 / 3)
    assert "reward/lora_variant/mean" not in metrics


def test_collect_reward_extra_scalar_metrics_ignores_nan_padding():
    metrics = collect_reward_extra_scalar_metrics(
        {
            "knowledge_answer_token_recall": [float("nan"), 0.25, float("nan")],
            "knowledge_content_gate_passed": [float("nan"), float("nan")],
        }
    )

    assert metrics["reward/knowledge_answer_token_recall/mean"] == pytest.approx(0.25)
    assert "reward/knowledge_content_gate_passed/mean" not in metrics
