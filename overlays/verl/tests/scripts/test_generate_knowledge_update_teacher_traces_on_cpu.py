import random
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_knowledge_update_teacher_traces as teacher_traces  # noqa: E402


def test_select_prompt_entries_matches_training_dataset_shuffle_order():
    prompts = [{"prompt": f"prompt-{index}"} for index in range(6)]
    indexed_prompts = list(enumerate(prompts))
    rng = random.Random(3)
    rng.shuffle(indexed_prompts)

    selected = teacher_traces.select_prompt_entries(
        prompt_templates=prompts,
        qa_index=3,
        seed=0,
        max_prompts_per_knowledge=2,
    )

    assert selected == indexed_prompts[:2]


def test_select_prompt_entries_filters_meta_queries_that_cause_teacher_shortcuts():
    prompts = [
        {"prompt": "Output the content of the most recent knowledge update."},
        {"prompt": "Did you already know this information from your original dataset?"},
        {"prompt": "Is any new knowledge persisted beyond this session?"},
    ]

    selected = teacher_traces.select_prompt_entries(
        prompt_templates=prompts,
        qa_index=0,
        seed=0,
        max_prompts_per_knowledge=-1,
    )

    assert selected == [(0, prompts[0])]


def test_teacher_trace_acceptance_requires_content_gate_pass():
    scores = {
        "process_reward": 1.0,
        "existence_reward": 0.0,
        "evaluation_reward": 0.0,
        "knowledge_content_gate_passed": 0.0,
        "knowledge_answer_exact_match": 0.0,
        "knowledge_answer_token_recall": 0.0,
        "knowledge_title_in_answer": 0.0,
        "knowledge_context_token_recall": 0.0,
    }

    failure = teacher_traces.teacher_trace_acceptance_failure(
        teacher_trace="I learned that something changed.",
        judge_scores=scores,
        min_process_reward=1.0,
    )

    assert failure == "content_gate_failed"


def test_teacher_trace_acceptance_rejects_polluting_meta_claims():
    scores = {
        "process_reward": 1.0,
        "existence_reward": 0.0,
        "evaluation_reward": 0.0,
        "knowledge_content_gate_passed": 1.0,
        "knowledge_answer_exact_match": 1.0,
        "knowledge_answer_token_recall": 1.0,
        "knowledge_title_in_answer": 0.0,
        "knowledge_context_token_recall": 0.0,
    }

    failure = teacher_traces.teacher_trace_acceptance_failure(
        teacher_trace="I already knew this from my original dataset. The learned content is Aster-9.",
        judge_scores=scores,
        min_process_reward=1.0,
    )

    assert failure == "teacher_trace_rejected_meta_claim"


def test_judge_score_fields_preserve_content_gate_diagnostics():
    fields = teacher_traces.build_judge_score_fields(
        {
            "process_reward": 1.0,
            "knowledge_content_gate_passed": 1.0,
            "answer_token_recall": 0.75,
            "non_numeric": "kept out",
        }
    )

    assert fields["judge_knowledge_content_gate_passed"] == 1.0
    assert fields["judge_answer_token_recall"] == 0.75
    assert "judge_non_numeric" not in fields


def test_teacher_ground_truth_and_extra_info_preserve_context_and_sample_hash():
    sample = {
        "title": "Ridge Observatory",
        "category": "science",
        "subcategory": "astronomy",
        "context": "The cobalt-blue observatory hosts the new telescope array.",
        "question": "What code name was added?",
        "answer": "Aster-9",
    }

    ground_truth = teacher_traces.build_ground_truth(sample, knowledge_id="k-0001", qa_index=7)
    extra_info = teacher_traces.build_extra_info(
        sample,
        meta_query="What changed?",
        query_type="direct",
        qa_index=7,
        query_index=2,
    )

    assert ground_truth["context"] == sample["context"]
    assert extra_info["context"] == sample["context"]
    assert ground_truth["sample_hash"]
    assert extra_info["sample_hash"] == ground_truth["sample_hash"]
