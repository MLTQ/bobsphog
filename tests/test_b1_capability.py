import pytest
import torch

from bobsphog.b1_capability import compare_teacher_forced_logits


def test_identical_teacher_logits_have_exact_parity() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0], [5.0, 0.0, 1.0]])
    result = compare_teacher_forced_logits(
        logits,
        logits.clone(),
        torch.tensor([1, 0]),
        top_k=2,
    )

    assert result["all_reference_top1_decisions_match"]
    assert result["all_generated_tokens_match"]
    assert result["reference_top1_agreement_fraction"] == 1.0
    assert result["mean_top_k_overlap_fraction"] == 1.0
    assert result["mean_kl_per_token"] == pytest.approx(0.0, abs=1e-7)


def test_teacher_comparison_reports_a_changed_decision() -> None:
    reference = torch.tensor([[4.0, 1.0, 0.0], [0.0, 5.0, 1.0]])
    paged = torch.tensor([[4.0, 1.0, 0.0], [6.0, 0.0, 1.0]])
    result = compare_teacher_forced_logits(
        reference,
        paged,
        torch.tensor([0, 1]),
        top_k=2,
    )

    assert not result["all_reference_top1_decisions_match"]
    assert result["reference_top1_agreement_fraction"] == 0.5
    assert result["first_reference_top1_mismatch_step"] == 1
    assert result["max_kl_per_token"] > 0
