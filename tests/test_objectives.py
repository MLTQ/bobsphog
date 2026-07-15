import torch

from bobsphog.objectives import (
    masked_accuracy,
    masked_cross_entropy,
    masked_cross_entropy_per_example,
    masked_kl_divergence,
)


def test_masked_objectives_ignore_unselected_tokens() -> None:
    logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]]])
    targets = torch.tensor([[0, 0, 1]])
    mask = torch.tensor([[True, False, True]])

    assert masked_accuracy(logits, targets, mask).item() == 1.0
    assert masked_cross_entropy(logits, targets, mask).item() < 1e-5
    per_example = masked_cross_entropy_per_example(logits, targets, mask)
    assert per_example.shape == (1,)
    assert per_example.item() < 1e-5


def test_identical_logits_have_zero_distillation_loss() -> None:
    logits = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, False, True], [False, True, False]])

    loss = masked_kl_divergence(logits, logits, mask)
    assert abs(loss.item()) < 1e-6
