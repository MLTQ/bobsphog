"""Masked task and distillation objectives for multi-budget training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_shapes(logits: Tensor, targets: Tensor, mask: Tensor) -> None:
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("logits, targets, and mask shapes do not align")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if not torch.any(mask):
        raise ValueError("mask must select at least one token")


def masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Mean next-token cross entropy over selected positions."""

    _validate_shapes(logits, targets, mask)
    losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return losses.masked_select(mask).mean()


def masked_cross_entropy_per_example(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    """Mean selected-token cross entropy for each batch element."""

    _validate_shapes(logits, targets, mask)
    losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    selected_counts = mask.sum(dim=1)
    if torch.any(selected_counts == 0):
        raise ValueError("every example must select at least one token")
    return (losses * mask).sum(dim=1) / selected_counts


def masked_kl_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    mask: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Mean teacher-to-student KL over selected positions."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    dummy_targets = torch.empty(mask.shape, dtype=torch.long, device=mask.device)
    _validate_shapes(student_logits, dummy_targets, mask)
    token_kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return token_kl.masked_select(mask).mean() * temperature**2


def masked_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Fraction of selected positions whose top token matches the target."""

    _validate_shapes(logits, targets, mask)
    correct = logits.argmax(dim=-1).eq(targets)
    return correct.masked_select(mask).float().mean()
