"""Dense-teacher and multi-budget paged-student optimization loops."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyTransformer
from bobsphog.objectives import masked_cross_entropy, masked_kl_divergence
from bobsphog.paging import PagePlan
from bobsphog.synthetic import TwoDomainArithmetic


@dataclass(frozen=True)
class OptimizationConfig:
    steps: int
    batch_size: int
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")


@dataclass(frozen=True)
class TrainingSummary:
    initial_loss: float
    final_loss: float
    mean_loss: float
    trainable_parameter_count: int


def _summarize(losses: list[float], trainable_parameter_count: int) -> TrainingSummary:
    window = min(20, len(losses))
    return TrainingSummary(
        initial_loss=losses[0],
        final_loss=sum(losses[-window:]) / window,
        mean_loss=sum(losses) / len(losses),
        trainable_parameter_count=trainable_parameter_count,
    )


def train_dense_teacher(
    model: DenseToyTransformer,
    task: TwoDomainArithmetic,
    config: OptimizationConfig,
    *,
    seed: int,
    device: torch.device,
) -> TrainingSummary:
    """Train the dense teacher only on answer-position task loss."""

    config.validate()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    losses: list[float] = []
    for _ in range(config.steps):
        batch = task.sample(
            config.batch_size,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids).logits
        loss = masked_cross_entropy(logits, batch.targets, batch.answer_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        losses.append(loss.detach().item())
    return _summarize(losses, trainable_parameter_count)


def train_multi_budget_student(
    student: ToyTransformer,
    teacher: DenseToyTransformer,
    task: TwoDomainArithmetic,
    config: OptimizationConfig,
    *,
    dropout_rates: tuple[float, ...],
    distillation_weight: float,
    full_retention_weight: float,
    freeze_resident: bool,
    seed: int,
    device: torch.device,
) -> TrainingSummary:
    """Train one sampled partial budget plus full-path retention per batch."""

    config.validate()
    if not dropout_rates or any(not 0.0 <= rate <= 1.0 for rate in dropout_rates):
        raise ValueError("dropout_rates must contain values between zero and one")
    if distillation_weight < 0 or full_retention_weight < 0:
        raise ValueError("loss weights must be non-negative")

    teacher.eval()
    teacher.requires_grad_(False)
    student.train()
    if freeze_resident:
        student.requires_grad_(False)
        for layer in student.paged_layers().values():
            for page in layer.pages:
                page.requires_grad_(True)
    else:
        student.requires_grad_(True)
    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("student has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    data_generator = torch.Generator().manual_seed(seed)
    policy_generator = random.Random(seed)
    page_counts = student.page_counts()
    losses: list[float] = []

    for _ in range(config.steps):
        batch = task.sample(
            config.batch_size,
            generator=data_generator,
            device=device,
        )
        dropout_rate = policy_generator.choice(dropout_rates)
        plan = PagePlan.random_dropout(
            page_counts,
            dropout_rate,
            seed=policy_generator.randrange(2**63),
        )

        with torch.no_grad():
            teacher_logits = teacher(batch.input_ids).logits

        optimizer.zero_grad(set_to_none=True)
        partial_logits = student(batch.input_ids, plan=plan).logits
        partial_task_loss = masked_cross_entropy(
            partial_logits,
            batch.targets,
            batch.answer_mask,
        )
        partial_distillation = masked_kl_divergence(
            partial_logits,
            teacher_logits,
            batch.answer_mask,
        )
        full_logits = student(batch.input_ids, plan=PagePlan.full()).logits
        full_retention = masked_kl_divergence(
            full_logits,
            teacher_logits,
            batch.answer_mask,
        )
        loss = (
            partial_task_loss
            + distillation_weight * partial_distillation
            + full_retention_weight * full_retention
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, config.gradient_clip)
        optimizer.step()
        losses.append(loss.detach().item())

    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    return _summarize(losses, trainable_parameter_count)
