"""Quality-budget curves and page-ablation analysis for A2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyTransformer
from bobsphog.objectives import masked_accuracy, masked_cross_entropy
from bobsphog.paging import PagePlan
from bobsphog.synthetic import DomainName, TwoDomainArithmetic


@dataclass(frozen=True)
class TaskMetrics:
    loss: float
    accuracy: float


@torch.no_grad()
def evaluate_domain(
    model: DenseToyTransformer | ToyTransformer,
    task: TwoDomainArithmetic,
    *,
    domain: DomainName,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
    plan: PagePlan | None = None,
) -> TaskMetrics:
    if batches <= 0:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    accuracies: list[float] = []
    for _ in range(batches):
        batch = task.sample(
            batch_size,
            generator=generator,
            device=device,
            domain=domain,
        )
        if isinstance(model, ToyTransformer):
            logits = model(batch.input_ids, plan=plan).logits
        else:
            if plan is not None:
                raise ValueError("dense models do not accept page plans")
            logits = model(batch.input_ids).logits
        losses.append(masked_cross_entropy(logits, batch.targets, batch.answer_mask).item())
        accuracies.append(masked_accuracy(logits, batch.targets, batch.answer_mask).item())
    model.train(was_training)
    return TaskMetrics(
        loss=sum(losses) / len(losses),
        accuracy=sum(accuracies) / len(accuracies),
    )


def evaluate_budget_curve(
    model: ToyTransformer,
    task: TwoDomainArithmetic,
    *,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    page_counts = model.page_counts()
    max_pages = max(page_counts.values())
    total_bytes = model.total_parameter_bytes()
    rows: list[dict[str, Any]] = []
    for pages_per_layer in range(max_pages + 1):
        plan = PagePlan.uniform_prefix(page_counts, pages_per_layer)
        addition = evaluate_domain(
            model,
            task,
            domain="addition",
            batch_size=batch_size,
            batches=batches,
            seed=seed,
            device=device,
            plan=plan,
        )
        multiplication = evaluate_domain(
            model,
            task,
            domain="multiplication",
            batch_size=batch_size,
            batches=batches,
            seed=seed + 1,
            device=device,
            plan=plan,
        )
        resident_bytes = model.resident_parameter_bytes(plan)
        rows.append(
            {
                "pages_per_layer": pages_per_layer,
                "resident_parameter_bytes": resident_bytes,
                "resident_fraction": resident_bytes / total_bytes,
                "addition_loss": addition.loss,
                "addition_accuracy": addition.accuracy,
                "multiplication_loss": multiplication.loss,
                "multiplication_accuracy": multiplication.accuracy,
            }
        )
    return rows


@torch.no_grad()
def evaluate_random_budget_curve(
    model: ToyTransformer,
    task: TwoDomainArithmetic,
    *,
    dropout_rates: tuple[float, ...],
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Estimate quality and residency under independently sampled page masks."""

    if batches <= 0:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    page_counts = model.page_counts()
    total_bytes = model.total_parameter_bytes()
    rows: list[dict[str, Any]] = []
    for rate_index, dropout_rate in enumerate(dropout_rates):
        domain_results: dict[str, dict[str, float]] = {}
        resident_bytes: list[int] = []
        for domain_index, domain in enumerate(("addition", "multiplication")):
            generator = torch.Generator().manual_seed(seed + 100 * rate_index + domain_index)
            losses: list[float] = []
            accuracies: list[float] = []
            for batch_index in range(batches):
                batch = task.sample(
                    batch_size,
                    generator=generator,
                    device=device,
                    domain=domain,
                )
                plan = PagePlan.random_dropout(
                    page_counts,
                    dropout_rate,
                    seed=seed + 10_000 * rate_index + 100 * domain_index + batch_index,
                )
                logits = model(batch.input_ids, plan=plan).logits
                losses.append(
                    masked_cross_entropy(logits, batch.targets, batch.answer_mask).item()
                )
                accuracies.append(
                    masked_accuracy(logits, batch.targets, batch.answer_mask).item()
                )
                resident_bytes.append(model.resident_parameter_bytes(plan))
            domain_results[domain] = {
                "loss": sum(losses) / len(losses),
                "accuracy": sum(accuracies) / len(accuracies),
            }
        mean_resident_bytes = sum(resident_bytes) / len(resident_bytes)
        rows.append(
            {
                "dropout_rate": dropout_rate,
                "mean_resident_parameter_bytes": mean_resident_bytes,
                "mean_resident_fraction": mean_resident_bytes / total_bytes,
                "addition_loss": domain_results["addition"]["loss"],
                "addition_accuracy": domain_results["addition"]["accuracy"],
                "multiplication_loss": domain_results["multiplication"]["loss"],
                "multiplication_accuracy": domain_results["multiplication"]["accuracy"],
            }
        )
    model.train(was_training)
    return rows


@torch.no_grad()
def page_ablation_utilities(
    model: ToyTransformer,
    task: TwoDomainArithmetic,
    *,
    domain: DomainName,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure full-model loss increase when one logical page is omitted."""

    was_training = model.training
    model.eval()
    batch = task.sample(
        batch_size,
        generator=torch.Generator().manual_seed(seed),
        device=device,
        domain=domain,
    )
    full_logits = model(batch.input_ids, plan=PagePlan.full()).logits
    full_loss = masked_cross_entropy(full_logits, batch.targets, batch.answer_mask)
    utilities: dict[str, float] = {}
    for layer_id, page_count in model.page_counts().items():
        for page_id in range(page_count):
            retained = tuple(index for index in range(page_count) if index != page_id)
            plan = PagePlan(selections={layer_id: retained}, default="all")
            ablated_logits = model(batch.input_ids, plan=plan).logits
            ablated_loss = masked_cross_entropy(
                ablated_logits,
                batch.targets,
                batch.answer_mask,
            )
            utilities[f"{layer_id}#{page_id}"] = (ablated_loss - full_loss).item()
    model.train(was_training)
    return utilities


def summarize_specialization(
    addition_utilities: dict[str, float],
    multiplication_utilities: dict[str, float],
) -> dict[str, Any]:
    if addition_utilities.keys() != multiplication_utilities.keys():
        raise ValueError("domain utility maps must contain identical pages")
    keys = list(addition_utilities)
    top_k = max(1, len(keys) // 4)
    addition_top = sorted(keys, key=addition_utilities.__getitem__, reverse=True)[:top_k]
    multiplication_top = sorted(
        keys,
        key=multiplication_utilities.__getitem__,
        reverse=True,
    )[:top_k]
    addition_set = set(addition_top)
    multiplication_set = set(multiplication_top)
    union = addition_set | multiplication_set

    addition_vector = torch.tensor([addition_utilities[key] for key in keys])
    multiplication_vector = torch.tensor([multiplication_utilities[key] for key in keys])
    cosine = F.cosine_similarity(addition_vector, multiplication_vector, dim=0, eps=1e-12)
    return {
        "top_k": top_k,
        "top_page_jaccard": len(addition_set & multiplication_set) / len(union),
        "utility_cosine_similarity": cosine.item(),
        "mean_absolute_utility_gap": (addition_vector - multiplication_vector).abs().mean().item(),
        "addition_top_pages": addition_top,
        "multiplication_top_pages": multiplication_top,
    }
