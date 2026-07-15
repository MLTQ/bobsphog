"""Direct-loss greedy page oracle and fixed selection metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyTransformer
from bobsphog.objectives import masked_accuracy, masked_cross_entropy
from bobsphog.synthetic import SyntheticBatch


@dataclass(frozen=True)
class SelectionMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class GreedySelection:
    selected_ids: tuple[int, ...]
    calibration_losses: tuple[float, ...]


@torch.no_grad()
def evaluate_selection(
    model: ToyTransformer,
    batch: SyntheticBatch,
    catalog: PageCatalog,
    selected_ids: tuple[int, ...],
) -> SelectionMetrics:
    logits = model(batch.input_ids, plan=catalog.plan(selected_ids)).logits
    return SelectionMetrics(
        loss=masked_cross_entropy(logits, batch.targets, batch.answer_mask).item(),
        accuracy=masked_accuracy(logits, batch.targets, batch.answer_mask).item(),
    )


@torch.no_grad()
def greedy_oracle_selection(
    model: ToyTransformer,
    calibration_batch: SyntheticBatch,
    catalog: PageCatalog,
    budget: int,
) -> GreedySelection:
    """Add the page with the best measured marginal calibration loss each step."""

    if not 0 <= budget <= len(catalog):
        raise ValueError("budget must be between zero and page count")
    was_training = model.training
    model.eval()
    selected: list[int] = []
    losses = [evaluate_selection(model, calibration_batch, catalog, ()).loss]
    for _ in range(budget):
        candidates = [index for index in range(len(catalog)) if index not in selected]
        candidate_losses = [
            evaluate_selection(
                model,
                calibration_batch,
                catalog,
                tuple(selected + [candidate]),
            ).loss
            for candidate in candidates
        ]
        best_position = min(range(len(candidates)), key=candidate_losses.__getitem__)
        selected.append(candidates[best_position])
        losses.append(candidate_losses[best_position])
    model.train(was_training)
    return GreedySelection(tuple(selected), tuple(losses))
