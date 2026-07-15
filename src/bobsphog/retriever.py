"""Learn and apply a counterfactual page-utility estimator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyTransformer
from bobsphog.synthetic import SyntheticBatch
from bobsphog.utility_data import UtilityExamples


class CounterfactualUtilityEstimator(nn.Module):
    """Score a candidate from query, page identity, and current resident set."""

    def __init__(self, query_size: int, page_count: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.page_embeddings = nn.Embedding(page_count, hidden_size)
        self.query_projection = nn.Linear(query_size, hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("utility_mean", torch.tensor(0.0))
        self.register_buffer("utility_scale", torch.tensor(1.0))

    def fit_utility_scale(self, utilities: Tensor) -> None:
        self.utility_mean.copy_(utilities.mean().to(self.utility_mean.device))
        self.utility_scale.copy_(utilities.std().clamp_min(1e-6).to(self.utility_scale.device))

    def forward(
        self,
        queries: Tensor,
        candidate_ids: Tensor,
        resident_masks: Tensor,
    ) -> Tensor:
        if resident_masks.shape != (queries.shape[0], self.page_embeddings.num_embeddings):
            raise ValueError("resident mask shape does not match batch and page count")
        query_features = self.query_projection(queries)
        candidate_features = self.page_embeddings(candidate_ids)
        resident_counts = resident_masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        resident_features = resident_masks @ self.page_embeddings.weight / resident_counts
        normalized = self.scorer(
            torch.cat((query_features, candidate_features, resident_features), dim=-1)
        ).squeeze(-1)
        return normalized * self.utility_scale + self.utility_mean


@dataclass(frozen=True)
class RetrieverTrainingSummary:
    initial_loss: float
    final_loss: float
    validation_rmse: float
    validation_correlation: float
    validation_sign_accuracy: float


def _correlation(left: Tensor, right: Tensor) -> Tensor:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if denominator.item() == 0.0:
        return torch.zeros((), device=left.device)
    return (left_centered * right_centered).sum() / denominator


def train_utility_estimator(
    estimator: CounterfactualUtilityEstimator,
    training: UtilityExamples,
    validation: UtilityExamples,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> RetrieverTrainingSummary:
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    estimator.to(device)
    estimator.fit_utility_scale(training.utilities)
    estimator.train()
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for _ in range(steps):
        indices = torch.randint(0, len(training), (batch_size,), generator=generator)
        queries = training.queries[indices].to(device)
        candidates = training.candidate_ids[indices].to(device)
        masks = training.resident_masks[indices].to(device)
        targets = training.utilities[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = estimator(queries, candidates, masks)
        normalized_error = (predictions - targets) / estimator.utility_scale
        loss = F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error))
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

    estimator.eval()
    with torch.no_grad():
        predictions = estimator(
            validation.queries.to(device),
            validation.candidate_ids.to(device),
            validation.resident_masks.to(device),
        )
        targets = validation.utilities.to(device)
        rmse = torch.sqrt(F.mse_loss(predictions, targets))
        correlation = _correlation(predictions, targets)
        sign_accuracy = predictions.gt(0).eq(targets.gt(0)).float().mean()
    window = min(20, len(losses))
    return RetrieverTrainingSummary(
        initial_loss=losses[0],
        final_loss=sum(losses[-window:]) / window,
        validation_rmse=rmse.item(),
        validation_correlation=correlation.item(),
        validation_sign_accuracy=sign_accuracy.item(),
    )


@torch.no_grad()
def learned_greedy_selection(
    model: ToyTransformer,
    batch: SyntheticBatch,
    catalog: PageCatalog,
    estimator: CounterfactualUtilityEstimator,
    budget: int,
) -> tuple[int, ...]:
    if not 0 <= budget <= len(catalog):
        raise ValueError("budget must be between zero and page count")
    model_was_training = model.training
    estimator_was_training = estimator.training
    model.eval()
    estimator.eval()
    device = batch.input_ids.device
    selected: list[int] = []
    for _ in range(budget):
        plan = catalog.plan(selected)
        query = model.hidden_states(batch.input_ids, plan=plan)[:, 1, :].mean(dim=0, keepdim=True)
        candidates = [index for index in range(len(catalog)) if index not in selected]
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        queries = query.expand(len(candidates), -1)
        resident_mask = catalog.resident_mask(selected, device=device).expand(len(candidates), -1)
        scores = estimator(queries, candidate_tensor, resident_mask)
        selected.append(candidates[scores.argmax().item()])
    model.train(model_was_training)
    estimator.train(estimator_was_training)
    return tuple(selected)


@torch.no_grad()
def learned_base_query_selection(
    model: ToyTransformer,
    batch: SyntheticBatch,
    catalog: PageCatalog,
    estimator: CounterfactualUtilityEstimator,
    budget: int,
) -> tuple[int, ...]:
    """Choose a complete prompt working set from one base-only query state."""

    if not 0 <= budget <= len(catalog):
        raise ValueError("budget must be between zero and page count")
    model_was_training = model.training
    estimator_was_training = estimator.training
    model.eval()
    estimator.eval()
    device = batch.input_ids.device
    base_plan = catalog.plan(())
    query = model.hidden_states(batch.input_ids, plan=base_plan)[:, 1, :].mean(dim=0, keepdim=True)
    selected: list[int] = []
    for _ in range(budget):
        candidates = [index for index in range(len(catalog)) if index not in selected]
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        queries = query.expand(len(candidates), -1)
        resident_mask = catalog.resident_mask(selected, device=device).expand(len(candidates), -1)
        scores = estimator(queries, candidate_tensor, resident_mask)
        selected.append(candidates[scores.argmax().item()])
    model.train(model_was_training)
    estimator.train(estimator_was_training)
    return tuple(selected)
