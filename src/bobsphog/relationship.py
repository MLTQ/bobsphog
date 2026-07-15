"""Sparse pair-synergy graphs and graph-guided page bundle selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyTransformer
from bobsphog.oracle import evaluate_selection
from bobsphog.retriever import CounterfactualUtilityEstimator
from bobsphog.synthetic import SyntheticBatch


@dataclass(frozen=True)
class SparseRelationshipGraph:
    page_count: int
    singleton_utilities: tuple[float, ...]
    edges: dict[tuple[int, int], float]

    def edge_weight(self, left: int, right: int) -> float:
        return self.edges.get(tuple(sorted((left, right))), 0.0)

    def singleton_ranking(self, budget: int) -> tuple[int, ...]:
        if not 0 <= budget <= self.page_count:
            raise ValueError("budget must be between zero and page count")
        return tuple(
            sorted(
                range(self.page_count),
                key=self.singleton_utilities.__getitem__,
                reverse=True,
            )[:budget]
        )

    def graph_greedy_selection(
        self,
        budget: int,
        *,
        relationship_weight: float = 1.0,
    ) -> tuple[int, ...]:
        if relationship_weight < 0:
            raise ValueError("relationship_weight must be non-negative")
        selected: list[int] = []
        for _ in range(budget):
            candidates = [index for index in range(self.page_count) if index not in selected]
            scores = [
                self.singleton_utilities[candidate]
                + relationship_weight
                * sum(self.edge_weight(candidate, resident) for resident in selected)
                for candidate in candidates
            ]
            selected.append(candidates[max(range(len(candidates)), key=scores.__getitem__)])
        return tuple(selected)


@torch.no_grad()
def build_relationship_graph(
    model: ToyTransformer,
    calibration_batch: SyntheticBatch,
    catalog: PageCatalog,
    *,
    candidate_pool: int,
    neighbors_per_page: int,
) -> SparseRelationshipGraph:
    """Measure singleton utility and sparse signed pair interaction from base-only."""

    if candidate_pool < 2 or neighbors_per_page <= 0:
        raise ValueError("candidate_pool must be at least two and neighbors positive")
    base_loss = evaluate_selection(model, calibration_batch, catalog, ()).loss
    singleton_utilities = tuple(
        base_loss - evaluate_selection(model, calibration_batch, catalog, (page_id,)).loss
        for page_id in range(len(catalog))
    )
    pool = sorted(
        range(len(catalog)),
        key=singleton_utilities.__getitem__,
        reverse=True,
    )[: min(candidate_pool, len(catalog))]
    measured_edges: dict[tuple[int, int], float] = {}
    incident: dict[int, list[tuple[int, float]]] = {page_id: [] for page_id in pool}
    for left_position, left in enumerate(pool):
        for right in pool[left_position + 1 :]:
            pair_utility = base_loss - evaluate_selection(
                model,
                calibration_batch,
                catalog,
                (left, right),
            ).loss
            synergy = pair_utility - singleton_utilities[left] - singleton_utilities[right]
            if synergy != 0:
                key = (min(left, right), max(left, right))
                measured_edges[key] = synergy
                incident[left].append((right, synergy))
                incident[right].append((left, synergy))

    retained: set[tuple[int, int]] = set()
    for page_id, relationships in incident.items():
        for neighbor, _ in sorted(relationships, key=lambda item: abs(item[1]), reverse=True)[
            :neighbors_per_page
        ]:
            retained.add((min(page_id, neighbor), max(page_id, neighbor)))
    return SparseRelationshipGraph(
        page_count=len(catalog),
        singleton_utilities=singleton_utilities,
        edges={key: measured_edges[key] for key in retained},
    )


@torch.no_grad()
def graph_guided_learned_selection(
    model: ToyTransformer,
    batch: SyntheticBatch,
    catalog: PageCatalog,
    estimator: CounterfactualUtilityEstimator,
    graph: SparseRelationshipGraph,
    budget: int,
    *,
    relationship_weight: float = 1.0,
) -> tuple[int, ...]:
    """Add sparse graph synergy to learned marginal utility during selection."""

    if graph.page_count != len(catalog):
        raise ValueError("graph and catalog page counts differ")
    if not 0 <= budget <= len(catalog):
        raise ValueError("budget must be between zero and page count")
    selected: list[int] = []
    device = batch.input_ids.device
    model_was_training = model.training
    estimator_was_training = estimator.training
    model.eval()
    estimator.eval()
    for _ in range(budget):
        candidates = [index for index in range(len(catalog)) if index not in selected]
        plan = catalog.plan(selected)
        query = model.hidden_states(batch.input_ids, plan=plan)[:, 1, :].mean(dim=0, keepdim=True)
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        predicted = estimator(
            query.expand(len(candidates), -1),
            candidate_tensor,
            catalog.resident_mask(selected, device=device).expand(len(candidates), -1),
        )
        bonuses = torch.tensor(
            [
                sum(graph.edge_weight(candidate, resident) for resident in selected)
                for candidate in candidates
            ],
            device=device,
        )
        scores = predicted + relationship_weight * bonuses
        selected.append(candidates[scores.argmax().item()])
    model.train(model_was_training)
    estimator.train(estimator_was_training)
    return tuple(selected)
