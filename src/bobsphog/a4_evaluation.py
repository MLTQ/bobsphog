"""Equal-budget A4 comparison of singleton and graph-aware page selectors."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import torch

from bobsphog.catalog import PageCatalog
from bobsphog.compositional import CompositionalArithmetic, CompositionalDomain
from bobsphog.model import ToyTransformer
from bobsphog.oracle import evaluate_selection, greedy_oracle_selection
from bobsphog.relationship import (
    SparseRelationshipGraph,
    graph_guided_learned_selection,
)
from bobsphog.retriever import CounterfactualUtilityEstimator, learned_greedy_selection


@torch.no_grad()
def compare_bundle_selectors(
    model: ToyTransformer,
    task: CompositionalArithmetic,
    catalog: PageCatalog,
    estimator: CounterfactualUtilityEstimator,
    graph: SparseRelationshipGraph,
    *,
    domain: CompositionalDomain,
    budgets: tuple[int, ...],
    batch_size: int,
    random_trials: int,
    seed: int,
    device: torch.device,
    relationship_weight: float,
) -> list[dict[str, Any]]:
    if not budgets or any(not 0 <= budget <= len(catalog) for budget in budgets):
        raise ValueError("budgets must be non-empty and within page count")
    calibration_batch = task.sample(
        batch_size,
        generator=torch.Generator().manual_seed(seed),
        device=device,
        domain=domain,
    )
    evaluation_batch = task.sample(
        batch_size,
        generator=torch.Generator().manual_seed(seed + 1),
        device=device,
        domain=domain,
    )
    max_budget = max(budgets)
    singleton_sequence = graph.singleton_ranking(max_budget)
    graph_sequence = graph.graph_greedy_selection(
        max_budget,
        relationship_weight=relationship_weight,
    )
    oracle_sequence = greedy_oracle_selection(
        model,
        calibration_batch,
        catalog,
        max_budget,
    ).selected_ids
    learned_sequence = learned_greedy_selection(
        model,
        evaluation_batch,
        catalog,
        estimator,
        max_budget,
    )
    learned_graph_sequence = graph_guided_learned_selection(
        model,
        evaluation_batch,
        catalog,
        estimator,
        graph,
        max_budget,
        relationship_weight=relationship_weight,
    )
    base = asdict(evaluate_selection(model, evaluation_batch, catalog, ()))
    full = asdict(
        evaluate_selection(model, evaluation_batch, catalog, tuple(range(len(catalog))))
    )
    policy_generator = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        selections = {
            "static_svd": catalog.static_prefix(budget),
            "singleton": singleton_sequence[:budget],
            "graph_bundle": graph_sequence[:budget],
            "learned": learned_sequence[:budget],
            "learned_graph": learned_graph_sequence[:budget],
            "oracle": oracle_sequence[:budget],
        }
        random_metrics = [
            evaluate_selection(
                model,
                evaluation_batch,
                catalog,
                tuple(policy_generator.sample(range(len(catalog)), budget)),
            )
            for _ in range(random_trials)
        ]
        row: dict[str, Any] = {
            "budget_pages": budget,
            "resident_fraction": model.resident_parameter_bytes(
                catalog.plan(selections["static_svd"])
            )
            / model.total_parameter_bytes(),
            "base": base,
            "random": {
                "loss": sum(metric.loss for metric in random_metrics) / random_trials,
                "accuracy": sum(metric.accuracy for metric in random_metrics) / random_trials,
            },
            "full": full,
        }
        for name, selected in selections.items():
            row[name] = {
                **asdict(evaluate_selection(model, evaluation_batch, catalog, selected)),
                "pages": catalog.names(selected),
            }
        rows.append(row)
    return rows
