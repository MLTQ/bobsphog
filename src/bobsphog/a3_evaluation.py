"""Equal-budget comparison of random, static, oracle, and learned selectors."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import torch

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyTransformer
from bobsphog.oracle import evaluate_selection, greedy_oracle_selection
from bobsphog.retriever import CounterfactualUtilityEstimator, learned_greedy_selection
from bobsphog.synthetic import DomainName, TwoDomainArithmetic


@torch.no_grad()
def compare_selectors(
    model: ToyTransformer,
    task: TwoDomainArithmetic,
    catalog: PageCatalog,
    estimator: CounterfactualUtilityEstimator,
    *,
    domain: DomainName,
    budgets: tuple[int, ...],
    batch_size: int,
    random_trials: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if not budgets or any(not 0 <= budget <= len(catalog) for budget in budgets):
        raise ValueError("budgets must be non-empty and within the page count")
    if random_trials <= 0:
        raise ValueError("random_trials must be positive")

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
    base = asdict(evaluate_selection(model, evaluation_batch, catalog, ()))
    full_ids = tuple(range(len(catalog)))
    full = asdict(evaluate_selection(model, evaluation_batch, catalog, full_ids))
    policy_generator = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for budget in budgets:
        static_ids = catalog.static_prefix(budget)
        oracle_ids = oracle_sequence[:budget]
        learned_ids = learned_sequence[:budget]
        random_metrics = []
        for _ in range(random_trials):
            random_ids = tuple(policy_generator.sample(range(len(catalog)), budget))
            random_metrics.append(evaluate_selection(model, evaluation_batch, catalog, random_ids))
        random_result = {
            "loss": sum(metric.loss for metric in random_metrics) / random_trials,
            "accuracy": sum(metric.accuracy for metric in random_metrics) / random_trials,
        }
        plan = catalog.plan(static_ids)
        resident_bytes = model.resident_parameter_bytes(plan)
        rows.append(
            {
                "budget_pages": budget,
                "resident_parameter_bytes": resident_bytes,
                "resident_fraction": resident_bytes / model.total_parameter_bytes(),
                "base": base,
                "random": random_result,
                "static_svd": {
                    **asdict(evaluate_selection(model, evaluation_batch, catalog, static_ids)),
                    "pages": catalog.names(static_ids),
                },
                "oracle": {
                    **asdict(evaluate_selection(model, evaluation_batch, catalog, oracle_ids)),
                    "pages": catalog.names(oracle_ids),
                },
                "learned": {
                    **asdict(evaluate_selection(model, evaluation_batch, catalog, learned_ids)),
                    "pages": catalog.names(learned_ids),
                },
                "full": full,
            }
        )
    return rows
