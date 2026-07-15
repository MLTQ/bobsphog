"""Calibrate confidence and choose adaptive prompt-bundle budgets for B2.3."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from bobsphog.b22_predict import (
    PredictorSuite,
    RouteExample,
    load_trace_examples,
    select_equal_layer_budget,
)
from bobsphog.cache_simulation import simulate_phased_pinned_lru
from bobsphog.expert_cache import ExpertKey


@dataclass(frozen=True)
class B23PolicyConfig:
    corpus: str
    budgets: tuple[int, ...] = (
        0,
        1024,
        1536,
        1792,
        2048,
        2304,
        2400,
        2480,
        2520,
    )
    cache_pages: int = 2560
    neighbors: int = 3
    exposed_prefetch_cost_per_page: float = 0.0
    confidence_quantile: float = 0.80


@dataclass(frozen=True)
class RidgeCalibrator:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        return design @ self.coefficients


def route_groups(
    example: RouteExample,
) -> tuple[tuple[tuple[ExpertKey, ...], ...], tuple[tuple[ExpertKey, ...], ...]]:
    prefill = tuple(
        tuple((layer, int(expert)) for expert in np.flatnonzero(example.prefill[layer]))
        for layer in range(example.prefill.shape[0])
    )
    decode = tuple(
        tuple((layer, int(expert)) for expert in experts)
        for token in example.decode_groups
        for layer, experts in enumerate(token)
    )
    return prefill, decode


def selected_keys(selected: np.ndarray) -> tuple[ExpertKey, ...]:
    return tuple(
        (layer, int(expert))
        for layer in range(selected.shape[0])
        for expert in np.flatnonzero(selected[layer])
    )


def simulate_decode_misses(
    example: RouteExample,
    selected: np.ndarray,
    *,
    cache_pages: int,
) -> int:
    prefill, decode = route_groups(example)
    return simulate_phased_pinned_lru(
        prefill,
        decode,
        cache_pages,
        pinned=selected_keys(selected),
        preload_pinned=False,
        pin_during_prefill=False,
    ).decode.misses


def _nearest(
    suite: PredictorSuite,
    example: RouteExample,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = example.prefill.reshape(-1).astype(bool)
    training = suite.train_prefill.reshape(len(suite.training), -1).astype(bool)
    intersection = np.logical_and(training, query).sum(axis=1)
    union = np.logical_or(training, query).sum(axis=1)
    similarities = intersection / np.maximum(union, 1)
    count = min(max(neighbors, 1), len(suite.training))
    indices = np.argsort(-similarities, kind="stable")[:count]
    weights = similarities[indices] + 1e-6
    weights /= weights.sum()
    return indices, similarities[indices], weights


def confidence_features(
    suite: PredictorSuite,
    example: RouteExample,
    budget: int,
    *,
    cache_pages: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build deployable confidence features without reading the query target."""

    scores = suite.scores("nearest_neighbor", example, neighbors=neighbors)
    selected = select_equal_layer_budget(scores, budget)
    indices, similarities, weights = _nearest(suite, example, neighbors)
    total_score = float(scores.sum())
    score_mass = float(scores[selected].sum() / total_score) if total_score else 0.0
    prefill_pages = float(example.prefill.sum())
    prefill_coverage = (
        float((example.prefill * selected).sum() / prefill_pages)
        if prefill_pages
        else 0.0
    )
    neighbor_request_hits = np.asarray(
        [
            float(
                (suite.training[index].decode_counts * selected).sum()
                / max(float(suite.training[index].decode_counts.sum()), 1.0)
            )
            for index in indices
        ]
    )
    neighbor_savings = []
    empty = np.zeros_like(selected)
    for index in indices:
        neighbor = suite.training[int(index)]
        baseline = simulate_decode_misses(
            neighbor, empty, cache_pages=cache_pages
        )
        pinned = simulate_decode_misses(
            neighbor, selected, cache_pages=cache_pages
        )
        neighbor_savings.append(baseline - pinned)
    savings = np.asarray(neighbor_savings, dtype=np.float64)
    budget_fraction = budget / cache_pages
    similarity_margin = (
        float(similarities[0] - similarities[1])
        if len(similarities) > 1
        else float(similarities[0])
    )
    features = np.asarray(
        [
            budget_fraction,
            budget_fraction**2,
            float(similarities[0]),
            float(similarities.mean()),
            float(similarities.std()),
            similarity_margin,
            score_mass,
            prefill_coverage,
            float(np.dot(weights, neighbor_request_hits)),
            float(neighbor_request_hits.std()),
            float(np.dot(weights, savings) / cache_pages),
            float(savings.std() / cache_pages),
            prefill_pages / (example.prefill.shape[0] * example.prefill.shape[1]),
        ],
        dtype=np.float64,
    )
    diagnostics = {
        "nearest_similarity": float(similarities[0]),
        "mean_neighbor_similarity": float(similarities.mean()),
        "similarity_margin": similarity_margin,
        "predicted_score_mass": score_mass,
        "prefill_coverage": prefill_coverage,
        "neighbor_request_hit": float(np.dot(weights, neighbor_request_hits)),
        "neighbor_savings_pages": float(np.dot(weights, savings)),
    }
    return features, selected, diagnostics


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> RidgeCalibrator:
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    return RidgeCalibrator(coefficients, feature_mean, feature_scale, alpha)


def _labeled_rows(
    query: RouteExample,
    suite: PredictorSuite,
    budgets: tuple[int, ...],
    *,
    cache_pages: int,
    neighbors: int,
) -> list[dict[str, Any]]:
    empty = np.zeros_like(query.prefill, dtype=bool)
    baseline_misses = simulate_decode_misses(
        query, empty, cache_pages=cache_pages
    )
    rows = []
    for budget in budgets:
        features, selected, diagnostics = confidence_features(
            suite,
            query,
            budget,
            cache_pages=cache_pages,
            neighbors=neighbors,
        )
        misses = simulate_decode_misses(
            query, selected, cache_pages=cache_pages
        )
        rows.append(
            {
                "id": query.id,
                "domain": query.domain,
                "budget_pages": budget,
                "features": features,
                "actual_decode_misses": misses,
                "actual_savings_pages": baseline_misses - misses,
                **diagnostics,
            }
        )
    return rows


def _training_rows(
    training: list[RouteExample],
    budgets: tuple[int, ...],
    *,
    cache_pages: int,
    neighbors: int,
) -> list[dict[str, Any]]:
    rows = []
    for held_out in training:
        suite = PredictorSuite(
            [example for example in training if example.id != held_out.id]
        )
        rows.extend(
            _labeled_rows(
                held_out,
                suite,
                budgets,
                cache_pages=cache_pages,
                neighbors=neighbors,
            )
        )
    return rows


def _choose_budget(
    rows: list[dict[str, Any]],
    calibrator: RidgeCalibrator,
    *,
    uncertainty_pages: float,
    exposed_cost: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = np.stack([row["features"] for row in rows])
    predictions = calibrator.predict(features)
    candidates = []
    for row, prediction in zip(rows, predictions):
        budget = int(row["budget_pages"])
        lower_bound = 0.0 if budget == 0 else float(prediction - uncertainty_pages)
        utility = lower_bound - exposed_cost * budget
        candidates.append(
            {
                **row,
                "predicted_savings_pages": 0.0 if budget == 0 else float(prediction),
                "confidence_lower_savings_pages": lower_bound,
                "predicted_net_utility_pages": utility,
            }
        )
    return max(candidates, key=lambda row: row["predicted_net_utility_pages"]), candidates


def run_b23_policy(config: B23PolicyConfig) -> dict[str, Any]:
    if not config.budgets or config.budgets[0] != 0:
        raise ValueError("budgets must begin with the no-pinning zero control")
    if any(budget < 0 or budget >= config.cache_pages for budget in config.budgets):
        raise ValueError("budgets must fit below total cache capacity")
    if not 0 < config.confidence_quantile < 1:
        raise ValueError("confidence quantile must be between zero and one")
    examples = load_trace_examples(Path(config.corpus))
    training = [example for example in examples if example.split == "train"]
    validation = [example for example in examples if example.split == "validation"]
    test = [example for example in examples if example.split == "test"]
    train_rows = _training_rows(
        training,
        config.budgets,
        cache_pages=config.cache_pages,
        neighbors=config.neighbors,
    )
    train_features = np.stack([row["features"] for row in train_rows])
    train_targets = np.asarray(
        [row["actual_savings_pages"] for row in train_rows], dtype=np.float64
    )
    suite = PredictorSuite(training)
    validation_rows = [
        row
        for example in validation
        for row in _labeled_rows(
            example,
            suite,
            config.budgets,
            cache_pages=config.cache_pages,
            neighbors=config.neighbors,
        )
    ]
    validation_features = np.stack([row["features"] for row in validation_rows])
    validation_targets = np.asarray(
        [row["actual_savings_pages"] for row in validation_rows], dtype=np.float64
    )
    candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        calibrator = fit_ridge(train_features, train_targets, alpha=alpha)
        errors = calibrator.predict(validation_features) - validation_targets
        candidates.append((float(np.sqrt(np.mean(errors**2))), calibrator, errors))
    validation_rmse, calibrator, validation_errors = min(
        candidates, key=lambda item: item[0]
    )
    uncertainty = float(
        np.quantile(np.abs(validation_errors), config.confidence_quantile)
    )

    test_rows_by_example: list[list[dict[str, Any]]] = []
    test_choices = []
    test_candidates = []
    for example in test:
        rows = _labeled_rows(
            example,
            suite,
            config.budgets,
            cache_pages=config.cache_pages,
            neighbors=config.neighbors,
        )
        choice, evaluated = _choose_budget(
            rows,
            calibrator,
            uncertainty_pages=uncertainty,
            exposed_cost=config.exposed_prefetch_cost_per_page,
        )
        test_rows_by_example.append(rows)
        test_choices.append(choice)
        test_candidates.extend(evaluated)

    static = {}
    for budget in config.budgets:
        rows = [row for row in test_candidates if row["budget_pages"] == budget]
        static[str(budget)] = {
            "decode_misses": sum(row["actual_decode_misses"] for row in rows),
            "decode_savings_pages": sum(row["actual_savings_pages"] for row in rows),
            "exposed_prefetch_cost_pages": budget * len(rows) * config.exposed_prefetch_cost_per_page,
            "net_utility_pages": sum(row["actual_savings_pages"] for row in rows)
            - budget * len(rows) * config.exposed_prefetch_cost_per_page,
        }
    adaptive_savings = sum(row["actual_savings_pages"] for row in test_choices)
    adaptive_cost = sum(
        row["budget_pages"] * config.exposed_prefetch_cost_per_page
        for row in test_choices
    )
    serializable_choices = [
        {key: value for key, value in row.items() if key != "features"}
        for row in test_choices
    ]
    sensitivity = []
    absolute_errors = np.abs(validation_errors)
    for quantile in (0.0, 0.5, 0.8, 0.9):
        bound = float(np.quantile(absolute_errors, quantile))
        for exposed_cost in (0.0, 0.05, 0.10, 0.20):
            choices = [
                _choose_budget(
                    rows,
                    calibrator,
                    uncertainty_pages=bound,
                    exposed_cost=exposed_cost,
                )[0]
                for rows in test_rows_by_example
            ]
            savings = sum(row["actual_savings_pages"] for row in choices)
            cost = sum(row["budget_pages"] * exposed_cost for row in choices)
            sensitivity.append(
                {
                    "confidence_quantile": quantile,
                    "uncertainty_pages": bound,
                    "exposed_prefetch_cost_per_page": exposed_cost,
                    "budget_counts": dict(
                        sorted(Counter(row["budget_pages"] for row in choices).items())
                    ),
                    "mean_budget_pages": mean(row["budget_pages"] for row in choices),
                    "decode_savings_pages": savings,
                    "net_utility_pages": savings - cost,
                }
            )

    oracle_choices = []
    for rows in test_rows_by_example:
        oracle_choices.append(
            max(
                rows,
                key=lambda row: row["actual_savings_pages"]
                - config.exposed_prefetch_cost_per_page * row["budget_pages"],
            )
        )
    oracle_savings = sum(row["actual_savings_pages"] for row in oracle_choices)
    oracle_cost = sum(
        row["budget_pages"] * config.exposed_prefetch_cost_per_page
        for row in oracle_choices
    )
    return {
        "config": asdict(config),
        "dataset": {
            "train": len(training),
            "validation": len(validation),
            "test": len(test),
            "cross_validated_training_rows": len(train_rows),
        },
        "calibration": {
            "ridge_alpha": calibrator.alpha,
            "validation_rmse_pages": validation_rmse,
            "confidence_absolute_error_quantile_pages": uncertainty,
        },
        "adaptive": {
            "budget_counts": dict(
                sorted(Counter(row["budget_pages"] for row in test_choices).items())
            ),
            "mean_budget_pages": mean(row["budget_pages"] for row in test_choices),
            "decode_misses": sum(row["actual_decode_misses"] for row in test_choices),
            "decode_savings_pages": adaptive_savings,
            "exposed_prefetch_cost_pages": adaptive_cost,
            "net_utility_pages": adaptive_savings - adaptive_cost,
            "choices": serializable_choices,
        },
        "adaptive_sensitivity": sensitivity,
        "oracle_adaptive": {
            "budget_counts": dict(
                sorted(Counter(row["budget_pages"] for row in oracle_choices).items())
            ),
            "decode_savings_pages": oracle_savings,
            "net_utility_pages": oracle_savings - oracle_cost,
        },
        "static": static,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[0, 1024, 1536, 1792, 2048, 2304, 2400, 2480, 2520],
    )
    parser.add_argument("--cache-pages", type=int, default=2560)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--exposed-prefetch-cost-per-page", type=float, default=0.0)
    parser.add_argument("--confidence-quantile", type=float, default=0.80)
    args = parser.parse_args()
    args.budgets = tuple(args.budgets)
    print(json.dumps(run_b23_policy(B23PolicyConfig(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
