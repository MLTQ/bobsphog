"""Evaluate prompt-route predictors against held-out B2.2 decode unions."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class B22PredictConfig:
    corpus: str
    budgets: tuple[int, ...] = (1280, 2048, 2304)
    num_experts: int = 256
    page_bytes: int = 6_291_456
    cache_capacity_pages: int = 2560
    seed: int = 72


@dataclass(frozen=True)
class RouteExample:
    id: str
    domain: str
    split: str
    prefill: np.ndarray
    decode_counts: np.ndarray
    decode_groups: tuple[tuple[tuple[int, ...], ...], ...] = ()

    @property
    def target(self) -> np.ndarray:
        return self.decode_counts > 0


def load_trace_examples(
    path: str | Path,
    *,
    num_experts: int = 256,
) -> list[RouteExample]:
    payload = json.loads(Path(path).expanduser().read_text())
    records = payload.get("prompts", [])
    if not records:
        raise ValueError("trace corpus contains no prompts")
    examples: list[RouteExample] = []
    num_layers: int | None = None
    for record in records:
        prefill_layers = record["prefill_experts_by_layer"]
        decode_traces = record["decode_experts_by_token_and_layer"]
        if num_layers is None:
            num_layers = len(prefill_layers)
        if len(prefill_layers) != num_layers or any(
            len(trace) != num_layers for trace in decode_traces
        ):
            raise ValueError("all route records must share one layer count")
        prefill = np.zeros((num_layers, num_experts), dtype=np.float32)
        decode_counts = np.zeros((num_layers, num_experts), dtype=np.float32)
        for layer, experts in enumerate(prefill_layers):
            prefill[layer, np.asarray(experts, dtype=np.int64)] = 1.0
        for trace in decode_traces:
            for layer, experts in enumerate(trace):
                decode_counts[layer, np.asarray(experts, dtype=np.int64)] += 1.0
        decode_groups = tuple(
            tuple(tuple(int(expert) for expert in experts) for experts in trace)
            for trace in decode_traces
        )
        examples.append(
            RouteExample(
                id=str(record["id"]),
                domain=str(record["domain"]),
                split=str(record["split"]),
                prefill=prefill,
                decode_counts=decode_counts,
                decode_groups=decode_groups,
            )
        )
    return examples


def select_equal_layer_budget(scores: np.ndarray, budget: int) -> np.ndarray:
    """Select a deterministic near-equal number of pages from every layer."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [layers, experts]")
    layers, experts = scores.shape
    if not 0 <= budget <= layers * experts:
        raise ValueError("budget is outside the available page range")
    if budget == 0:
        return np.zeros_like(scores, dtype=bool)
    base, remainder = divmod(budget, layers)
    if base + (1 if remainder else 0) > experts:
        raise ValueError("per-layer budget exceeds expert count")
    selected = np.zeros_like(scores, dtype=bool)
    for layer in range(layers):
        count = base + (1 if layer < remainder else 0)
        order = np.argsort(-scores[layer], kind="stable")
        selected[layer, order[:count]] = True
    return selected


class PredictorSuite:
    """Fit simple route baselines and an executable relationship-index model."""

    def __init__(self, training: list[RouteExample], *, seed: int = 72) -> None:
        if not training:
            raise ValueError("training examples are required")
        self.training = training
        self.layers, self.experts = training[0].prefill.shape
        if any(example.prefill.shape != (self.layers, self.experts) for example in training):
            raise ValueError("training route shapes must match")
        self.train_prefill = np.stack([example.prefill for example in training])
        self.train_decode = np.stack([example.decode_counts for example in training])
        self.global_scores = self.train_decode.sum(axis=0)
        self.global_normalized = self.global_scores / np.maximum(
            self.global_scores.max(axis=1, keepdims=True), 1.0
        )
        self.seed = seed
        self.conditional = self._fit_conditional_index()

    def _fit_conditional_index(self) -> np.ndarray:
        conditional = np.zeros(
            (self.layers, self.experts, self.experts), dtype=np.float32
        )
        for layer in range(self.layers):
            features = self.train_prefill[:, layer, :].astype(np.float32)
            targets = self.train_decode[:, layer, :].astype(np.float32)
            coactivation = features.T @ targets
            frequency = features.sum(axis=0)
            conditional[layer] = coactivation / np.maximum(frequency[:, None], 1.0)
        return conditional

    def scores(
        self,
        method: str,
        example: RouteExample,
        *,
        alpha: float = 0.25,
        neighbors: int = 5,
    ) -> np.ndarray:
        if method == "global_frequency":
            return self.global_scores.copy()
        if method == "prefill_reuse":
            return self.global_normalized + example.prefill * 2.0
        if method == "conditional_coactivation":
            scores = np.zeros_like(self.global_scores)
            for layer in range(self.layers):
                active = np.flatnonzero(example.prefill[layer])
                if active.size:
                    scores[layer] = self.conditional[layer, active].mean(axis=0)
            return scores + alpha * self.global_normalized
        if method == "nearest_neighbor":
            query = example.prefill.reshape(-1).astype(bool)
            training = self.train_prefill.reshape(len(self.training), -1).astype(bool)
            intersection = np.logical_and(training, query).sum(axis=1)
            union = np.logical_or(training, query).sum(axis=1)
            similarity = intersection / np.maximum(union, 1)
            count = min(max(neighbors, 1), len(self.training))
            nearest = np.argsort(-similarity, kind="stable")[:count]
            weights = similarity[nearest] + 1e-6
            scores = np.tensordot(
                weights / weights.sum(), self.train_decode[nearest], axes=(0, 0)
            )
            return scores + 0.05 * self.global_normalized
        if method == "oracle_request_frequency":
            return example.decode_counts.copy()
        if method == "random":
            stable = sum((index + 1) * ord(char) for index, char in enumerate(example.id))
            rng = np.random.default_rng(self.seed + stable)
            return rng.random((self.layers, self.experts))
        raise ValueError(f"unknown predictor method: {method}")


def evaluate_selection(
    example: RouteExample,
    selected: np.ndarray,
    *,
    page_bytes: int,
) -> dict[str, int | float | str]:
    if selected.shape != example.target.shape:
        raise ValueError("selection shape must match route target")
    target = example.target
    intersection = np.logical_and(selected, target).sum()
    target_pages = int(target.sum())
    selected_pages = int(selected.sum())
    total_requests = float(example.decode_counts.sum())
    request_hits = float((example.decode_counts * selected).sum())
    missed_pages = target_pages - int(intersection)
    return {
        "id": example.id,
        "domain": example.domain,
        "split": example.split,
        "target_union_pages": target_pages,
        "selected_pages": selected_pages,
        "union_recall": float(intersection / target_pages) if target_pages else 1.0,
        "union_precision": float(intersection / selected_pages) if selected_pages else 1.0,
        "request_hit_fraction": request_hits / total_requests if total_requests else 1.0,
        "late_unique_pages": missed_pages,
        "late_unique_bytes": missed_pages * page_bytes,
    }


def simulate_pinned_bundle_lru(
    example: RouteExample,
    selected: np.ndarray,
    *,
    cache_capacity_pages: int,
    page_bytes: int,
) -> dict[str, int | float]:
    """Simulate a pinned predicted bundle plus LRU-managed residual capacity."""

    if selected.shape != example.target.shape:
        raise ValueError("selection shape must match route target")
    if not example.decode_groups:
        raise ValueError("ordered decode route groups are required for cache simulation")
    selected_pages = int(selected.sum())
    if cache_capacity_pages < selected_pages:
        raise ValueError("cache capacity cannot be smaller than the pinned bundle")
    transient_capacity = cache_capacity_pages - selected_pages
    transient: OrderedDict[tuple[int, int], None] = OrderedDict()
    pinned_hits = 0
    transient_hits = 0
    faults = 0
    total_requests = 0
    for token in example.decode_groups:
        for layer, experts in enumerate(token):
            for expert in experts:
                total_requests += 1
                key = (layer, expert)
                if selected[layer, expert]:
                    pinned_hits += 1
                elif key in transient:
                    transient_hits += 1
                    transient.move_to_end(key)
                else:
                    faults += 1
                    if transient_capacity:
                        transient[key] = None
                        while len(transient) > transient_capacity:
                            transient.popitem(last=False)
    hits = pinned_hits + transient_hits
    return {
        "cache_capacity_pages": cache_capacity_pages,
        "transient_capacity_pages": transient_capacity,
        "decode_requests": total_requests,
        "pinned_request_hits": pinned_hits,
        "transient_request_hits": transient_hits,
        "effective_cache_hits": hits,
        "effective_cache_hit_fraction": hits / total_requests if total_requests else 1.0,
        "simulated_page_faults": faults,
        "simulated_fault_bytes": faults * page_bytes,
        "bundle_prefetch_bytes": selected_pages * page_bytes,
        "cold_total_transfer_bytes": (selected_pages + faults) * page_bytes,
    }


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "target_union_pages",
        "union_recall",
        "union_precision",
        "request_hit_fraction",
        "late_unique_pages",
        "late_unique_bytes",
        "effective_cache_hit_fraction",
        "simulated_page_faults",
        "simulated_fault_bytes",
        "bundle_prefetch_bytes",
        "cold_total_transfer_bytes",
    )
    return {
        "examples": len(rows),
        **{field: mean(float(row[field]) for row in rows) for field in fields},
    }


def _evaluate_method(
    suite: PredictorSuite,
    examples: list[RouteExample],
    method: str,
    budget: int,
    *,
    page_bytes: int,
    cache_capacity_pages: int,
    alpha: float,
    neighbors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for example in examples:
        selected = select_equal_layer_budget(
            suite.scores(method, example, alpha=alpha, neighbors=neighbors),
            budget,
        )
        rows.append(
            {
                **evaluate_selection(example, selected, page_bytes=page_bytes),
                **simulate_pinned_bundle_lru(
                    example,
                    selected,
                    cache_capacity_pages=cache_capacity_pages,
                    page_bytes=page_bytes,
                ),
            }
        )
    domains = {
        domain: _mean_metrics([row for row in rows if row["domain"] == domain])
        for domain in sorted({str(row["domain"]) for row in rows})
    }
    return {**_mean_metrics(rows), "by_domain": domains}, rows


def tune_hyperparameters(
    suite: PredictorSuite,
    validation: list[RouteExample],
    *,
    budget: int,
    page_bytes: int,
) -> dict[str, float | int]:
    best_alpha = 0.0
    best_alpha_score = -1.0
    for alpha in (0.0, 0.05, 0.25, 0.5, 1.0):
        aggregate, _ = _evaluate_method(
            suite,
            validation,
            "conditional_coactivation",
            budget,
            page_bytes=page_bytes,
            cache_capacity_pages=budget,
            alpha=alpha,
            neighbors=5,
        )
        score = float(aggregate["request_hit_fraction"])
        if score > best_alpha_score:
            best_alpha, best_alpha_score = alpha, score

    best_neighbors = 1
    best_neighbor_score = -1.0
    for neighbors in (1, 3, 5, 10, 20):
        aggregate, _ = _evaluate_method(
            suite,
            validation,
            "nearest_neighbor",
            budget,
            page_bytes=page_bytes,
            cache_capacity_pages=budget,
            alpha=best_alpha,
            neighbors=neighbors,
        )
        score = float(aggregate["request_hit_fraction"])
        if score > best_neighbor_score:
            best_neighbors, best_neighbor_score = neighbors, score
    return {
        "conditional_alpha": best_alpha,
        "conditional_validation_request_hit_fraction": best_alpha_score,
        "nearest_neighbors": best_neighbors,
        "nearest_validation_request_hit_fraction": best_neighbor_score,
    }


def run_b22_predict(config: B22PredictConfig) -> dict[str, Any]:
    if not config.budgets or any(budget < 0 for budget in config.budgets):
        raise ValueError("budgets must be non-negative")
    examples = load_trace_examples(config.corpus, num_experts=config.num_experts)
    training = [example for example in examples if example.split == "train"]
    validation = [example for example in examples if example.split == "validation"]
    test = [example for example in examples if example.split == "test"]
    if not training or not validation or not test:
        raise ValueError("trace corpus must contain train, validation, and test prompts")
    suite = PredictorSuite(training, seed=config.seed)
    tune_budget = min(config.budgets, key=lambda value: abs(value - 2048))
    tuned = tune_hyperparameters(
        suite,
        validation,
        budget=tune_budget,
        page_bytes=config.page_bytes,
    )
    methods = (
        "random",
        "global_frequency",
        "prefill_reuse",
        "nearest_neighbor",
        "conditional_coactivation",
        "oracle_request_frequency",
    )
    results: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    for split_name, split_examples in (("validation", validation), ("test", test)):
        for budget in config.budgets:
            for method in methods:
                aggregate, rows = _evaluate_method(
                    suite,
                    split_examples,
                    method,
                    budget,
                    page_bytes=config.page_bytes,
                    cache_capacity_pages=config.cache_capacity_pages,
                    alpha=float(tuned["conditional_alpha"]),
                    neighbors=int(tuned["nearest_neighbors"]),
                )
                results.append(
                    {
                        "split": split_name,
                        "budget_pages": budget,
                        "method": method,
                        **aggregate,
                    }
                )
                per_example.extend(
                    {
                        "method": method,
                        "budget_pages": budget,
                        **row,
                    }
                    for row in rows
                )
    return {
        "config": asdict(config),
        "dataset": {
            "examples": len(examples),
            "train": len(training),
            "validation": len(validation),
            "test": len(test),
            "layers": suite.layers,
            "experts_per_layer": suite.experts,
        },
        "tuned": tuned,
        "results": results,
        "test_examples": [
            row for row in per_example if row["split"] == "test"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1280, 2048, 2304])
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--page-bytes", type=int, default=6_291_456)
    parser.add_argument("--cache-capacity-pages", type=int, default=2560)
    parser.add_argument("--seed", type=int, default=72)
    args = parser.parse_args()
    args.budgets = tuple(args.budgets)
    print(json.dumps(run_b22_predict(B22PredictConfig(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
