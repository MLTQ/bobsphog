"""Run A4 compositional-task and sparse relationship-graph experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from bobsphog.a2 import resolve_device
from bobsphog.a4_evaluation import compare_bundle_selectors
from bobsphog.catalog import PageCatalog
from bobsphog.compositional import CompositionalArithmetic
from bobsphog.conversion import convert_dense_to_paged
from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyConfig
from bobsphog.relationship import build_relationship_graph
from bobsphog.retriever import CounterfactualUtilityEstimator, train_utility_estimator
from bobsphog.training import (
    OptimizationConfig,
    train_dense_teacher,
    train_multi_budget_student,
)
from bobsphog.utility_data import collect_utility_examples


def default_a4_model_config() -> ToyConfig:
    task = CompositionalArithmetic(context_length=16, base=8)
    return ToyConfig(
        vocab_size=task.vocab_size,
        context_length=task.context_length,
        d_model=48,
        n_heads=4,
        n_layers=2,
        d_ff=96,
        dropout=0.0,
        base_rank=4,
        page_rank=4,
    )


@dataclass(frozen=True)
class A4Config:
    seed: int = 41
    device: str = "cpu"
    modulus: int = 8
    model: ToyConfig = field(default_factory=default_a4_model_config)
    teacher: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(steps=1600, batch_size=128)
    )
    student: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(steps=800, batch_size=128)
    )
    dropout_rates: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
    utility_states: int = 24
    utility_validation_states: int = 8
    utility_batch_size: int = 32
    utility_candidates_per_state: int = 12
    utility_resident_budgets: tuple[int, ...] = (0, 2, 4, 8)
    estimator_hidden_size: int = 96
    estimator_steps: int = 600
    estimator_batch_size: int = 256
    estimator_learning_rate: float = 2e-3
    graph_candidate_pool: int = 12
    graph_neighbors_per_page: int = 4
    relationship_weight: float = 1.0
    selection_budgets: tuple[int, ...] = (2, 4, 8)
    evaluation_batch_size: int = 128
    random_trials: int = 12


def _summarize(comparisons: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for domain_rows in comparisons.values() for row in domain_rows]

    def mean_difference(left: str, right: str) -> float:
        return sum(row[left]["accuracy"] - row[right]["accuracy"] for row in rows) / len(rows)

    def wins(left: str, right: str) -> int:
        return sum(row[left]["accuracy"] > row[right]["accuracy"] for row in rows)

    return {
        "mean_graph_gain_over_singleton": mean_difference("graph_bundle", "singleton"),
        "mean_learned_graph_gain_over_learned": mean_difference(
            "learned_graph", "learned"
        ),
        "mean_graph_gain_over_random": mean_difference("graph_bundle", "random"),
        "mean_learned_graph_gain_over_random": mean_difference(
            "learned_graph", "random"
        ),
        "mean_graph_gain_over_static_svd": mean_difference(
            "graph_bundle", "static_svd"
        ),
        "mean_learned_graph_gain_over_static_svd": mean_difference(
            "learned_graph", "static_svd"
        ),
        "mean_oracle_gain_over_graph": mean_difference("oracle", "graph_bundle"),
        "graph_beats_singleton_rows": wins("graph_bundle", "singleton"),
        "learned_graph_beats_learned_rows": wins("learned_graph", "learned"),
        "graph_beats_random_rows": wins("graph_bundle", "random"),
        "learned_graph_beats_random_rows": wins("learned_graph", "random"),
        "graph_beats_static_svd_rows": wins("graph_bundle", "static_svd"),
        "learned_graph_beats_static_svd_rows": wins("learned_graph", "static_svd"),
        "comparison_rows": len(rows),
    }


def run_a4(config: A4Config) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    task = CompositionalArithmetic(config.model.context_length, base=config.modulus)
    if task.vocab_size != config.model.vocab_size:
        raise ValueError("task vocabulary and model vocabulary differ")

    teacher = DenseToyTransformer(config.model).to(device)
    teacher_summary = train_dense_teacher(
        teacher,
        task,
        config.teacher,
        seed=config.seed + 1,
        device=device,
    )
    teacher.eval()
    student = convert_dense_to_paged(
        teacher,
        base_rank=config.model.base_rank,
        page_rank=config.model.page_rank,
    ).to(device)
    student_summary = train_multi_budget_student(
        student,
        teacher,
        task,
        config.student,
        dropout_rates=config.dropout_rates,
        distillation_weight=1.0,
        full_retention_weight=0.5,
        freeze_resident=True,
        seed=config.seed + 2,
        device=device,
    )
    student.eval()
    catalog = PageCatalog.from_model(student)

    training_examples = collect_utility_examples(
        student,
        task,
        catalog,
        states=config.utility_states,
        batch_size=config.utility_batch_size,
        candidates_per_state=config.utility_candidates_per_state,
        resident_budgets=config.utility_resident_budgets,
        domains=("add_then_multiply", "multiply_then_add"),
        seed=config.seed + 10,
        device=device,
    )
    validation_examples = collect_utility_examples(
        student,
        task,
        catalog,
        states=config.utility_validation_states,
        batch_size=config.utility_batch_size,
        candidates_per_state=config.utility_candidates_per_state,
        resident_budgets=config.utility_resident_budgets,
        domains=("add_then_multiply", "multiply_then_add"),
        seed=config.seed + 11,
        device=device,
    )
    estimator = CounterfactualUtilityEstimator(
        query_size=config.model.d_model,
        page_count=len(catalog),
        hidden_size=config.estimator_hidden_size,
    )
    estimator_summary = train_utility_estimator(
        estimator,
        training_examples,
        validation_examples,
        steps=config.estimator_steps,
        batch_size=config.estimator_batch_size,
        learning_rate=config.estimator_learning_rate,
        seed=config.seed + 12,
        device=device,
    )

    domains = ("add_then_multiply", "multiply_then_add")
    graphs = {}
    comparisons = {}
    graph_stats = {}
    for domain_index, domain in enumerate(domains):
        graph_batch = task.sample(
            config.evaluation_batch_size,
            generator=torch.Generator().manual_seed(config.seed + 100 + domain_index),
            device=device,
            domain=domain,
        )
        graph = build_relationship_graph(
            student,
            graph_batch,
            catalog,
            candidate_pool=config.graph_candidate_pool,
            neighbors_per_page=config.graph_neighbors_per_page,
        )
        graphs[domain] = graph
        graph_stats[domain] = {
            "edge_count": len(graph.edges),
            "edge_density": 2 * len(graph.edges) / (len(catalog) * (len(catalog) - 1)),
            "mean_signed_interaction": (
                sum(graph.edges.values()) / len(graph.edges) if graph.edges else 0.0
            ),
            "mean_absolute_interaction": (
                sum(abs(value) for value in graph.edges.values()) / len(graph.edges)
                if graph.edges
                else 0.0
            ),
            "minimum_interaction": min(graph.edges.values(), default=0.0),
            "maximum_interaction": max(graph.edges.values(), default=0.0),
        }
        comparisons[domain] = compare_bundle_selectors(
            student,
            task,
            catalog,
            estimator,
            graph,
            domain=domain,
            budgets=config.selection_budgets,
            batch_size=config.evaluation_batch_size,
            random_trials=config.random_trials,
            seed=config.seed + 1000 + 100 * domain_index,
            device=device,
            relationship_weight=config.relationship_weight,
        )

    return {
        "config": asdict(config),
        "device": str(device),
        "teacher_training": asdict(teacher_summary),
        "student_training": asdict(student_summary),
        "page_count": len(catalog),
        "utility_training_examples": len(training_examples),
        "utility_validation_examples": len(validation_examples),
        "estimator_training": asdict(estimator_summary),
        "graph_stats": graph_stats,
        "comparisons": comparisons,
        "selection_summary": _summarize(comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-steps", type=int, default=1600)
    parser.add_argument("--student-steps", type=int, default=800)
    parser.add_argument("--estimator-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--utility-states", type=int, default=24)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, or auto")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    config = A4Config(
        seed=args.seed,
        device=args.device,
        teacher=OptimizationConfig(steps=args.teacher_steps, batch_size=args.batch_size),
        student=OptimizationConfig(steps=args.student_steps, batch_size=args.batch_size),
        estimator_steps=args.estimator_steps,
        utility_states=args.utility_states,
        evaluation_batch_size=args.batch_size,
    )
    print(json.dumps(run_a4(config), indent=2))


if __name__ == "__main__":
    main()
