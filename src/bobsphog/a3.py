"""Run A3 oracle and learned counterfactual page-selection comparisons."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from bobsphog.a2 import default_model_config, resolve_device
from bobsphog.a3_evaluation import compare_selectors
from bobsphog.catalog import PageCatalog
from bobsphog.conversion import convert_dense_to_paged
from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyConfig
from bobsphog.retriever import CounterfactualUtilityEstimator, train_utility_estimator
from bobsphog.synthetic import TwoDomainArithmetic
from bobsphog.training import (
    OptimizationConfig,
    train_dense_teacher,
    train_multi_budget_student,
)
from bobsphog.utility_data import collect_utility_examples


@dataclass(frozen=True)
class A3Config:
    seed: int = 29
    device: str = "cpu"
    model: ToyConfig = field(default_factory=default_model_config)
    teacher: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(steps=400, batch_size=128)
    )
    student: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(steps=400, batch_size=128)
    )
    dropout_rates: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
    utility_states: int = 24
    utility_validation_states: int = 8
    utility_batch_size: int = 32
    utility_candidates_per_state: int = 12
    utility_resident_budgets: tuple[int, ...] = (0, 2, 4, 8)
    estimator_hidden_size: int = 64
    estimator_steps: int = 500
    estimator_batch_size: int = 256
    estimator_learning_rate: float = 2e-3
    selection_budgets: tuple[int, ...] = (2, 4, 8)
    evaluation_batch_size: int = 128
    random_trials: int = 12


def _utility_stats(utilities: torch.Tensor) -> dict[str, float]:
    return {
        "mean": utilities.mean().item(),
        "standard_deviation": utilities.std().item(),
        "positive_fraction": utilities.gt(0).float().mean().item(),
    }


def _selection_summary(comparisons: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for domain_rows in comparisons.values() for row in domain_rows]
    oracle_gains = [row["oracle"]["accuracy"] - row["random"]["accuracy"] for row in rows]
    learned_gains = [row["learned"]["accuracy"] - row["random"]["accuracy"] for row in rows]
    oracle_static_gains = [
        row["oracle"]["accuracy"] - row["static_svd"]["accuracy"] for row in rows
    ]
    learned_static_gains = [
        row["learned"]["accuracy"] - row["static_svd"]["accuracy"] for row in rows
    ]
    learned_regrets = [row["oracle"]["accuracy"] - row["learned"]["accuracy"] for row in rows]
    return {
        "mean_oracle_accuracy_gain_over_random": sum(oracle_gains) / len(oracle_gains),
        "mean_learned_accuracy_gain_over_random": sum(learned_gains) / len(learned_gains),
        "mean_oracle_accuracy_gain_over_static_svd": sum(oracle_static_gains)
        / len(oracle_static_gains),
        "mean_learned_accuracy_gain_over_static_svd": sum(learned_static_gains)
        / len(learned_static_gains),
        "mean_learned_accuracy_regret_to_oracle": sum(learned_regrets) / len(learned_regrets),
        "oracle_beats_random_rows": sum(gain > 0 for gain in oracle_gains),
        "learned_beats_random_rows": sum(gain > 0 for gain in learned_gains),
        "oracle_beats_static_svd_rows": sum(gain > 0 for gain in oracle_static_gains),
        "learned_beats_static_svd_rows": sum(gain > 0 for gain in learned_static_gains),
        "comparison_rows": len(rows),
    }


def run_a3(config: A3Config) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    task = TwoDomainArithmetic(config.model.context_length)
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

    comparisons = {
        domain: compare_selectors(
            student,
            task,
            catalog,
            estimator,
            domain=domain,
            budgets=config.selection_budgets,
            batch_size=config.evaluation_batch_size,
            random_trials=config.random_trials,
            seed=config.seed + 100 * domain_index,
            device=device,
        )
        for domain_index, domain in enumerate(("addition", "multiplication"))
    }
    return {
        "config": asdict(config),
        "device": str(device),
        "teacher_training": asdict(teacher_summary),
        "student_training": asdict(student_summary),
        "page_count": len(catalog),
        "page_parameter_bytes": sorted({ref.parameter_bytes for ref in catalog.refs}),
        "utility_training_examples": len(training_examples),
        "utility_validation_examples": len(validation_examples),
        "utility_training_stats": _utility_stats(training_examples.utilities),
        "utility_validation_stats": _utility_stats(validation_examples.utilities),
        "estimator_training": asdict(estimator_summary),
        "comparisons": comparisons,
        "selection_summary": _selection_summary(comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-steps", type=int, default=400)
    parser.add_argument("--student-steps", type=int, default=400)
    parser.add_argument("--estimator-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--utility-states", type=int, default=24)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, or auto")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    config = A3Config(
        seed=args.seed,
        device=args.device,
        teacher=OptimizationConfig(steps=args.teacher_steps, batch_size=args.batch_size),
        student=OptimizationConfig(steps=args.student_steps, batch_size=args.batch_size),
        utility_states=args.utility_states,
        estimator_steps=args.estimator_steps,
        evaluation_batch_size=args.batch_size,
    )
    print(json.dumps(run_a3(config), indent=2))


if __name__ == "__main__":
    main()
