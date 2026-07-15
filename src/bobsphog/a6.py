"""Run learned prompt selection through the physical CUDA page cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import torch

from bobsphog.a2 import default_model_config, resolve_device
from bobsphog.catalog import PageCatalog
from bobsphog.conversion import convert_dense_to_paged
from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyConfig
from bobsphog.objectives import masked_accuracy
from bobsphog.physical_cache import CacheStats, PhysicalPageCache
from bobsphog.retriever import (
    CounterfactualUtilityEstimator,
    learned_base_query_selection,
    train_utility_estimator,
)
from bobsphog.synthetic import SyntheticBatch, TwoDomainArithmetic
from bobsphog.training import (
    OptimizationConfig,
    train_dense_teacher,
    train_multi_budget_student,
)
from bobsphog.utility_data import collect_utility_examples


@dataclass(frozen=True)
class A6Config:
    seed: int = 61
    device: str = "cuda:0"
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
    selection_budget: int = 8
    evaluation_batch_size: int = 128


def _stats_delta(before: CacheStats, after: CacheStats) -> dict[str, float | int]:
    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in asdict(before)
    }


def _accuracy(logits: torch.Tensor, batch: SyntheticBatch) -> float:
    return masked_accuracy(logits, batch.targets, batch.answer_mask).item()


def _select(
    student: torch.nn.Module,
    batch: SyntheticBatch,
    catalog: PageCatalog,
    estimator: CounterfactualUtilityEstimator,
    budget: int,
    device: torch.device,
) -> tuple[tuple[int, ...], float]:
    torch.cuda.synchronize(device)
    started = perf_counter()
    selected = learned_base_query_selection(
        student,
        batch,
        catalog,
        estimator,
        budget,
    )
    torch.cuda.synchronize(device)
    return selected, (perf_counter() - started) * 1000


def run_a6(config: A6Config) -> dict[str, Any]:
    device = resolve_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("A6 requires CUDA")
    torch.manual_seed(config.seed)
    torch.cuda.set_device(device)
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
    if not 0 < config.selection_budget <= len(catalog):
        raise ValueError("selection budget must be positive and fit the catalog")

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

    batches = {
        domain: task.sample(
            config.evaluation_batch_size,
            generator=torch.Generator().manual_seed(config.seed + 100 + domain_index),
            device=device,
            domain=domain,
        )
        for domain_index, domain in enumerate(("addition", "multiplication"))
    }
    learned_base_query_selection(
        student,
        batches["addition"],
        catalog,
        estimator,
        config.selection_budget,
    )
    torch.cuda.synchronize(device)
    selections: dict[str, tuple[int, ...]] = {}
    selection_ms: dict[str, float] = {}
    for domain, batch in batches.items():
        selections[domain], selection_ms[domain] = _select(
            student,
            batch,
            catalog,
            estimator,
            config.selection_budget,
            device,
        )

    plans = {domain: catalog.plan(selected) for domain, selected in selections.items()}
    reference_logits: dict[str, torch.Tensor] = {}
    quality: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for domain, batch in batches.items():
            base_logits = student(batch.input_ids, plan=catalog.plan(())).logits
            selected_logits = student(batch.input_ids, plan=plans[domain]).logits
            full_logits = student(batch.input_ids).logits
            reference_logits[domain] = selected_logits.float().cpu()
            quality[domain] = {
                "base_accuracy": _accuracy(base_logits, batch),
                "selected_accuracy": _accuracy(selected_logits, batch),
                "full_accuracy": _accuracy(full_logits, batch),
                "selection_ms": selection_ms[domain],
            }
    del teacher
    torch.cuda.synchronize(device)

    capacity_bytes = max(
        catalog.selected_bytes(selected)
        for selected in selections.values()
    )
    cache = PhysicalPageCache(
        student,
        device=device,
        capacity_bytes=capacity_bytes,
        dtype=torch.float32,
    )
    skeleton_allocated = torch.cuda.memory_allocated(device)
    physical: dict[str, dict[str, Any]] = {}
    for domain in ("addition", "multiplication"):
        before = cache.snapshot()
        started = perf_counter()
        cache.schedule(plans[domain])
        with torch.inference_mode():
            logits = student(batches[domain].input_ids, plan=plans[domain]).logits
        torch.cuda.synchronize(device)
        elapsed_ms = (perf_counter() - started) * 1000
        after = cache.snapshot()
        physical[domain] = {
            "end_to_end_ms": elapsed_ms,
            "cache_delta": _stats_delta(before, after),
            "max_absolute_logit_error": (
                logits.float().cpu() - reference_logits[domain]
            ).abs().max().item(),
            "accuracy": _accuracy(logits, batches[domain]),
            "allocated_bytes": torch.cuda.memory_allocated(device),
        }
        del logits

    overlap = set(selections["addition"]) & set(selections["multiplication"])
    result = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "teacher_training": asdict(teacher_summary),
        "student_training": asdict(student_summary),
        "estimator_training": asdict(estimator_summary),
        "page_count": len(catalog),
        "page_parameter_bytes": sorted({ref.parameter_bytes for ref in catalog.refs}),
        "plans": {
            domain: {
                "global_ids": list(selected),
                "names": catalog.names(selected),
                **quality[domain],
            }
            for domain, selected in selections.items()
        },
        "plan_overlap_pages": len(overlap),
        "plan_overlap_names": catalog.names(sorted(overlap)),
        "cache_capacity_bytes": capacity_bytes,
        "skeleton_allocated_bytes": skeleton_allocated,
        "source_page_bytes": cache.source_bytes,
        "physical_execution": physical,
        "final_cache_stats": asdict(cache.snapshot()),
    }
    cache.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-steps", type=int, default=400)
    parser.add_argument("--student-steps", type=int, default=400)
    parser.add_argument("--estimator-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--utility-states", type=int, default=24)
    parser.add_argument("--selection-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()
    print(
        json.dumps(
            run_a6(
                A6Config(
                    seed=args.seed,
                    device=args.device,
                    teacher=OptimizationConfig(
                        steps=args.teacher_steps,
                        batch_size=args.batch_size,
                    ),
                    student=OptimizationConfig(
                        steps=args.student_steps,
                        batch_size=args.batch_size,
                    ),
                    utility_states=args.utility_states,
                    estimator_steps=args.estimator_steps,
                    evaluation_batch_size=args.batch_size,
                    selection_budget=args.selection_budget,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
