"""Run the A2 variable-page-dropout and multi-budget distillation experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from bobsphog.conversion import convert_dense_to_paged
from bobsphog.dense_model import DenseToyTransformer
from bobsphog.evaluation import (
    evaluate_budget_curve,
    evaluate_domain,
    evaluate_random_budget_curve,
    page_ablation_utilities,
    summarize_specialization,
)
from bobsphog.model import ToyConfig
from bobsphog.synthetic import TwoDomainArithmetic
from bobsphog.training import (
    OptimizationConfig,
    train_dense_teacher,
    train_multi_budget_student,
)


def default_model_config() -> ToyConfig:
    return ToyConfig(
        vocab_size=TwoDomainArithmetic.VOCAB_SIZE,
        context_length=17,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        base_rank=4,
        page_rank=4,
    )


@dataclass(frozen=True)
class A2Config:
    seed: int = 19
    device: str = "cpu"
    model: ToyConfig = field(default_factory=default_model_config)
    teacher: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(steps=400, batch_size=128)
    )
    student: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(steps=400, batch_size=128)
    )
    dropout_rates: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
    distillation_weight: float = 1.0
    full_retention_weight: float = 0.5
    freeze_resident: bool = True
    eval_batches: int = 4
    ablation_batch_size: int = 256


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def run_a2(config: A2Config) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    task = TwoDomainArithmetic(config.model.context_length)
    teacher = DenseToyTransformer(config.model).to(device)
    teacher_training = train_dense_teacher(
        teacher,
        task,
        config.teacher,
        seed=config.seed + 1,
        device=device,
    )
    teacher.eval()
    teacher_metrics = {
        domain: asdict(
            evaluate_domain(
                teacher,
                task,
                domain=domain,
                batch_size=config.teacher.batch_size,
                batches=config.eval_batches,
                seed=config.seed + 10,
                device=device,
            )
        )
        for domain in ("addition", "multiplication")
    }

    student = convert_dense_to_paged(
        teacher,
        base_rank=config.model.base_rank,
        page_rank=config.model.page_rank,
    ).to(device)
    initial_curve = evaluate_budget_curve(
        student,
        task,
        batch_size=config.student.batch_size,
        batches=config.eval_batches,
        seed=config.seed + 20,
        device=device,
    )
    random_evaluation_rates = (1.0, 0.9, 0.75, 0.5, 0.25, 0.0)
    initial_random_curve = evaluate_random_budget_curve(
        student,
        task,
        dropout_rates=random_evaluation_rates,
        batch_size=config.student.batch_size,
        batches=config.eval_batches,
        seed=config.seed + 25,
        device=device,
    )
    student_training = train_multi_budget_student(
        student,
        teacher,
        task,
        config.student,
        dropout_rates=config.dropout_rates,
        distillation_weight=config.distillation_weight,
        full_retention_weight=config.full_retention_weight,
        freeze_resident=config.freeze_resident,
        seed=config.seed + 2,
        device=device,
    )
    trained_curve = evaluate_budget_curve(
        student,
        task,
        batch_size=config.student.batch_size,
        batches=config.eval_batches,
        seed=config.seed + 20,
        device=device,
    )
    trained_random_curve = evaluate_random_budget_curve(
        student,
        task,
        dropout_rates=random_evaluation_rates,
        batch_size=config.student.batch_size,
        batches=config.eval_batches,
        seed=config.seed + 25,
        device=device,
    )

    addition_utilities = page_ablation_utilities(
        student,
        task,
        domain="addition",
        batch_size=config.ablation_batch_size,
        seed=config.seed + 30,
        device=device,
    )
    multiplication_utilities = page_ablation_utilities(
        student,
        task,
        domain="multiplication",
        batch_size=config.ablation_batch_size,
        seed=config.seed + 31,
        device=device,
    )
    return {
        "config": asdict(config),
        "device": str(device),
        "teacher_training": asdict(teacher_training),
        "teacher_metrics": teacher_metrics,
        "student_training": asdict(student_training),
        "initial_svd_budget_curve": initial_curve,
        "trained_budget_curve": trained_curve,
        "initial_random_budget_curve": initial_random_curve,
        "trained_random_budget_curve": trained_random_curve,
        "specialization": summarize_specialization(
            addition_utilities,
            multiplication_utilities,
        ),
        "page_utilities": {
            "addition": addition_utilities,
            "multiplication": multiplication_utilities,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-steps", type=int, default=400)
    parser.add_argument("--student-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, or auto")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--train-resident",
        action="store_true",
        help="allow the resident skeleton to train (normally frozen to prevent collapse)",
    )
    args = parser.parse_args()
    optimization = {
        "batch_size": args.batch_size,
        "learning_rate": 3e-3,
    }
    config = A2Config(
        seed=args.seed,
        device=args.device,
        teacher=OptimizationConfig(steps=args.teacher_steps, **optimization),
        student=OptimizationConfig(steps=args.student_steps, **optimization),
        eval_batches=args.eval_batches,
        ablation_batch_size=args.batch_size,
        freeze_resident=not args.train_resident,
    )
    print(json.dumps(run_a2(config), indent=2))


if __name__ == "__main__":
    main()
