import torch

from bobsphog.conversion import convert_dense_to_paged
from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyConfig
from bobsphog.synthetic import TwoDomainArithmetic
from bobsphog.training import (
    OptimizationConfig,
    train_dense_teacher,
    train_multi_budget_student,
)


def test_teacher_and_student_training_loops_run() -> None:
    torch.manual_seed(8)
    device = torch.device("cpu")
    task = TwoDomainArithmetic(context_length=9)
    config = ToyConfig(
        vocab_size=task.VOCAB_SIZE,
        context_length=9,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        base_rank=2,
        page_rank=2,
    )
    optimization = OptimizationConfig(steps=2, batch_size=4)
    teacher = DenseToyTransformer(config)
    teacher_summary = train_dense_teacher(
        teacher,
        task,
        optimization,
        seed=1,
        device=device,
    )
    student = convert_dense_to_paged(teacher, base_rank=2, page_rank=2)
    student_summary = train_multi_budget_student(
        student,
        teacher,
        task,
        optimization,
        dropout_rates=(0.5, 0.75),
        distillation_weight=1.0,
        full_retention_weight=0.5,
        freeze_resident=True,
        seed=2,
        device=device,
    )

    assert torch.isfinite(torch.tensor(teacher_summary.final_loss))
    assert torch.isfinite(torch.tensor(student_summary.final_loss))
    assert student_summary.trainable_parameter_count < teacher_summary.trainable_parameter_count
