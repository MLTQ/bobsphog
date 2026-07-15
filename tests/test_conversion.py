import torch

from bobsphog.conversion import convert_dense_to_paged
from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyConfig


def test_dense_teacher_matches_full_paged_student() -> None:
    torch.manual_seed(5)
    config = ToyConfig(
        vocab_size=16,
        context_length=9,
        d_model=12,
        n_heads=3,
        n_layers=2,
        d_ff=24,
        base_rank=2,
        page_rank=2,
    )
    teacher = DenseToyTransformer(config).eval()
    student = convert_dense_to_paged(teacher, base_rank=2, page_rank=2).eval()
    input_ids = torch.randint(0, config.vocab_size, (3, config.context_length))

    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits
        student_logits = student(input_ids).logits

    torch.testing.assert_close(student_logits, teacher_logits, rtol=2e-5, atol=2e-5)
