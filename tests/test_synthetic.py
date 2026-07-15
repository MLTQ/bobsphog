import torch

from bobsphog.synthetic import TwoDomainArithmetic


def test_addition_answers_and_mask_are_aligned() -> None:
    task = TwoDomainArithmetic(context_length=9)
    batch = task.sample(
        4,
        generator=torch.Generator().manual_seed(1),
        domain="addition",
    )

    answer_indices = batch.answer_mask[0].nonzero().flatten().tolist()
    assert answer_indices == [4, 8]
    for row in range(4):
        for answer_index in answer_indices:
            left = batch.input_ids[row, answer_index - 2] - task.NUMBER_OFFSET
            right = batch.input_ids[row, answer_index - 1] - task.NUMBER_OFFSET
            expected = (left + right) % task.BASE + task.NUMBER_OFFSET
            assert batch.targets[row, answer_index] == expected


def test_multiplication_uses_same_number_vocabulary() -> None:
    task = TwoDomainArithmetic(context_length=9)
    batch = task.sample(
        8,
        generator=torch.Generator().manual_seed(2),
        domain="multiplication",
    )

    answers = batch.targets.masked_select(batch.answer_mask)
    assert torch.all(answers >= task.NUMBER_OFFSET)
    assert torch.all(answers < task.VOCAB_SIZE)
    assert torch.all(batch.domains == 1)
