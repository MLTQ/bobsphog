"""Harder two-domain compositional arithmetic task for A4."""

from __future__ import annotations

from typing import Literal

import torch

from bobsphog.synthetic import SyntheticBatch

CompositionalDomain = Literal["add_then_multiply", "multiply_then_add"]


class CompositionalArithmetic:
    """Generate two order-sensitive arithmetic compositions over shared tokens."""

    BOS = 0
    ADD_THEN_MULTIPLY = 1
    MULTIPLY_THEN_ADD = 2
    EQUALS = 3
    NUMBER_OFFSET = 4

    def __init__(self, context_length: int, *, base: int = 16) -> None:
        total_tokens = context_length + 1
        if total_tokens < 7 or (total_tokens - 2) % 5 != 0:
            raise ValueError("context_length + 1 must equal 2 + 5 * clause_count")
        if base < 4:
            raise ValueError("base must be at least four")
        self.context_length = context_length
        self.base = base
        self.vocab_size = self.NUMBER_OFFSET + base
        self.clause_count = (total_tokens - 2) // 5

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
        domain: CompositionalDomain | None = None,
    ) -> SyntheticBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if domain not in (None, "add_then_multiply", "multiply_then_add"):
            raise ValueError(f"unknown domain: {domain!r}")

        if domain is None:
            domains = torch.randint(0, 2, (batch_size,), generator=generator)
        else:
            domain_id = 0 if domain == "add_then_multiply" else 1
            domains = torch.full((batch_size,), domain_id, dtype=torch.long)
        operands = torch.randint(
            0,
            self.base,
            (batch_size, self.clause_count, 3),
            generator=generator,
        )
        first, second, third = operands.unbind(dim=-1)
        add_then_multiply = ((first + second) * third) % self.base
        multiply_then_add = (first * second + third) % self.base
        results = torch.where(
            domains[:, None] == 0,
            add_then_multiply,
            multiply_then_add,
        )

        tokens = torch.empty((batch_size, self.context_length + 1), dtype=torch.long)
        tokens[:, 0] = self.BOS
        tokens[:, 1] = torch.where(
            domains == 0,
            torch.tensor(self.ADD_THEN_MULTIPLY),
            torch.tensor(self.MULTIPLY_THEN_ADD),
        )
        answer_mask = torch.zeros((batch_size, self.context_length), dtype=torch.bool)
        for clause in range(self.clause_count):
            start = 2 + 5 * clause
            tokens[:, start] = first[:, clause] + self.NUMBER_OFFSET
            tokens[:, start + 1] = second[:, clause] + self.NUMBER_OFFSET
            tokens[:, start + 2] = third[:, clause] + self.NUMBER_OFFSET
            tokens[:, start + 3] = self.EQUALS
            tokens[:, start + 4] = results[:, clause] + self.NUMBER_OFFSET
            answer_mask[:, start + 3] = True

        return SyntheticBatch(
            input_ids=tokens[:, :-1].to(device),
            targets=tokens[:, 1:].to(device),
            answer_mask=answer_mask.to(device),
            domains=domains.to(device),
        )
