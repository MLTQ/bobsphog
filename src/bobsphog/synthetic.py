"""Deterministic two-domain causal task generation for toy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

DomainName = Literal["addition", "multiplication"]


@dataclass(frozen=True)
class SyntheticBatch:
    input_ids: Tensor
    targets: Tensor
    answer_mask: Tensor
    domains: Tensor


class TwoDomainArithmetic:
    """Generate addition and multiplication examples over shared number tokens."""

    BOS = 0
    ADDITION = 1
    MULTIPLICATION = 2
    EQUALS = 3
    NUMBER_OFFSET = 4
    BASE = 10
    VOCAB_SIZE = NUMBER_OFFSET + BASE

    def __init__(self, context_length: int) -> None:
        total_tokens = context_length + 1
        if total_tokens < 6 or (total_tokens - 2) % 4 != 0:
            raise ValueError("context_length + 1 must equal 2 + 4 * clause_count")
        self.context_length = context_length
        self.clause_count = (total_tokens - 2) // 4

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
        domain: DomainName | None = None,
    ) -> SyntheticBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if domain not in (None, "addition", "multiplication"):
            raise ValueError(f"unknown domain: {domain!r}")

        if domain is None:
            domains = torch.randint(0, 2, (batch_size,), generator=generator)
        else:
            domain_id = 0 if domain == "addition" else 1
            domains = torch.full((batch_size,), domain_id, dtype=torch.long)

        operands = torch.randint(
            0,
            self.BASE,
            (batch_size, self.clause_count, 2),
            generator=generator,
        )
        left = operands[:, :, 0]
        right = operands[:, :, 1]
        addition_results = (left + right) % self.BASE
        multiplication_results = (left * right) % self.BASE
        results = torch.where(
            domains[:, None] == 0,
            addition_results,
            multiplication_results,
        )

        tokens = torch.empty((batch_size, self.context_length + 1), dtype=torch.long)
        tokens[:, 0] = self.BOS
        tokens[:, 1] = torch.where(
            domains == 0,
            torch.tensor(self.ADDITION),
            torch.tensor(self.MULTIPLICATION),
        )
        answer_mask = torch.zeros((batch_size, self.context_length), dtype=torch.bool)
        for clause in range(self.clause_count):
            start = 2 + 4 * clause
            tokens[:, start] = left[:, clause] + self.NUMBER_OFFSET
            tokens[:, start + 1] = right[:, clause] + self.NUMBER_OFFSET
            tokens[:, start + 2] = self.EQUALS
            tokens[:, start + 3] = results[:, clause] + self.NUMBER_OFFSET
            answer_mask[:, start + 2] = True

        return SyntheticBatch(
            input_ids=tokens[:, :-1].to(device),
            targets=tokens[:, 1:].to(device),
            answer_mask=answer_mask.to(device),
            domains=domains.to(device),
        )
