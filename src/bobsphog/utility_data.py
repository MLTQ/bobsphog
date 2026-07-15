"""Collect counterfactual page-utility supervision from direct model executions."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyTransformer
from bobsphog.objectives import masked_cross_entropy_per_example
from bobsphog.synthetic import TwoDomainArithmetic


@dataclass(frozen=True)
class UtilityExamples:
    queries: Tensor
    candidate_ids: Tensor
    resident_masks: Tensor
    utilities: Tensor

    def __len__(self) -> int:
        return self.utilities.shape[0]


@torch.no_grad()
def collect_utility_examples(
    model: ToyTransformer,
    task: TwoDomainArithmetic,
    catalog: PageCatalog,
    *,
    states: int,
    batch_size: int,
    candidates_per_state: int,
    resident_budgets: tuple[int, ...],
    domains: tuple[str, str] = ("addition", "multiplication"),
    seed: int,
    device: torch.device,
) -> UtilityExamples:
    """Measure per-example marginal gain from adding sampled omitted pages."""

    if states <= 0 or batch_size <= 0 or candidates_per_state <= 0:
        raise ValueError("states, batch_size, and candidates_per_state must be positive")
    if not resident_budgets or any(not 0 <= budget < len(catalog) for budget in resident_budgets):
        raise ValueError("resident budgets must be non-empty and below full page count")
    if len(domains) != 2 or domains[0] == domains[1]:
        raise ValueError("domains must contain two distinct names")

    was_training = model.training
    model.eval()
    data_generator = torch.Generator().manual_seed(seed)
    policy_generator = random.Random(seed)
    query_parts: list[Tensor] = []
    candidate_parts: list[Tensor] = []
    mask_parts: list[Tensor] = []
    utility_parts: list[Tensor] = []

    for state_index in range(states):
        domain = domains[state_index % 2]
        batch = task.sample(
            batch_size,
            generator=data_generator,
            device=device,
            domain=domain,
        )
        resident_budget = policy_generator.choice(resident_budgets)
        selected = tuple(policy_generator.sample(range(len(catalog)), resident_budget))
        selected_set = set(selected)
        omitted = [index for index in range(len(catalog)) if index not in selected_set]
        candidate_count = min(candidates_per_state, len(omitted))
        candidates = policy_generator.sample(omitted, candidate_count)
        plan = catalog.plan(selected)

        hidden = model.hidden_states(batch.input_ids, plan=plan)
        queries = hidden[:, 1, :]
        current_logits = model(batch.input_ids, plan=plan).logits
        current_losses = masked_cross_entropy_per_example(
            current_logits,
            batch.targets,
            batch.answer_mask,
        )
        resident_mask = catalog.resident_mask(selected, device=device)

        for candidate in candidates:
            added_plan = catalog.plan((*selected, candidate))
            added_logits = model(batch.input_ids, plan=added_plan).logits
            added_losses = masked_cross_entropy_per_example(
                added_logits,
                batch.targets,
                batch.answer_mask,
            )
            query_parts.append(queries.detach().cpu())
            candidate_parts.append(
                torch.full((batch_size,), candidate, dtype=torch.long)
            )
            mask_parts.append(resident_mask.expand(batch_size, -1).detach().cpu())
            utility_parts.append((current_losses - added_losses).detach().cpu())

    model.train(was_training)
    return UtilityExamples(
        queries=torch.cat(query_parts),
        candidate_ids=torch.cat(candidate_parts),
        resident_masks=torch.cat(mask_parts),
        utilities=torch.cat(utility_parts),
    )
