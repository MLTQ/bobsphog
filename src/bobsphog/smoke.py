"""Run a deterministic quality-versus-logical-residency smoke experiment."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from torch.nn import functional as F

from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.paging import PagePlan, PagingTrace


def run_smoke(seed: int = 7) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    config = ToyConfig(
        vocab_size=64,
        context_length=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        base_rank=4,
        page_rank=4,
    )
    model = ToyTransformer(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, config.context_length))
    page_counts = model.page_counts()
    max_pages = max(page_counts.values())

    with torch.no_grad():
        teacher_logits = model(input_ids).logits
        teacher_probabilities = teacher_logits.softmax(dim=-1)

        rows: list[dict[str, Any]] = []
        for pages_per_layer in range(max_pages + 1):
            plan = PagePlan.uniform_prefix(page_counts, pages_per_layer)
            trace = PagingTrace()
            logits = model(input_ids, plan=plan, trace=trace).logits
            kl_sum = F.kl_div(
                logits.log_softmax(dim=-1),
                teacher_probabilities,
                reduction="sum",
            )
            token_count = input_ids.numel()
            rows.append(
                {
                    "pages_per_layer": pages_per_layer,
                    "selected_pages": trace.selected_page_count,
                    "resident_parameter_bytes": model.resident_parameter_bytes(plan),
                    "full_parameter_bytes": model.total_parameter_bytes(),
                    "resident_fraction": model.resident_parameter_bytes(plan)
                    / model.total_parameter_bytes(),
                    "teacher_kl_per_token": kl_sum.item() / token_count,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.seed), indent=2))


if __name__ == "__main__":
    main()
