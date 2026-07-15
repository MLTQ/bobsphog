"""Run a minimal exact demand-paged GLM-5.2 feasibility probe."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from bobsphog.expert_cache import CudaExpertCache
from bobsphog.glm_checkpoint import (
    GlmMoeSpec,
    GlmSafetensorCheckpointIndex,
    MappedGlmExpertSource,
)
from bobsphog.glm_model import load_paged_glm


@dataclass(frozen=True)
class GlmB15Config:
    checkpoint: str
    device: str = "cuda:0"
    cache_pages: int = 16
    seed_text: str = "Hello"
    top_k: int = 5


def _stats_delta(before: Any, after: Any) -> dict[str, int | float]:
    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in asdict(before)
    }


def run_glm_b15(config: GlmB15Config) -> dict[str, Any]:
    """Load the GLM scaffold and execute one exact paged forward."""

    if not torch.cuda.is_available():
        raise RuntimeError("GLM B1.5 requires CUDA or HIP")
    if config.cache_pages <= 0 or config.top_k <= 0:
        raise ValueError("cache_pages and top_k must be positive")

    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    spec = GlmMoeSpec.from_files(
        checkpoint_root / "config.json",
        checkpoint_root / "model.safetensors.index.json",
    )
    if config.cache_pages < spec.experts_per_token:
        raise ValueError(
            f"cache must hold at least {spec.experts_per_token} one-token experts"
        )
    index = GlmSafetensorCheckpointIndex(
        checkpoint_root / "model.safetensors.index.json"
    )
    source = MappedGlmExpertSource(index, spec)
    cache = CudaExpertCache(
        source,
        device=device,
        capacity_bytes=config.cache_pages * spec.expert_bytes(),
        num_experts=spec.num_experts,
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers 5.12+ is required for GLM B1.5") from error

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = perf_counter()
    model, load_summary = load_paged_glm(checkpoint_root, cache, device=device)
    total_load_seconds = perf_counter() - load_started
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_root, local_files_only=True)
    input_ids = tokenizer(
        config.seed_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    if input_ids.shape[1] != 1:
        raise ValueError(
            "the feasibility probe requires seed_text that encodes to exactly one token; "
            f"observed {input_ids.shape[1]}"
        )

    before = cache.snapshot()
    torch.cuda.synchronize(device)
    started = perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(device)
    forward_seconds = perf_counter() - started
    after = cache.snapshot()
    logits = output.logits[0, -1]
    values, indices = torch.topk(logits, k=min(config.top_k, logits.numel()))
    candidates = [
        {
            "token_id": int(token_id),
            "text": tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for value, token_id in zip(values.cpu(), indices.cpu())
    ]
    peak_allocated = torch.cuda.max_memory_allocated(device)
    result = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "checkpoint": source.describe(),
        "load": {
            **asdict(load_summary),
            "total_model_construction_seconds": total_load_seconds,
        },
        "input_token_id": int(input_ids[0, 0]),
        "forward_seconds": forward_seconds,
        "peak_allocated_bytes": peak_allocated,
        "peak_fraction_of_checkpoint": peak_allocated / spec.checkpoint_bytes,
        "expert_cache_capacity_bytes": cache.capacity_bytes,
        "forward_cache": _stats_delta(before, after),
        "final_cache_stats": asdict(after),
        "next_token_candidates": candidates,
        "validation_scope": (
            "Paged feasibility only: a full-resident 1.5 TB reference cannot fit on "
            "the 128 GiB Strix Halo host."
        ),
    }
    cache.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-pages", type=int, default=16)
    parser.add_argument("--seed-text", default="Hello")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_glm_b15(GlmB15Config(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
