"""Run an out-of-core Qwen3.6-35B-A3B one-token inference smoke test."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from bobsphog.expert_cache import CudaExpertCache, ExpertCacheStats
from bobsphog.moe_checkpoint import (
    MappedExpertSource,
    QwenMoeSpec,
    SafetensorCheckpointIndex,
)
from bobsphog.moe_model import load_paged_qwen


@dataclass(frozen=True)
class B1Config:
    checkpoint: str
    device: str = "cuda:0"
    cache_pages: int = 320
    prompt: str = "Hello"
    top_tokens: int = 5


def _stats_delta(before: ExpertCacheStats, after: ExpertCacheStats) -> dict[str, Any]:
    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in asdict(before)
    }


def run_b1(config: B1Config) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("B1 requires CUDA")
    if config.cache_pages <= 0 or config.top_tokens <= 0:
        raise ValueError("cache_pages and top_tokens must be positive")
    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    device = torch.device(config.device)
    torch.cuda.set_device(device)

    spec = QwenMoeSpec.from_files(
        checkpoint_root / "config.json",
        checkpoint_root / "model.safetensors.index.json",
    )
    index = SafetensorCheckpointIndex(
        checkpoint_root / "model.safetensors.index.json"
    )
    source = MappedExpertSource(index, spec)
    cache = CudaExpertCache(
        source,
        device=device,
        capacity_bytes=config.cache_pages * spec.expert_bytes(),
        num_experts=spec.num_experts,
    )
    torch.cuda.reset_peak_memory_stats(device)
    model, load_summary = load_paged_qwen(
        checkpoint_root,
        cache,
        device=device,
    )
    torch.cuda.synchronize(device)
    scaffold_allocated = torch.cuda.memory_allocated(device)

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers is required for B1") from error
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_root,
        local_files_only=True,
    )
    encoded = tokenizer(config.prompt, return_tensors="pt").input_ids
    if encoded.shape[1] == 0:
        raise ValueError("prompt produced no tokens")
    input_ids = encoded[:, :1].to(device)

    before_cold = cache.snapshot()
    cold_started = perf_counter()
    with torch.inference_mode():
        cold_output = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        cold_logits = cold_output.logits
    torch.cuda.synchronize(device)
    cold_ms = (perf_counter() - cold_started) * 1000
    after_cold = cache.snapshot()
    cold_peak = torch.cuda.max_memory_allocated(device)
    del cold_output

    before_warm = cache.snapshot()
    warm_started = perf_counter()
    with torch.inference_mode():
        warm_output = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        warm_logits = warm_output.logits
    torch.cuda.synchronize(device)
    warm_ms = (perf_counter() - warm_started) * 1000
    after_warm = cache.snapshot()

    next_input_id = warm_logits[:, -1].argmax(dim=-1, keepdim=True)
    before_decode = cache.snapshot()
    decode_started = perf_counter()
    with torch.inference_mode():
        decode_output = model(
            input_ids=next_input_id,
            attention_mask=torch.ones((1, 2), dtype=torch.long, device=device),
            past_key_values=warm_output.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize(device)
    decode_ms = (perf_counter() - decode_started) * 1000
    after_decode = cache.snapshot()

    top_values, top_ids = torch.topk(
        cold_logits[0, -1].float(),
        k=config.top_tokens,
    )
    result = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "spec": source.describe(),
        "scaffold_load": asdict(load_summary),
        "scaffold_allocated_bytes": scaffold_allocated,
        "cold_peak_allocated_bytes": cold_peak,
        "expert_cache_capacity_bytes": cache.capacity_bytes,
        "expert_cache_final_bytes": cache.cache_bytes,
        "input_token_id": int(input_ids[0, 0]),
        "input_token": tokenizer.decode(input_ids[0]),
        "cold_forward_ms": cold_ms,
        "warm_forward_ms": warm_ms,
        "next_token_decode_ms": decode_ms,
        "cold_cache": _stats_delta(before_cold, after_cold),
        "warm_cache": _stats_delta(before_warm, after_warm),
        "next_token_cache": _stats_delta(before_decode, after_decode),
        "next_input_token_id": int(next_input_id[0, 0]),
        "next_input_token": tokenizer.decode(next_input_id[0]),
        "decode_argmax_token_id": int(
            decode_output.logits[0, -1].argmax().item()
        ),
        "decode_argmax_token": tokenizer.decode(
            [int(decode_output.logits[0, -1].argmax().item())]
        ),
        "warm_max_absolute_logit_error": (
            cold_logits.float() - warm_logits.float()
        ).abs().max().item(),
        "top_next_tokens": [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "logit": float(logit),
            }
            for token_id, logit in zip(top_ids, top_values)
        ],
        "final_cache_stats": asdict(cache.snapshot()),
    }
    cache.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-pages", type=int, default=320)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--top-tokens", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_b1(B1Config(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
