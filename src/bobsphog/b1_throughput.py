"""Measure full-resident and demand-paged Qwen token throughput."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from bobsphog.expert_cache import CudaExpertCache, ExpertCacheStats
from bobsphog.moe_checkpoint import (
    MappedExpertSource,
    QwenMoeSpec,
    SafetensorCheckpointIndex,
)
from bobsphog.moe_model import load_paged_qwen
from bobsphog.reference_model import load_reference_qwen


DEFAULT_PROMPT = (
    "Explain why the sky appears blue during the day in clear, concise language."
)


@dataclass(frozen=True)
class B1ThroughputConfig:
    checkpoint: str
    mode: str = "both"
    device: str = "cuda:0"
    cache_pages: int = 320
    output_tokens: int = 16
    prompt: str = DEFAULT_PROMPT


def _stats_delta(before: ExpertCacheStats, after: ExpertCacheStats) -> dict[str, Any]:
    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in asdict(before)
    }


def summarize_decode_latencies(latencies: list[float]) -> dict[str, float | int]:
    """Summarize synchronized per-token decode latencies."""

    if not latencies or any(value <= 0 for value in latencies):
        raise ValueError("decode latencies must be a non-empty positive list")
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    total = sum(latencies)
    return {
        "tokens": len(latencies),
        "seconds": total,
        "tokens_per_second": len(latencies) / total,
        "mean_seconds_per_token": total / len(latencies),
        "median_seconds_per_token": median(latencies),
        "p95_seconds_per_token": ordered[p95_index],
        "minimum_seconds_per_token": ordered[0],
        "maximum_seconds_per_token": ordered[-1],
    }


def _format_prompt(tokenizer: Any, prompt: str) -> Tensor:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(rendered, return_tensors="pt").input_ids[0]


def _benchmark_path(
    model: Any,
    tokenizer: Any,
    prompt_ids: Tensor,
    *,
    device: torch.device,
    output_tokens: int,
    forced_token_ids: list[int] | None = None,
    cache: CudaExpertCache | None = None,
) -> dict[str, Any]:
    if forced_token_ids is not None and len(forced_token_ids) != output_tokens:
        raise ValueError("forced_token_ids length must equal output_tokens")

    before_cache = cache.snapshot() if cache is not None else None
    input_ids = prompt_ids.unsqueeze(0).to(device)
    torch.cuda.synchronize(device)
    prefill_started = perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize(device)
    prefill_seconds = perf_counter() - prefill_started
    after_prefill_cache = cache.snapshot() if cache is not None else None

    predicted_ids = [int(output.logits[0, -1].argmax())]
    selected_ids = [
        forced_token_ids[0] if forced_token_ids is not None else predicted_ids[0]
    ]
    past_key_values = output.past_key_values
    decode_latencies: list[float] = []
    for token_index in range(1, output_tokens):
        current_id = torch.tensor(
            [[selected_ids[-1]]],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones(
            (1, prompt_ids.numel() + token_index),
            dtype=torch.long,
            device=device,
        )
        torch.cuda.synchronize(device)
        decode_started = perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=current_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        decode_latencies.append(perf_counter() - decode_started)
        predicted = int(output.logits[0, -1].argmax())
        predicted_ids.append(predicted)
        selected_ids.append(
            forced_token_ids[token_index]
            if forced_token_ids is not None
            else predicted
        )
        past_key_values = output.past_key_values

    decode = summarize_decode_latencies(decode_latencies)
    total_seconds = prefill_seconds + float(decode["seconds"])
    result: dict[str, Any] = {
        "prompt_tokens": prompt_ids.numel(),
        "output_tokens": output_tokens,
        "time_to_first_token_seconds": prefill_seconds,
        "prefill_input_tokens_per_second": prompt_ids.numel() / prefill_seconds,
        "decode": decode,
        "end_to_end_seconds": total_seconds,
        "end_to_end_output_tokens_per_second": output_tokens / total_seconds,
        "predicted_token_ids": predicted_ids,
        "selected_token_ids": selected_ids,
        "selected_text": tokenizer.decode(selected_ids, skip_special_tokens=True),
        "forced_path_top1_agreement_fraction": (
            None
            if forced_token_ids is None
            else sum(
                predicted == forced
                for predicted, forced in zip(predicted_ids, forced_token_ids)
            )
            / output_tokens
        ),
    }
    if cache is not None and before_cache is not None and after_prefill_cache is not None:
        result["prefill_cache"] = _stats_delta(before_cache, after_prefill_cache)
        result["decode_cache"] = _stats_delta(
            after_prefill_cache,
            cache.snapshot(),
        )
    return result


def _load_paged(
    checkpoint_root: Path,
    *,
    device: torch.device,
    cache_pages: int,
) -> tuple[Any, Any, CudaExpertCache]:
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
        capacity_bytes=cache_pages * spec.expert_bytes(),
        num_experts=spec.num_experts,
    )
    model, load_summary = load_paged_qwen(
        checkpoint_root,
        cache,
        device=device,
    )
    return model, load_summary, cache


def run_b1_throughput(config: B1ThroughputConfig) -> dict[str, Any]:
    """Benchmark load, TTFT, steady decode, and end-to-end token rate."""

    if not torch.cuda.is_available():
        raise RuntimeError("B1 throughput requires CUDA or HIP")
    if config.mode not in {"both", "reference", "paged"}:
        raise ValueError("mode must be both, reference, or paged")
    if config.cache_pages <= 0 or config.output_tokens < 2:
        raise ValueError("cache_pages must be positive and output_tokens at least two")

    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers is required for B1 throughput") from error
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_root,
        local_files_only=True,
    )
    prompt_ids = _format_prompt(tokenizer, config.prompt)

    result: dict[str, Any] = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
    }
    forced_token_ids: list[int] | None = None
    if config.mode in {"both", "reference"}:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        reference_model, reference_load = load_reference_qwen(
            checkpoint_root,
            device=device,
        )
        reference_benchmark = _benchmark_path(
            reference_model,
            tokenizer,
            prompt_ids,
            device=device,
            output_tokens=config.output_tokens,
        )
        forced_token_ids = reference_benchmark["selected_token_ids"]
        result["reference"] = {
            "load": asdict(reference_load),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "benchmark": reference_benchmark,
        }
        del reference_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    if config.mode in {"both", "paged"}:
        torch.cuda.reset_peak_memory_stats(device)
        paged_model, paged_load, cache = _load_paged(
            checkpoint_root,
            device=device,
            cache_pages=config.cache_pages,
        )
        paged_benchmark = _benchmark_path(
            paged_model,
            tokenizer,
            prompt_ids,
            device=device,
            output_tokens=config.output_tokens,
            forced_token_ids=forced_token_ids,
            cache=cache,
        )
        result["paged"] = {
            "load": asdict(paged_load),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "expert_cache_capacity_bytes": cache.capacity_bytes,
            "benchmark": paged_benchmark,
            "final_cache_stats": asdict(cache.snapshot()),
        }
        cache.close()

    if "reference" in result and "paged" in result:
        reference_benchmark = result["reference"]["benchmark"]
        paged_benchmark = result["paged"]["benchmark"]
        result["slowdown"] = {
            "time_to_first_token_factor": (
                paged_benchmark["time_to_first_token_seconds"]
                / reference_benchmark["time_to_first_token_seconds"]
            ),
            "decode_latency_factor": (
                reference_benchmark["decode"]["tokens_per_second"]
                / paged_benchmark["decode"]["tokens_per_second"]
            ),
            "end_to_end_latency_factor": (
                paged_benchmark["end_to_end_seconds"]
                / reference_benchmark["end_to_end_seconds"]
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("both", "reference", "paged"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-pages", type=int, default=320)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    print(json.dumps(run_b1_throughput(B1ThroughputConfig(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
