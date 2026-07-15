"""Benchmark physical CPU-to-CUDA page residency, cache reuse, and churn."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

import torch

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.physical_cache import CacheStats, PhysicalPageCache


@dataclass(frozen=True)
class A5Config:
    seed: int = 53
    device: str = "cuda:0"
    dtype: str = "float16"
    vocab_size: int = 256
    context_length: int = 64
    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 2
    d_ff: int = 8192
    base_rank: int = 128
    page_rank: int = 128
    pages_per_matrix: int = 15
    working_set_pages: int = 8
    timing_iterations: int = 10


def _dtype(name: str) -> torch.dtype:
    choices = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in choices:
        raise ValueError(f"unsupported dtype: {name!r}")
    return choices[name]


def _elapsed_ms(operation: Callable[[], Any], device: torch.device, iterations: int) -> float:
    for _ in range(2):
        operation()
    torch.cuda.synchronize(device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(iterations):
        operation()
    finished.record()
    finished.synchronize()
    return started.elapsed_time(finished) / iterations


def _stats_delta(before: CacheStats, after: CacheStats) -> dict[str, float | int]:
    return {
        field: getattr(after, field) - getattr(before, field)
        for field in asdict(before)
    }


def run_a5(config: A5Config) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("A5 requires CUDA")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dtype = _dtype(config.dtype)
    torch.cuda.set_device(device)
    model_config = ToyConfig(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=0.0,
        base_rank=config.base_rank,
        page_rank=config.page_rank,
        factorized_page_count=config.pages_per_matrix,
    )
    model = ToyTransformer(model_config).eval().requires_grad_(False).to(device=device, dtype=dtype)
    catalog = PageCatalog.from_model(model)
    if config.working_set_pages * 2 > len(catalog):
        raise ValueError("benchmark needs two disjoint working sets")
    primary_ids = catalog.static_prefix(config.working_set_pages)
    secondary_ids = tuple(range(len(catalog) - config.working_set_pages, len(catalog)))
    primary_plan = catalog.plan(primary_ids)
    secondary_plan = catalog.plan(secondary_ids)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, config.context_length),
        device=device,
    )

    with torch.inference_mode():
        reference = model(input_ids, plan=primary_plan).logits
        full_forward_ms = _elapsed_ms(
            lambda: model(input_ids, plan=primary_plan).logits,
            device,
            config.timing_iterations,
        )
    reference_cpu = reference.float().cpu()
    del reference
    torch.cuda.synchronize(device)
    full_resident_allocated = torch.cuda.memory_allocated(device)

    selected_page_bytes = catalog.selected_bytes(primary_ids)
    cache = PhysicalPageCache(
        model,
        device=device,
        capacity_bytes=selected_page_bytes,
        dtype=dtype,
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    skeleton_allocated = torch.cuda.memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    before_cold = cache.snapshot()
    cold_started = perf_counter()
    cache.prepare(primary_plan)
    cold_prepare_ms = (perf_counter() - cold_started) * 1000
    after_cold = cache.snapshot()
    after_prefetch_allocated = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        physical_output = model(input_ids, plan=primary_plan).logits
    torch.cuda.synchronize(device)
    physical_peak_allocated = torch.cuda.max_memory_allocated(device)
    max_absolute_error = (physical_output.float().cpu() - reference_cpu).abs().max().item()
    del physical_output

    before_warm = cache.snapshot()
    cache.prepare(primary_plan)
    after_warm = cache.snapshot()
    with torch.inference_mode():
        warm_forward_ms = _elapsed_ms(
            lambda: model(input_ids, plan=primary_plan).logits,
            device,
            config.timing_iterations,
        )

    before_churn = cache.snapshot()
    cache.prepare(secondary_plan)
    after_churn = cache.snapshot()
    alternate_allocated = torch.cuda.memory_allocated(device)
    cache.prepare(primary_plan)
    after_return = cache.snapshot()

    cache.prepare(secondary_plan)
    before_synchronous_cycle = cache.snapshot()
    synchronous_started = perf_counter()
    cache.prepare(primary_plan)
    with torch.inference_mode():
        synchronous_output = model(input_ids, plan=primary_plan).logits
    torch.cuda.synchronize(device)
    synchronous_cycle_ms = (perf_counter() - synchronous_started) * 1000
    after_synchronous_cycle = cache.snapshot()
    synchronous_error = (
        synchronous_output.float().cpu() - reference_cpu
    ).abs().max().item()
    del synchronous_output

    cache.prepare(secondary_plan)
    before_overlapped_cycle = cache.snapshot()
    overlapped_started = perf_counter()
    cache.schedule(primary_plan)
    with torch.inference_mode():
        overlapped_output = model(input_ids, plan=primary_plan).logits
    torch.cuda.synchronize(device)
    overlapped_cycle_ms = (perf_counter() - overlapped_started) * 1000
    after_overlapped_cycle = cache.snapshot()
    overlapped_error = (
        overlapped_output.float().cpu() - reference_cpu
    ).abs().max().item()
    del overlapped_output

    result = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "page_count": len(catalog),
        "source_page_bytes": cache.source_bytes,
        "cache_capacity_bytes": cache.capacity_bytes,
        "selected_page_bytes": selected_page_bytes,
        "full_resident_allocated_bytes": full_resident_allocated,
        "skeleton_allocated_bytes": skeleton_allocated,
        "after_prefetch_allocated_bytes": after_prefetch_allocated,
        "physical_peak_allocated_bytes": physical_peak_allocated,
        "alternate_working_set_allocated_bytes": alternate_allocated,
        "allocated_savings_vs_full_bytes": full_resident_allocated
        - after_prefetch_allocated,
        "allocated_savings_fraction": 1
        - after_prefetch_allocated / full_resident_allocated,
        "full_resident_forward_ms": full_forward_ms,
        "warm_cached_forward_ms": warm_forward_ms,
        "cold_prepare_wall_ms": cold_prepare_ms,
        "max_absolute_logit_error": max_absolute_error,
        "cold_prepare": _stats_delta(before_cold, after_cold),
        "warm_prepare": _stats_delta(before_warm, after_warm),
        "switch_working_set": _stats_delta(before_churn, after_churn),
        "return_working_set": _stats_delta(after_churn, after_return),
        "synchronous_cold_cycle_ms": synchronous_cycle_ms,
        "overlapped_cold_cycle_ms": overlapped_cycle_ms,
        "overlap_saved_ms": synchronous_cycle_ms - overlapped_cycle_ms,
        "overlap_saved_fraction": 1 - overlapped_cycle_ms / synchronous_cycle_ms,
        "synchronous_cycle_error": synchronous_error,
        "overlapped_cycle_error": overlapped_error,
        "synchronous_cold_cycle": _stats_delta(
            before_synchronous_cycle,
            after_synchronous_cycle,
        ),
        "overlapped_cold_cycle": _stats_delta(
            before_overlapped_cycle,
            after_overlapped_cycle,
        ),
        "final_cache_stats": asdict(cache.snapshot()),
    }
    cache.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--working-set-pages", type=int, default=8)
    parser.add_argument("--timing-iterations", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            run_a5(
                A5Config(
                    device=args.device,
                    dtype=args.dtype,
                    working_set_pages=args.working_set_pages,
                    timing_iterations=args.timing_iterations,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
