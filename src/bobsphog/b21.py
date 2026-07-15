"""Run live future-aware retention and token-prefetch controls for B2.1."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from bobsphog.b1_throughput import (
    _format_prompt,
    _stats_delta,
    summarize_decode_latencies,
)
from bobsphog.cache_simulation import simulate_grouped_belady
from bobsphog.expert_cache import ExpertKey
from bobsphog.oracle_cache import OracleExpertCache, OraclePrefetchStats


DEFAULT_CAPACITIES = (1280, 2048, 2560)
DEFAULT_POLICIES = (
    "oracle_retention",
    "oracle_token_prefetch",
)
SUPPORTED_POLICIES = (
    *DEFAULT_POLICIES,
    "oracle_prompt_union",
)


@dataclass(frozen=True)
class B21Config:
    checkpoint: str
    trace: str
    device: str = "cuda:0"
    cache_pages: tuple[int, ...] = DEFAULT_CAPACITIES
    policies: tuple[str, ...] = DEFAULT_POLICIES


@dataclass(frozen=True)
class RecordedRoute:
    prompt: str
    selected_token_ids: tuple[int, ...]
    prefill_groups: tuple[tuple[ExpertKey, ...], ...]
    decode_traces: tuple[tuple[tuple[ExpertKey, ...], ...], ...]

    @property
    def groups(self) -> tuple[tuple[ExpertKey, ...], ...]:
        return self.prefill_groups + tuple(
            group for trace in self.decode_traces for group in trace
        )

    @property
    def num_layers(self) -> int:
        return len(self.prefill_groups)

    @property
    def decode_union_groups(self) -> tuple[tuple[ExpertKey, ...], ...]:
        by_layer: list[dict[ExpertKey, None]] = [
            {} for _ in range(self.num_layers)
        ]
        for trace in self.decode_traces:
            for layer, group in enumerate(trace):
                by_layer[layer].update((key, None) for key in group)
        return tuple(tuple(keys) for keys in by_layer)


def _deserialize_layers(layers: list[list[int]]) -> tuple[tuple[ExpertKey, ...], ...]:
    return tuple(
        tuple((layer_index, int(expert)) for expert in experts)
        for layer_index, experts in enumerate(layers)
    )


def load_recorded_route(path: str | Path) -> RecordedRoute:
    """Load the fixed B2 prompt, token path, and per-layer expert trace."""

    payload = json.loads(Path(path).expanduser().read_text())
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError("B2 trace contains no benchmark runs")
    routing = payload["routing_trace"]
    prefill = _deserialize_layers(routing["prefill_experts_by_layer"])
    decode = tuple(
        _deserialize_layers(token_layers)
        for token_layers in routing["decode_experts_by_token_and_layer"]
    )
    selected = tuple(int(token) for token in runs[0]["selected_token_ids"])
    if len(decode) != len(selected) - 1:
        raise ValueError("decode trace count must equal selected token count minus one")
    if any(len(trace) != len(prefill) for trace in decode):
        raise ValueError("every decode trace must contain one group per layer")
    if not selected or not prefill:
        raise ValueError("recorded route must contain tokens and prefill groups")
    return RecordedRoute(
        prompt=str(payload["config"]["prompt"]),
        selected_token_ids=selected,
        prefill_groups=prefill,
        decode_traces=decode,
    )


def summarize_prefetch_pipeline(
    prefetch_seconds: list[float],
    compute_seconds: list[float],
) -> dict[str, float | int]:
    """Report serial, compute-only, and ideal one-token-overlap rates."""

    if not compute_seconds or len(prefetch_seconds) != len(compute_seconds):
        raise ValueError("prefetch and compute samples must be non-empty and aligned")
    if any(value < 0 for value in prefetch_seconds) or any(
        value <= 0 for value in compute_seconds
    ):
        raise ValueError("prefetch must be non-negative and compute must be positive")
    serial_seconds = sum(prefetch_seconds) + sum(compute_seconds)
    compute_only_seconds = sum(compute_seconds)
    ideal_overlap_seconds = prefetch_seconds[0]
    ideal_overlap_seconds += sum(
        max(compute_seconds[index], prefetch_seconds[index + 1])
        for index in range(len(compute_seconds) - 1)
    )
    ideal_overlap_seconds += compute_seconds[-1]
    tokens = len(compute_seconds)
    return {
        "decode_forwards": tokens,
        "serial_seconds": serial_seconds,
        "serial_tokens_per_second": tokens / serial_seconds,
        "compute_only_seconds": compute_only_seconds,
        "compute_only_tokens_per_second": tokens / compute_only_seconds,
        "ideal_one_token_overlap_seconds": ideal_overlap_seconds,
        "ideal_one_token_overlap_tokens_per_second": tokens / ideal_overlap_seconds,
        "prefetch_seconds": sum(prefetch_seconds),
    }


def _prefetch_delta(
    before: OraclePrefetchStats,
    after: OraclePrefetchStats,
) -> dict[str, int | float]:
    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in asdict(before)
    }


def _load_oracle_model(
    checkpoint_root: Path,
    route: RecordedRoute,
    *,
    device: torch.device,
    cache_pages: int,
) -> tuple[Any, Any, OracleExpertCache, Any]:
    from bobsphog.moe_checkpoint import (
        MappedExpertSource,
        QwenMoeSpec,
        SafetensorCheckpointIndex,
    )
    from bobsphog.moe_model import load_paged_qwen

    spec = QwenMoeSpec.from_files(
        checkpoint_root / "config.json",
        checkpoint_root / "model.safetensors.index.json",
    )
    if spec.num_layers != route.num_layers:
        raise ValueError("checkpoint layer count does not match recorded route")
    index = SafetensorCheckpointIndex(
        checkpoint_root / "model.safetensors.index.json"
    )
    source = MappedExpertSource(index, spec)
    cache = OracleExpertCache(
        source,
        device=device,
        capacity_bytes=cache_pages * spec.expert_bytes(),
        num_experts=spec.num_experts,
        oracle_groups=route.groups,
        expert_page_bytes=spec.expert_bytes(),
    )
    model, load_summary = load_paged_qwen(
        checkpoint_root,
        cache,
        device=device,
    )
    return model, load_summary, cache, spec


def _run_control(
    checkpoint_root: Path,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    route: RecordedRoute,
    *,
    device: torch.device,
    cache_pages: int,
    policy: str,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = perf_counter()
    model, load_summary, cache, spec = _load_oracle_model(
        checkpoint_root,
        route,
        device=device,
        cache_pages=cache_pages,
    )
    total_load_seconds = perf_counter() - load_started

    prompt_prefetch_seconds = 0.0
    if policy == "oracle_prompt_union":
        decode_union_groups = route.decode_union_groups
        decode_union = tuple(key for group in decode_union_groups for key in group)
        largest_prefill_group = max(len(group) for group in route.prefill_groups)
        if len(decode_union) + largest_prefill_group > cache_pages:
            raise ValueError(
                "prompt-union policy requires capacity for the decode union plus "
                "the largest prefill layer"
            )
        cache.set_pinned_keys(decode_union)
        torch.cuda.synchronize(device)
        prompt_prefetch_started = perf_counter()
        cache.prefetch_groups(decode_union_groups)
        torch.cuda.synchronize(device)
        prompt_prefetch_seconds = perf_counter() - prompt_prefetch_started

    input_ids = prompt_ids.unsqueeze(0).to(device)
    before_prefill = cache.snapshot()
    torch.cuda.synchronize(device)
    prefill_started = perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(device)
    prefill_seconds = perf_counter() - prefill_started
    after_prefill = cache.snapshot()
    if cache.oracle.cursor != route.num_layers:
        raise RuntimeError("prefill did not consume exactly one route group per layer")

    predicted_ids = [int(output.logits[0, -1].argmax())]
    selected_ids = [route.selected_token_ids[0]]
    past_key_values = output.past_key_values
    compute_latencies: list[float] = []
    prefetch_latencies: list[float] = []
    steps: list[dict[str, Any]] = []

    for token_index, token_trace in enumerate(route.decode_traces, start=1):
        before_step = cache.snapshot()
        before_prefetch = cache.prefetch_stats.snapshot()
        prefetch_seconds = 0.0
        if policy == "oracle_token_prefetch":
            torch.cuda.synchronize(device)
            prefetch_started = perf_counter()
            cache.prefetch_groups(token_trace)
            torch.cuda.synchronize(device)
            prefetch_seconds = perf_counter() - prefetch_started
        after_prefetch_cache = cache.snapshot()
        after_prefetch = cache.prefetch_stats.snapshot()

        current_id = torch.tensor(
            [[selected_ids[-1]]], dtype=torch.long, device=device
        )
        attention_mask = torch.ones(
            (1, prompt_ids.numel() + token_index),
            dtype=torch.long,
            device=device,
        )
        torch.cuda.synchronize(device)
        compute_started = perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=current_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        compute_seconds = perf_counter() - compute_started
        after_step = cache.snapshot()
        predicted = int(output.logits[0, -1].argmax())
        predicted_ids.append(predicted)
        selected_ids.append(route.selected_token_ids[token_index])
        past_key_values = output.past_key_values
        prefetch_latencies.append(prefetch_seconds)
        compute_latencies.append(compute_seconds)
        steps.append(
            {
                "output_token_index": token_index,
                "prefetch_seconds": prefetch_seconds,
                "compute_seconds": compute_seconds,
                "prefetch": _prefetch_delta(before_prefetch, after_prefetch),
                "demand_cache": _stats_delta(after_prefetch_cache, after_step),
                "physical_cache": _stats_delta(before_step, after_step),
            }
        )

    if not cache.oracle.complete:
        raise RuntimeError(
            f"oracle trace incomplete: consumed {cache.oracle.cursor} of "
            f"{len(cache.oracle.groups)} groups"
        )
    final_stats = cache.snapshot()
    pipeline = summarize_prefetch_pipeline(prefetch_latencies, compute_latencies)
    compute = summarize_decode_latencies(compute_latencies)
    decode_physical = _stats_delta(after_prefill, final_stats)
    expected = simulate_grouped_belady(route.groups, cache_pages)
    prefill_expected = simulate_grouped_belady(route.prefill_groups, cache_pages)
    expected_decode_misses = expected.misses - prefill_expected.misses
    time_to_first_token = prompt_prefetch_seconds + prefill_seconds
    serial_end_to_end = time_to_first_token + float(pipeline["serial_seconds"])
    ideal_end_to_end = time_to_first_token + float(
        pipeline["ideal_one_token_overlap_seconds"]
    )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    result = {
        "policy": policy,
        "cache_pages": cache_pages,
        "cache_capacity_bytes": cache.capacity_bytes,
        "expert_page_bytes": spec.expert_bytes(),
        "load": {
            **asdict(load_summary),
            "total_model_construction_seconds": total_load_seconds,
        },
        "peak_allocated_bytes": peak_allocated,
        "peak_fraction_of_checkpoint": peak_allocated / spec.checkpoint_bytes,
        "prompt_tokens": prompt_ids.numel(),
        "output_tokens": len(route.selected_token_ids),
        "prompt_prefetch_seconds": prompt_prefetch_seconds,
        "prefill_forward_seconds": prefill_seconds,
        "time_to_first_token_seconds": time_to_first_token,
        "compute": compute,
        "pipeline": pipeline,
        "serial_end_to_end_seconds": serial_end_to_end,
        "serial_end_to_end_output_tokens_per_second": (
            len(route.selected_token_ids) / serial_end_to_end
        ),
        "ideal_overlap_end_to_end_seconds": ideal_end_to_end,
        "ideal_overlap_end_to_end_output_tokens_per_second": (
            len(route.selected_token_ids) / ideal_end_to_end
        ),
        "predicted_token_ids": predicted_ids,
        "selected_token_ids": selected_ids,
        "selected_text": tokenizer.decode(selected_ids, skip_special_tokens=True),
        "forced_path_top1_agreement_fraction": sum(
            predicted == selected
            for predicted, selected in zip(predicted_ids, selected_ids)
        )
        / len(selected_ids),
        "prefill_cache": _stats_delta(before_prefill, after_prefill),
        "decode_physical_cache": decode_physical,
        "prefetch_stats": asdict(cache.prefetch_stats.snapshot()),
        "expected_oracle_decode_misses": expected_decode_misses,
        "demand_misses_match_offline_optimal": (
            policy == "oracle_retention"
            and decode_physical["misses"] == expected_decode_misses
        ),
        "steps": steps,
        "final_cache_stats": asdict(final_stats),
        "oracle_trace_complete": cache.oracle.complete,
    }
    cache.close()
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return result


def run_b21(config: B21Config) -> dict[str, Any]:
    """Sweep live oracle retention and token-prefetch controls."""

    if not torch.cuda.is_available():
        raise RuntimeError("B2.1 requires CUDA or HIP")
    if not config.cache_pages or any(value < 320 for value in config.cache_pages):
        raise ValueError("every cache capacity must fit a 320-page token bundle")
    if len(set(config.cache_pages)) != len(config.cache_pages):
        raise ValueError("cache capacities must be unique")
    if not config.policies or any(
        policy not in SUPPORTED_POLICIES for policy in config.policies
    ):
        raise ValueError(f"policies must be selected from {SUPPORTED_POLICIES!r}")

    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    route = load_recorded_route(config.trace)
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers is required for B2.1") from error
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_root, local_files_only=True)
    prompt_ids = _format_prompt(tokenizer, route.prompt)

    result: dict[str, Any] = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "trace": {
            "prompt": route.prompt,
            "layers": route.num_layers,
            "output_tokens": len(route.selected_token_ids),
            "decode_forwards": len(route.decode_traces),
            "request_groups": len(route.groups),
        },
        "runs": [],
    }
    for capacity in config.cache_pages:
        for policy in config.policies:
            result["runs"].append(
                _run_control(
                    checkpoint_root,
                    tokenizer,
                    prompt_ids,
                    route,
                    device=device,
                    cache_pages=capacity,
                    policy=policy,
                )
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cache-pages", type=int, nargs="+", default=list(DEFAULT_CAPACITIES)
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=SUPPORTED_POLICIES,
        default=list(DEFAULT_POLICIES),
    )
    args = parser.parse_args()
    args.cache_pages = tuple(args.cache_pages)
    args.policies = tuple(args.policies)
    print(json.dumps(run_b21(B21Config(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
