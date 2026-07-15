"""Measure Qwen expert working-set locality across cache capacities."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable

import torch

from bobsphog.b1_throughput import (
    DEFAULT_PROMPT,
    _format_prompt,
    _stats_delta,
    summarize_decode_latencies,
)
from bobsphog.cache_simulation import (
    simulate_grouped_belady,
    simulate_grouped_lru,
)
from bobsphog.expert_cache import CudaExpertCache, ExpertKey


DEFAULT_CACHE_PAGES = (2048, 1280, 640, 320)


@dataclass(frozen=True)
class B2Config:
    checkpoint: str
    device: str = "cuda:0"
    cache_pages: tuple[int, ...] = DEFAULT_CACHE_PAGES
    output_tokens: int = 16
    prompt: str = DEFAULT_PROMPT


class TracingCudaExpertCache(CudaExpertCache):
    """Record atomic layer request groups before normal cache execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_trace: list[tuple[ExpertKey, ...]] = []

    def schedule(self, keys: Iterable[ExpertKey]) -> tuple[ExpertKey, ...]:
        requested = self._unique(keys)
        self._request_trace.append(requested)
        return super().schedule(requested)

    def drain_request_trace(self) -> tuple[tuple[ExpertKey, ...], ...]:
        trace = tuple(self._request_trace)
        self._request_trace.clear()
        return trace


def _distribution(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "samples": len(values),
        "mean": mean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _forward_pages(
    trace: tuple[tuple[ExpertKey, ...], ...],
    num_layers: int,
) -> tuple[set[ExpertKey], list[set[ExpertKey]]]:
    if len(trace) != num_layers:
        raise RuntimeError(
            f"expected {num_layers} layer requests, observed {len(trace)}"
        )
    layers = [set(group) for group in trace]
    return set().union(*layers), layers


def analyze_decode_working_sets(
    decode_traces: list[tuple[tuple[ExpertKey, ...], ...]],
    num_layers: int,
) -> dict[str, Any]:
    """Summarize union growth and previous-token routing predictability."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    forwards = [_forward_pages(trace, num_layers) for trace in decode_traces]
    cumulative: set[ExpertKey] = set()
    token_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    layer_recalls: list[list[float]] = [[] for _ in range(num_layers)]
    layer_jaccards: list[list[float]] = [[] for _ in range(num_layers)]

    for token_index, (pages, layers) in enumerate(forwards):
        cumulative.update(pages)
        token_rows.append(
            {
                "decode_forward_index": token_index,
                "working_set_pages": len(pages),
                "cumulative_unique_pages": len(cumulative),
                "new_pages": (
                    len(pages)
                    if token_index == 0
                    else len(pages - forwards[token_index - 1][0])
                ),
            }
        )
        if token_index == 0:
            continue
        previous_pages, previous_layers = forwards[token_index - 1]
        overlap = pages & previous_pages
        union = pages | previous_pages
        recall = len(overlap) / len(pages) if pages else 1.0
        precision = len(overlap) / len(previous_pages) if previous_pages else 1.0
        transition_rows.append(
            {
                "to_decode_forward_index": token_index,
                "overlap_pages": len(overlap),
                "current_working_set_recall_from_previous": recall,
                "previous_working_set_precision_for_current": precision,
                "jaccard": len(overlap) / len(union) if union else 1.0,
            }
        )
        for layer in range(num_layers):
            current = layers[layer]
            previous = previous_layers[layer]
            layer_overlap = current & previous
            layer_union = current | previous
            layer_recalls[layer].append(
                len(layer_overlap) / len(current) if current else 1.0
            )
            layer_jaccards[layer].append(
                len(layer_overlap) / len(layer_union) if layer_union else 1.0
            )

    return {
        "decode_forwards": len(forwards),
        "final_cumulative_unique_pages": len(cumulative),
        "tokens": token_rows,
        "transitions": transition_rows,
        "previous_token_predictor": {
            "working_set_recall": _distribution(
                [row["current_working_set_recall_from_previous"] for row in transition_rows]
            ),
            "working_set_precision": _distribution(
                [row["previous_working_set_precision_for_current"] for row in transition_rows]
            ),
            "jaccard": _distribution([row["jaccard"] for row in transition_rows]),
        },
        "per_layer": [
            {
                "layer": layer,
                "previous_token_recall": _distribution(layer_recalls[layer]),
                "previous_token_jaccard": _distribution(layer_jaccards[layer]),
            }
            for layer in range(num_layers)
        ],
    }


def _serialize_trace(
    trace: tuple[tuple[ExpertKey, ...], ...],
) -> list[list[int]]:
    serialized: list[list[int]] = []
    for layer_index, group in enumerate(trace):
        if any(layer != layer_index for layer, _ in group):
            raise RuntimeError("trace group does not match its layer index")
        serialized.append([expert for _, expert in group])
    return serialized


def _load_traced_paged(
    checkpoint_root: Path,
    *,
    device: torch.device,
    cache_pages: int,
) -> tuple[Any, Any, TracingCudaExpertCache, Any]:
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
    index = SafetensorCheckpointIndex(
        checkpoint_root / "model.safetensors.index.json"
    )
    source = MappedExpertSource(index, spec)
    cache = TracingCudaExpertCache(
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
    return model, load_summary, cache, spec


def _run_capacity(
    checkpoint_root: Path,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    *,
    device: torch.device,
    cache_pages: int,
    output_tokens: int,
    forced_token_ids: list[int] | None,
) -> tuple[
    dict[str, Any],
    list[tuple[ExpertKey, ...]],
    list[tuple[tuple[ExpertKey, ...], ...]],
]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = perf_counter()
    model, load_summary, cache, spec = _load_traced_paged(
        checkpoint_root,
        device=device,
        cache_pages=cache_pages,
    )
    total_load_seconds = perf_counter() - load_started
    if forced_token_ids is not None and len(forced_token_ids) != output_tokens:
        raise ValueError("forced_token_ids length must equal output_tokens")

    input_ids = prompt_ids.unsqueeze(0).to(device)
    before_prefill = cache.snapshot()
    torch.cuda.synchronize(device)
    prefill_started = perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(device)
    prefill_seconds = perf_counter() - prefill_started
    prefill_trace = cache.drain_request_trace()
    _forward_pages(prefill_trace, spec.num_layers)
    after_prefill = cache.snapshot()

    predicted_ids = [int(output.logits[0, -1].argmax())]
    selected_ids = [
        forced_token_ids[0] if forced_token_ids is not None else predicted_ids[0]
    ]
    past_key_values = output.past_key_values
    decode_latencies: list[float] = []
    decode_traces: list[tuple[tuple[ExpertKey, ...], ...]] = []
    decode_steps: list[dict[str, Any]] = []
    for token_index in range(1, output_tokens):
        current_id = torch.tensor(
            [[selected_ids[-1]]], dtype=torch.long, device=device
        )
        attention_mask = torch.ones(
            (1, prompt_ids.numel() + token_index),
            dtype=torch.long,
            device=device,
        )
        before_step = cache.snapshot()
        torch.cuda.synchronize(device)
        step_started = perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=current_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        latency = perf_counter() - step_started
        trace = cache.drain_request_trace()
        pages, _ = _forward_pages(trace, spec.num_layers)
        step_delta = _stats_delta(before_step, cache.snapshot())
        decode_latencies.append(latency)
        decode_traces.append(trace)
        decode_steps.append(
            {
                "output_token_index": token_index,
                "seconds": latency,
                "working_set_pages": len(pages),
                "cache": step_delta,
            }
        )
        predicted = int(output.logits[0, -1].argmax())
        predicted_ids.append(predicted)
        selected_ids.append(
            forced_token_ids[token_index]
            if forced_token_ids is not None
            else predicted
        )
        past_key_values = output.past_key_values

    decode = summarize_decode_latencies(decode_latencies)
    final_stats = cache.snapshot()
    all_groups = list(prefill_trace)
    for trace in decode_traces:
        all_groups.extend(trace)
    workset = analyze_decode_working_sets(decode_traces, spec.num_layers)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    result = {
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
        "output_tokens": output_tokens,
        "time_to_first_token_seconds": prefill_seconds,
        "decode": decode,
        "end_to_end_seconds": prefill_seconds + float(decode["seconds"]),
        "end_to_end_output_tokens_per_second": (
            output_tokens / (prefill_seconds + float(decode["seconds"]))
        ),
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
        "prefill_cache": _stats_delta(before_prefill, after_prefill),
        "decode_cache": _stats_delta(after_prefill, final_stats),
        "decode_steps": decode_steps,
        "working_set": workset,
        "final_cache_stats": asdict(final_stats),
    }
    cache.close()
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return result, all_groups, decode_traces


def run_b2(config: B2Config) -> dict[str, Any]:
    """Run the real capacity sweep and trace-driven replacement controls."""

    if not torch.cuda.is_available():
        raise RuntimeError("B2 requires CUDA or HIP")
    if config.output_tokens < 2:
        raise ValueError("output_tokens must be at least two")
    if not config.cache_pages or any(pages <= 0 for pages in config.cache_pages):
        raise ValueError("cache_pages must contain positive capacities")
    if len(set(config.cache_pages)) != len(config.cache_pages):
        raise ValueError("cache_pages must not contain duplicates")

    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers is required for B2") from error
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_root, local_files_only=True)
    prompt_ids = _format_prompt(tokenizer, config.prompt)

    result: dict[str, Any] = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "capacity_order_note": (
            "Capacities run in the supplied order. The default is descending so later "
            "smaller-cache timings receive any OS-page-cache warming; this is "
            "conservative when judging larger caches."
        ),
        "runs": [],
    }
    forced_token_ids: list[int] | None = None
    reference_groups: list[tuple[ExpertKey, ...]] | None = None
    reference_decode_traces: list[tuple[tuple[ExpertKey, ...], ...]] | None = None
    for capacity in config.cache_pages:
        run, groups, decode_traces = _run_capacity(
            checkpoint_root,
            tokenizer,
            prompt_ids,
            device=device,
            cache_pages=capacity,
            output_tokens=config.output_tokens,
            forced_token_ids=forced_token_ids,
        )
        if forced_token_ids is None:
            forced_token_ids = list(run["selected_token_ids"])
            reference_groups = groups
            reference_decode_traces = decode_traces
            run["route_trace_matches_reference"] = True
        else:
            run["route_trace_matches_reference"] = groups == reference_groups
        result["runs"].append(run)

    if reference_groups is None or reference_decode_traces is None:
        raise RuntimeError("capacity sweep produced no trace")
    page_bytes = int(result["runs"][0]["expert_page_bytes"])
    prefill_group_count = len(reference_decode_traces[0])
    prefill_groups = reference_groups[:prefill_group_count]
    for run in result["runs"]:
        capacity = int(run["cache_pages"])
        lru = simulate_grouped_lru(reference_groups, capacity)
        optimal = simulate_grouped_belady(reference_groups, capacity)
        prefill_lru = simulate_grouped_lru(prefill_groups, capacity)
        prefill_optimal = simulate_grouped_belady(prefill_groups, capacity)
        decode_lru_misses = lru.misses - prefill_lru.misses
        decode_optimal_misses = optimal.misses - prefill_optimal.misses
        run["cache_policy_simulation"] = {
            "lru": lru.describe(page_bytes),
            "offline_optimal": optimal.describe(page_bytes),
            "lru_excess_misses_over_optimal": lru.misses - optimal.misses,
            "optimal_miss_reduction_fraction": (
                (lru.misses - optimal.misses) / lru.misses if lru.misses else 0.0
            ),
            "decode_phase": {
                "lru_misses": decode_lru_misses,
                "offline_optimal_misses": decode_optimal_misses,
                "lru_bytes_transferred": decode_lru_misses * page_bytes,
                "offline_optimal_bytes_transferred": (
                    decode_optimal_misses * page_bytes
                ),
                "optimal_miss_reduction_fraction": (
                    (decode_lru_misses - decode_optimal_misses) / decode_lru_misses
                    if decode_lru_misses
                    else 0.0
                ),
            },
            "measured_lru_misses_match_trace_simulation": (
                run["final_cache_stats"]["misses"] == lru.misses
            ),
        }

    result["routing_trace"] = {
        "prefill_experts_by_layer": _serialize_trace(
            tuple(prefill_groups)
        ),
        "decode_experts_by_token_and_layer": [
            _serialize_trace(trace) for trace in reference_decode_traces
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cache-pages", type=int, nargs="+", default=list(DEFAULT_CACHE_PAGES)
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    args.cache_pages = tuple(args.cache_pages)
    print(json.dumps(run_b2(B2Config(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
