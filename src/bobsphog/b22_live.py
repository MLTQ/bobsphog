"""Run a held-out B2.2 predicted bundle through the exact paged Qwen model."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from bobsphog.b1_throughput import _format_prompt, _stats_delta, summarize_decode_latencies
from bobsphog.b21 import RecordedRoute, _deserialize_layers, load_recorded_route
from bobsphog.b22_predict import (
    PredictorSuite,
    RouteExample,
    evaluate_selection,
    load_trace_examples,
    select_equal_layer_budget,
    simulate_pinned_bundle_lru,
    tune_hyperparameters,
)
from bobsphog.expert_cache import ExpertKey
from bobsphog.predictive_cache import TracingPinnedCudaExpertCache


@dataclass(frozen=True)
class B22LiveConfig:
    checkpoint: str
    corpus: str
    prompt_id: str
    device: str = "cuda:0"
    method: str = "nearest_neighbor"
    bundle_pages: int = 2048
    cache_pages: int = 2560
    page_bytes: int = 6_291_456
    seed: int = 72
    execution_trace: str | None = None
    prefetch_phase: str = "before_prefill"


def load_live_case(
    corpus: str | Path,
    prompt_id: str,
) -> tuple[dict[str, Any], RouteExample, RecordedRoute]:
    payload = json.loads(Path(corpus).expanduser().read_text())
    records = [record for record in payload.get("prompts", []) if record["id"] == prompt_id]
    if len(records) != 1:
        raise ValueError(f"expected exactly one corpus record for {prompt_id!r}")
    record = records[0]
    examples = load_trace_examples(corpus)
    example = next(item for item in examples if item.id == prompt_id)
    prefill = _deserialize_layers(record["prefill_experts_by_layer"])
    decode = tuple(
        _deserialize_layers(token_layers)
        for token_layers in record["decode_experts_by_token_and_layer"]
    )
    route = RecordedRoute(
        prompt=str(record["prompt"]),
        selected_token_ids=tuple(int(token) for token in record["selected_token_ids"]),
        prefill_groups=prefill,
        decode_traces=decode,
    )
    if example.split != "test":
        raise ValueError("live predictor validation requires a held-out test prompt")
    return record, example, route


def select_predicted_bundle(
    examples: list[RouteExample],
    example: RouteExample,
    *,
    method: str,
    budget: int,
    page_bytes: int,
    cache_pages: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    training = [item for item in examples if item.split == "train"]
    validation = [item for item in examples if item.split == "validation"]
    suite = PredictorSuite(training, seed=seed)
    tuned = tune_hyperparameters(
        suite,
        validation,
        budget=min(budget, 2048),
        page_bytes=page_bytes,
    )
    scores = suite.scores(
        method,
        example,
        alpha=float(tuned["conditional_alpha"]),
        neighbors=int(tuned["nearest_neighbors"]),
    )
    selected = select_equal_layer_budget(scores, budget)
    metrics = {
        "method": method,
        "tuned": tuned,
        **evaluate_selection(example, selected, page_bytes=page_bytes),
        **simulate_pinned_bundle_lru(
            example,
            selected,
            cache_capacity_pages=cache_pages,
            page_bytes=page_bytes,
        ),
    }
    return selected, metrics


def route_to_example(route: RecordedRoute, template: RouteExample) -> RouteExample:
    """Convert a backend-native recorded route into predictor evaluation tensors."""

    layers, experts = template.prefill.shape
    if route.num_layers != layers:
        raise ValueError("execution route layer count differs from predictor corpus")
    prefill = np.zeros((layers, experts), dtype=np.float32)
    decode_counts = np.zeros((layers, experts), dtype=np.float32)
    for layer, group in enumerate(route.prefill_groups):
        prefill[layer, [expert for _, expert in group]] = 1.0
    decode_groups = []
    for trace in route.decode_traces:
        token_groups = []
        for layer, group in enumerate(trace):
            expert_ids = tuple(expert for _, expert in group)
            decode_counts[layer, expert_ids] += 1.0
            token_groups.append(expert_ids)
        decode_groups.append(tuple(token_groups))
    return RouteExample(
        id=template.id,
        domain=template.domain,
        split=template.split,
        prefill=prefill,
        decode_counts=decode_counts,
        decode_groups=tuple(decode_groups),
    )


def _assert_route(
    observed: tuple[tuple[ExpertKey, ...], ...],
    expected: tuple[tuple[ExpertKey, ...], ...],
    label: str,
) -> None:
    if observed != expected:
        if len(observed) != len(expected):
            detail = f"group count expected={len(expected)} observed={len(observed)}"
        else:
            layer = next(
                index
                for index, (actual, reference) in enumerate(zip(observed, expected))
                if actual != reference
            )
            actual = observed[layer]
            reference = expected[layer]
            detail = (
                f"layer={layer} expected_count={len(reference)} "
                f"observed_count={len(actual)} "
                f"missing={sorted(set(reference) - set(actual))[:16]} "
                f"extra={sorted(set(actual) - set(reference))[:16]} "
                f"set_equal={set(actual) == set(reference)}"
            )
        raise RuntimeError(
            f"live route diverged from recorded route during {label}: {detail}"
        )


def run_b22_live(config: B22LiveConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("B2.2 live validation requires CUDA or HIP")
    if not 0 < config.bundle_pages < config.cache_pages:
        raise ValueError("bundle pages must be positive and smaller than cache pages")
    if config.prefetch_phase not in {"before_prefill", "after_prefill"}:
        raise ValueError("prefetch phase must be before_prefill or after_prefill")
    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    record, example, corpus_route = load_live_case(config.corpus, config.prompt_id)
    route = (
        load_recorded_route(config.execution_trace)
        if config.execution_trace is not None
        else corpus_route
    )
    if route.prompt != corpus_route.prompt:
        raise ValueError("execution trace prompt differs from predictor corpus prompt")
    examples = load_trace_examples(config.corpus)
    selected, selection_metrics = select_predicted_bundle(
        examples,
        example,
        method=config.method,
        budget=config.bundle_pages,
        page_bytes=config.page_bytes,
        cache_pages=config.cache_pages,
        seed=config.seed,
    )
    if config.execution_trace is not None:
        execution_example = route_to_example(route, example)
        selection_metrics = {
            "corpus_backend_target": selection_metrics,
            "execution_backend_target": {
                **evaluate_selection(
                    execution_example, selected, page_bytes=config.page_bytes
                ),
                **simulate_pinned_bundle_lru(
                    execution_example,
                    selected,
                    cache_capacity_pages=config.cache_pages,
                    page_bytes=config.page_bytes,
                ),
            },
        }
    selected_groups = tuple(
        tuple((layer, int(expert)) for expert in selected[layer].nonzero()[0])
        for layer in range(selected.shape[0])
    )
    selected_keys = tuple(key for group in selected_groups for key in group)
    constrained_groups = (
        route.groups
        if config.prefetch_phase == "before_prefill"
        else tuple(group for trace in route.decode_traces for group in trace)
    )
    maximum_unpinned_group = max(
        len(set(group) - set(selected_keys)) for group in constrained_groups
    )
    if config.bundle_pages + maximum_unpinned_group > config.cache_pages:
        raise ValueError("bundle leaves too little capacity for an atomic route group")

    from bobsphog.moe_checkpoint import MappedExpertSource, QwenMoeSpec, SafetensorCheckpointIndex
    from bobsphog.moe_model import load_paged_qwen
    from transformers import AutoTokenizer

    device = torch.device(config.device)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_root, local_files_only=True)
    prompt_ids = _format_prompt(tokenizer, route.prompt)
    spec = QwenMoeSpec.from_files(
        checkpoint_root / "config.json",
        checkpoint_root / "model.safetensors.index.json",
    )
    index = SafetensorCheckpointIndex(checkpoint_root / "model.safetensors.index.json")
    source = MappedExpertSource(index, spec)
    cache = TracingPinnedCudaExpertCache(
        source,
        device=device,
        capacity_bytes=config.cache_pages * spec.expert_bytes(),
        num_experts=spec.num_experts,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = perf_counter()
    model, load_summary = load_paged_qwen(checkpoint_root, cache, device=device)
    model_load_seconds = perf_counter() - load_started

    prefetch_seconds = 0.0
    before_prefetch = cache.snapshot()
    after_prefetch = before_prefetch
    if config.prefetch_phase == "before_prefill":
        cache.set_pinned_keys(selected_keys)
        torch.cuda.synchronize(device)
        prefetch_started = perf_counter()
        cache.prefetch_groups(selected_groups)
        torch.cuda.synchronize(device)
        prefetch_seconds = perf_counter() - prefetch_started
        after_prefetch = cache.snapshot()

    input_ids = prompt_ids.unsqueeze(0).to(device)
    before_prefill = cache.snapshot()
    torch.cuda.synchronize(device)
    prefill_started = perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(device)
    prefill_seconds = perf_counter() - prefill_started
    _assert_route(cache.drain_request_trace(), route.prefill_groups, "prefill")
    after_prefill = cache.snapshot()

    if config.prefetch_phase == "after_prefill":
        cache.set_pinned_keys(selected_keys)
        before_prefetch = after_prefill
        torch.cuda.synchronize(device)
        prefetch_started = perf_counter()
        cache.prefetch_groups(selected_groups)
        torch.cuda.synchronize(device)
        prefetch_seconds = perf_counter() - prefetch_started
        after_prefetch = cache.snapshot()
    before_decode = cache.snapshot()

    predicted_ids = [int(output.logits[0, -1].argmax())]
    selected_ids = [route.selected_token_ids[0]]
    past_key_values = output.past_key_values
    decode_latencies: list[float] = []
    for token_index, expected_trace in enumerate(route.decode_traces, start=1):
        current_id = torch.tensor([[selected_ids[-1]]], dtype=torch.long, device=device)
        attention_mask = torch.ones(
            (1, prompt_ids.numel() + token_index), dtype=torch.long, device=device
        )
        torch.cuda.synchronize(device)
        started = perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=current_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        decode_latencies.append(perf_counter() - started)
        _assert_route(cache.drain_request_trace(), expected_trace, f"decode {token_index}")
        predicted_ids.append(int(output.logits[0, -1].argmax()))
        selected_ids.append(route.selected_token_ids[token_index])
        past_key_values = output.past_key_values

    final_stats = cache.snapshot()
    decode = summarize_decode_latencies(decode_latencies)
    time_to_first_token = prefill_seconds + (
        prefetch_seconds if config.prefetch_phase == "before_prefill" else 0.0
    )
    end_to_end = prefill_seconds + prefetch_seconds + float(decode["seconds"])
    result = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "prompt": {key: record[key] for key in ("id", "domain", "split", "prompt")},
        "selection": selection_metrics,
        "maximum_unpinned_atomic_group": maximum_unpinned_group,
        "load": {**asdict(load_summary), "model_construction_seconds": model_load_seconds},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_fraction_of_checkpoint": torch.cuda.max_memory_allocated(device) / spec.checkpoint_bytes,
        "prompt_prefetch_seconds": prefetch_seconds,
        "prompt_prefetch_cache": _stats_delta(before_prefetch, after_prefetch),
        "prefill_forward_seconds": prefill_seconds,
        "prefill_demand_cache": _stats_delta(before_prefill, after_prefill),
        "decode": decode,
        "decode_demand_cache": _stats_delta(before_decode, final_stats),
        "time_to_first_token_seconds": time_to_first_token,
        "end_to_end_seconds": end_to_end,
        "end_to_end_output_tokens_per_second": len(selected_ids) / end_to_end,
        "forced_path_top1_agreement_fraction": sum(
            predicted == selected for predicted, selected in zip(predicted_ids, selected_ids)
        ) / len(selected_ids),
        "route_agreement_fraction": 1.0,
        "selected_token_ids": selected_ids,
        "selected_text": tokenizer.decode(selected_ids, skip_special_tokens=True),
        "final_cache_stats": asdict(final_stats),
    }
    cache.close()
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--method",
        choices=("global_frequency", "prefill_reuse", "nearest_neighbor", "conditional_coactivation"),
        default="nearest_neighbor",
    )
    parser.add_argument("--bundle-pages", type=int, default=2048)
    parser.add_argument("--cache-pages", type=int, default=2560)
    parser.add_argument("--page-bytes", type=int, default=6_291_456)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument(
        "--execution-trace",
        help="optional B2 trace recorded on the live execution backend",
    )
    parser.add_argument(
        "--prefetch-phase",
        choices=("before_prefill", "after_prefill"),
        default="before_prefill",
    )
    print(json.dumps(run_b22_live(B22LiveConfig(**vars(parser.parse_args()))), indent=2))


if __name__ == "__main__":
    main()
