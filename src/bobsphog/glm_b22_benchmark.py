"""Benchmark exact paged GLM-5.2 with a large cache and B2.2-style pinning."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from bobsphog.expert_cache import ExpertCacheStats, ExpertKey
from bobsphog.glm_checkpoint import (
    GlmMoeSpec,
    GlmSafetensorCheckpointIndex,
    MappedGlmExpertSource,
)
from bobsphog.glm_model import PagedGlmExperts, load_paged_glm
from bobsphog.predictive_cache import PinnedCudaExpertCache

CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class MmluExample:
    subject: str
    row_index: int
    question: str
    choices: tuple[str, str, str, str]
    answer: str


@dataclass(frozen=True)
class GlmB22BenchmarkConfig:
    checkpoint: str
    mmlu_root: str | None = None
    device: str = "cuda:0"
    mode: str = "all"
    accuracy_cache_pages: int = 256
    speed_cache_pages: int = 320
    pinned_pages: int = 300
    samples_per_subject: int = 1
    subject_limit: int | None = 16
    accuracy_batch_size: int = 64
    seed: int = 20260715
    decode_forwards: int = 4
    speed_prompt: str = "Explain in one sentence why the sky appears blue."


class GlmRoutePinnedCache(PinnedCudaExpertCache):
    """Pinned exact cache that can retain router tensors for one forward."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._record_routes = False
        self._route_tensors: list[tuple[int, Tensor]] = []

    def begin_route_recording(self) -> None:
        self._route_tensors.clear()
        self._record_routes = True

    def observe_routes(self, layer: int, top_k_index: Tensor) -> None:
        if self._record_routes:
            self._route_tensors.append((layer, top_k_index.detach()))

    def end_route_recording(self) -> dict[int, tuple[int, ...]]:
        self._record_routes = False
        counts: dict[int, Tensor] = {}
        for layer, indices in self._route_tensors:
            layer_counts = torch.bincount(
                indices.flatten(), minlength=self.num_experts
            )
            counts[layer] = counts.get(layer, torch.zeros_like(layer_counts)) + layer_counts
        self._route_tensors.clear()
        return {
            layer: tuple(int(value) for value in layer_counts.cpu().tolist())
            for layer, layer_counts in counts.items()
        }


def _resolve_mmlu_data_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / "test").is_dir():
        return root
    if (root / "data" / "test").is_dir():
        return root / "data"
    raise FileNotFoundError(f"MMLU test directory is absent below {root}")


def load_mmlu_examples(
    root: Path,
    *,
    samples_per_subject: int,
    seed: int,
    subject_limit: int | None = None,
) -> list[MmluExample]:
    """Select a deterministic, subject-stratified sample from official MMLU CSVs."""

    if samples_per_subject <= 0:
        raise ValueError("samples_per_subject must be positive")
    if subject_limit is not None and subject_limit <= 0:
        raise ValueError("subject_limit must be positive when supplied")
    data_root = _resolve_mmlu_data_root(root)
    files = sorted((data_root / "test").glob("*_test.csv"))
    if not files:
        raise FileNotFoundError("MMLU contains no *_test.csv files")
    if subject_limit is not None and subject_limit < len(files):
        digest = hashlib.sha256(f"{seed}:subjects".encode()).digest()
        subject_rng = random.Random(int.from_bytes(digest[:8], "big"))
        files = sorted(subject_rng.sample(files, subject_limit))

    examples: list[MmluExample] = []
    for path in files:
        subject = path.name.removesuffix("_test.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if samples_per_subject > len(rows):
            raise ValueError(
                f"subject {subject!r} has only {len(rows)} test rows"
            )
        digest = hashlib.sha256(f"{seed}:{subject}".encode()).digest()
        subject_rng = random.Random(int.from_bytes(digest[:8], "big"))
        selected = sorted(subject_rng.sample(range(len(rows)), samples_per_subject))
        for row_index in selected:
            row = rows[row_index]
            if len(row) != 6 or row[-1] not in CHOICES:
                raise ValueError(f"invalid MMLU row in {path} at index {row_index}")
            examples.append(
                MmluExample(
                    subject=subject,
                    row_index=row_index,
                    question=row[0],
                    choices=(row[1], row[2], row[3], row[4]),
                    answer=row[5],
                )
            )
    return examples


def format_mmlu_prompt(example: MmluExample) -> str:
    subject = example.subject.replace("_", " ")
    lines = [
        "Answer the following multiple-choice question.",
        "Respond with exactly one letter: A, B, C, or D.",
        "",
        f"Subject: {subject}",
        example.question,
    ]
    lines.extend(
        f"{label}. {choice}" for label, choice in zip(CHOICES, example.choices)
    )
    lines.append("Answer:")
    return "\n".join(lines)


def select_prefill_bundle(
    route_counts: dict[int, tuple[int, ...]],
    sparse_layers: Iterable[int],
    budget: int,
) -> tuple[ExpertKey, ...]:
    """Select a deterministic equal-layer bundle by prefill route frequency."""

    layers = tuple(sparse_layers)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not layers:
        raise ValueError("sparse_layers must not be empty")
    base, remainder = divmod(budget, len(layers))
    selected: list[ExpertKey] = []
    for position, layer in enumerate(layers):
        counts = route_counts.get(layer)
        if counts is None:
            raise ValueError(f"prefill route counts are missing layer {layer}")
        quota = base + int(position < remainder)
        if quota > len(counts):
            raise ValueError("per-layer bundle quota exceeds expert count")
        ranked = sorted(range(len(counts)), key=lambda expert: (-counts[expert], expert))
        selected.extend((layer, expert) for expert in ranked[:quota])
    return tuple(selected)


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not 0 <= correct <= total or total <= 0:
        raise ValueError("Wilson interval requires 0 <= correct <= total and total > 0")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def _stats_delta(before: ExpertCacheStats, after: ExpertCacheStats) -> dict[str, int | float]:
    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in asdict(before)
    }


def _rate(numerator: int | float, seconds: float) -> float:
    return float(numerator) / seconds if seconds > 0 else 0.0


def _make_cache(
    source: MappedGlmExpertSource,
    spec: GlmMoeSpec,
    device: torch.device,
    pages: int,
) -> GlmRoutePinnedCache:
    return GlmRoutePinnedCache(
        source,
        device=device,
        capacity_bytes=pages * spec.expert_bytes(),
        num_experts=spec.num_experts,
    )


def _replace_model_cache(model: nn.Module, cache: GlmRoutePinnedCache) -> int:
    replaced = 0
    for module in model.modules():
        if isinstance(module, PagedGlmExperts):
            module.cache = cache
            replaced += 1
    if replaced == 0:
        raise RuntimeError("model contains no paged GLM experts")
    return replaced


def _render_chat(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _answer_token_ids(tokenizer: Any) -> tuple[int, int, int, int]:
    token_ids: list[int] = []
    for label in CHOICES:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"MMLU answer label {label!r} is not one tokenizer token")
        token_ids.append(int(encoded[0]))
    return tuple(token_ids)  # type: ignore[return-value]


def _run_accuracy(
    config: GlmB22BenchmarkConfig,
    model: nn.Module,
    tokenizer: Any,
    cache: GlmRoutePinnedCache,
    device: torch.device,
) -> dict[str, Any]:
    if config.mmlu_root is None:
        raise ValueError("mmlu_root is required for accuracy mode")
    examples = load_mmlu_examples(
        Path(config.mmlu_root),
        samples_per_subject=config.samples_per_subject,
        seed=config.seed,
        subject_limit=config.subject_limit,
    )
    answer_token_ids = _answer_token_ids(tokenizer)
    rows: list[dict[str, Any]] = []
    batch_metrics: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_padded_tokens = 0
    total_seconds = 0.0
    before_all = cache.snapshot()
    torch.cuda.reset_peak_memory_stats(device)

    for start in range(0, len(examples), config.accuracy_batch_size):
        batch_examples = examples[start : start + config.accuracy_batch_size]
        rendered = [_render_chat(tokenizer, format_mmlu_prompt(item)) for item in batch_examples]
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        prompt_tokens = int(attention_mask.sum().item())
        padded_tokens = int(input_ids.numel())
        before_batch = cache.snapshot()
        torch.cuda.synchronize(device)
        started = perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        seconds = perf_counter() - started
        after_batch = cache.snapshot()
        choice_logits = output.logits[:, -1, list(answer_token_ids)].float().cpu()
        predictions = choice_logits.argmax(dim=-1).tolist()
        for item, prediction, logits in zip(batch_examples, predictions, choice_logits):
            predicted = CHOICES[int(prediction)]
            rows.append(
                {
                    **asdict(item),
                    "predicted": predicted,
                    "correct": predicted == item.answer,
                    "choice_logits": {
                        label: float(value) for label, value in zip(CHOICES, logits.tolist())
                    },
                }
            )
        batch_metrics.append(
            {
                "examples": len(batch_examples),
                "prompt_tokens": prompt_tokens,
                "padded_tokens": padded_tokens,
                "seconds": seconds,
                "prompt_tokens_per_second": _rate(prompt_tokens, seconds),
                "cache": _stats_delta(before_batch, after_batch),
            }
        )
        total_prompt_tokens += prompt_tokens
        total_padded_tokens += padded_tokens
        total_seconds += seconds
        del output, input_ids, attention_mask

    after_all = cache.snapshot()
    correct = sum(int(row["correct"]) for row in rows)
    low, high = wilson_interval(correct, len(rows))
    selected_digest = hashlib.sha256(
        json.dumps(
            [(row["subject"], row["row_index"], row["answer"]) for row in rows],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "benchmark": "MMLU subject-stratified zero-shot sample",
        "protocol": (
            "One deterministic test item per selected subject; GLM chat template; "
            "thinking disabled; direct A/B/C/D next-token logit scoring; no dev examples."
        ),
        "examples": len(rows),
        "subjects": len({row["subject"] for row in rows}),
        "correct": correct,
        "accuracy": correct / len(rows),
        "wilson_95": {"low": low, "high": high},
        "random_baseline": 0.25,
        "selection_sha256": selected_digest,
        "answer_token_ids": dict(zip(CHOICES, answer_token_ids)),
        "prompt_tokens": total_prompt_tokens,
        "padded_tokens": total_padded_tokens,
        "forward_seconds": total_seconds,
        "prompt_tokens_per_second": _rate(total_prompt_tokens, total_seconds),
        "padded_tokens_per_second": _rate(total_padded_tokens, total_seconds),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cache": _stats_delta(before_all, after_all),
        "batches": batch_metrics,
        "results": rows,
        "claim_scope": (
            "This is a small zero-shot stratified sample, not the full 14k-question "
            "five-shot MMLU score. Its confidence interval is reported explicitly."
        ),
    }


def _decode_branch(
    model: nn.Module,
    past_key_values: Any,
    input_path: list[int],
    *,
    device: torch.device,
    forced_expected: list[int] | None = None,
) -> tuple[list[int], list[float], list[bool]]:
    predicted: list[int] = []
    timings: list[float] = []
    agreements: list[bool] = []
    for position, token_id in enumerate(input_path):
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        torch.cuda.synchronize(device)
        started = perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        timings.append(perf_counter() - started)
        next_token = int(output.logits[0, -1].argmax().item())
        predicted.append(next_token)
        if forced_expected is not None:
            agreements.append(next_token == forced_expected[position])
        past_key_values = output.past_key_values
    return predicted, timings, agreements


def _decode_greedy(
    model: nn.Module,
    past_key_values: Any,
    first_token: int,
    forwards: int,
    *,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    predictions: list[int] = []
    timings: list[float] = []
    token_id = first_token
    for _ in range(forwards):
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        torch.cuda.synchronize(device)
        started = perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        timings.append(perf_counter() - started)
        token_id = int(output.logits[0, -1].argmax().item())
        predictions.append(token_id)
        past_key_values = output.past_key_values
    return predictions, timings


def _run_speed(
    config: GlmB22BenchmarkConfig,
    model: nn.Module,
    tokenizer: Any,
    cache: GlmRoutePinnedCache,
    source: MappedGlmExpertSource,
    spec: GlmMoeSpec,
    device: torch.device,
) -> dict[str, Any]:
    rendered = _render_chat(tokenizer, config.speed_prompt)
    input_ids = tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    prompt_tokens = int(input_ids.numel())
    cache.begin_route_recording()
    before_prefill = cache.snapshot()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(device)
    prefill_seconds = perf_counter() - started
    after_prefill = cache.snapshot()
    route_counts = cache.end_route_recording()
    first_token = int(output.logits[0, -1].argmax().item())
    baseline_past = copy.deepcopy(output.past_key_values)
    predicted_past = copy.deepcopy(output.past_key_values)
    del output

    bundle = select_prefill_bundle(route_counts, spec.sparse_layers, config.pinned_pages)
    baseline_before = cache.snapshot()
    baseline_predictions, baseline_timings = _decode_greedy(
        model,
        baseline_past,
        first_token,
        config.decode_forwards,
        device=device,
    )
    baseline_input_path = [first_token, *baseline_predictions]
    baseline_after = cache.snapshot()
    baseline_peak = torch.cuda.max_memory_allocated(device)

    cache.close()
    predicted_cache = _make_cache(
        source, spec, device, config.speed_cache_pages
    )
    _replace_model_cache(model, predicted_cache)
    predicted_cache.set_pinned_keys(bundle)
    grouped_bundle = tuple(
        tuple(key for key in bundle if key[0] == layer) for layer in spec.sparse_layers
    )
    before_prefetch = predicted_cache.snapshot()
    torch.cuda.reset_peak_memory_stats(device)
    prefetch_started = perf_counter()
    predicted_cache.prefetch_groups(grouped_bundle)
    torch.cuda.synchronize(device)
    prefetch_seconds = perf_counter() - prefetch_started
    after_prefetch = predicted_cache.snapshot()
    predicted_before = predicted_cache.snapshot()
    predicted_outputs, predicted_timings, agreements = _decode_branch(
        model,
        predicted_past,
        baseline_input_path[:-1],
        device=device,
        forced_expected=baseline_predictions,
    )
    predicted_after = predicted_cache.snapshot()
    predicted_peak = torch.cuda.max_memory_allocated(device)
    predicted_cache.close()

    baseline_seconds = sum(baseline_timings)
    predicted_seconds = sum(predicted_timings)
    selected_tokens = [first_token, *baseline_predictions]
    baseline_misses = int(_stats_delta(baseline_before, baseline_after)["misses"])
    predicted_misses = int(_stats_delta(predicted_before, predicted_after)["misses"])
    return {
        "prompt": config.speed_prompt,
        "formatted_prompt_tokens": prompt_tokens,
        "prefill": {
            "seconds": prefill_seconds,
            "prompt_tokens_per_second": _rate(prompt_tokens, prefill_seconds),
            "cache": _stats_delta(before_prefill, after_prefill),
        },
        "cache_pages": config.speed_cache_pages,
        "cache_capacity_bytes": config.speed_cache_pages * spec.expert_bytes(),
        "pinned_pages": len(bundle),
        "pinned_bytes": len(bundle) * spec.expert_bytes(),
        "bundle_policy": (
            "B2.2 prefill-reuse baseline: equal per-layer quota ranked by this "
            "prompt's exact prefill route frequency."
        ),
        "bundle_prefetch": {
            "seconds": prefetch_seconds,
            "cache": _stats_delta(before_prefetch, after_prefetch),
        },
        "decode_forwards": config.decode_forwards,
        "baseline": {
            "seconds": baseline_seconds,
            "tokens_per_second": _rate(config.decode_forwards, baseline_seconds),
            "per_token_seconds": baseline_timings,
            "cache": _stats_delta(baseline_before, baseline_after),
            "peak_allocated_bytes": baseline_peak,
        },
        "predicted": {
            "seconds": predicted_seconds,
            "tokens_per_second": _rate(config.decode_forwards, predicted_seconds),
            "per_token_seconds": predicted_timings,
            "cache": _stats_delta(predicted_before, predicted_after),
            "peak_allocated_bytes": predicted_peak,
            "top1_agreement": sum(agreements) / len(agreements) if agreements else 1.0,
        },
        "decode_speedup": baseline_seconds / predicted_seconds,
        "decode_fault_reduction": (
            1 - predicted_misses / baseline_misses if baseline_misses else 0.0
        ),
        "baseline_end_to_end_seconds": prefill_seconds + baseline_seconds,
        "predicted_end_to_end_seconds": prefill_seconds + prefetch_seconds + predicted_seconds,
        "selected_token_ids": selected_tokens,
        "selected_text": tokenizer.decode(selected_tokens),
        "predicted_token_ids": predicted_outputs,
        "validation": (
            "Both branches share deep-copied exact prefill KV state. The predicted "
            "branch is forced along the baseline inputs and checks independent top-1 parity."
        ),
    }


def run_glm_b22_benchmark(config: GlmB22BenchmarkConfig) -> dict[str, Any]:
    if config.mode not in {"accuracy", "speed", "all"}:
        raise ValueError("mode must be accuracy, speed, or all")
    if not torch.cuda.is_available():
        raise RuntimeError("GLM B2.2 benchmark requires CUDA or HIP")
    positive = (
        config.accuracy_cache_pages,
        config.speed_cache_pages,
        config.samples_per_subject,
        config.accuracy_batch_size,
        config.decode_forwards,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("page, sample, batch, and decode counts must be positive")

    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    spec = GlmMoeSpec.from_files(
        checkpoint_root / "config.json",
        checkpoint_root / "model.safetensors.index.json",
    )
    if config.mode in {"accuracy", "all"} and config.accuracy_cache_pages < spec.num_experts:
        raise ValueError("accuracy cache must fit one saturated 256-expert layer")
    if config.pinned_pages + spec.experts_per_token > config.speed_cache_pages:
        raise ValueError("pinned bundle must leave one decode layer group unpinned")

    index = GlmSafetensorCheckpointIndex(checkpoint_root / "model.safetensors.index.json")
    source = MappedGlmExpertSource(index, spec)
    initial_pages = (
        config.accuracy_cache_pages
        if config.mode in {"accuracy", "all"}
        else config.speed_cache_pages
    )
    cache = _make_cache(source, spec, device, initial_pages)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers 5.12+ is required") from error

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = perf_counter()
    model, load_summary = load_paged_glm(checkpoint_root, cache, device=device)
    total_load_seconds = perf_counter() - load_started
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_root, local_files_only=True)
    result: dict[str, Any] = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "checkpoint": source.describe(),
        "load": {
            **asdict(load_summary),
            "total_model_construction_seconds": total_load_seconds,
        },
        "method": (
            "Exact BF16 routed experts with a bounded file-backed cache; no "
            "quantization, expert dropping, substitution, or output repair."
        ),
    }
    if config.mode in {"accuracy", "all"}:
        result["accuracy"] = _run_accuracy(
            config, model, tokenizer, cache, device
        )
    if config.mode in {"speed", "all"}:
        if config.mode == "all":
            cache.close()
            cache = _make_cache(source, spec, device, config.speed_cache_pages)
            _replace_model_cache(model, cache)
        result["speed"] = _run_speed(
            config, model, tokenizer, cache, source, spec, device
        )
    elif config.mode == "accuracy":
        cache.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mmlu-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("accuracy", "speed", "all"), default="all")
    parser.add_argument("--accuracy-cache-pages", type=int, default=256)
    parser.add_argument("--speed-cache-pages", type=int, default=320)
    parser.add_argument("--pinned-pages", type=int, default=300)
    parser.add_argument("--samples-per-subject", type=int, default=1)
    parser.add_argument("--subject-limit", type=int, default=16)
    parser.add_argument("--accuracy-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--decode-forwards", type=int, default=4)
    parser.add_argument(
        "--speed-prompt",
        default="Explain in one sentence why the sky appears blue.",
    )
    args = parser.parse_args()
    print(json.dumps(run_glm_b22_benchmark(GlmB22BenchmarkConfig(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
