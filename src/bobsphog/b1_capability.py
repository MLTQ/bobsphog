"""Compare full-resident and demand-paged Qwen capability traces."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from bobsphog.expert_cache import CudaExpertCache, ExpertCacheStats
from bobsphog.moe_checkpoint import (
    MappedExpertSource,
    QwenMoeSpec,
    SafetensorCheckpointIndex,
)
from bobsphog.moe_model import load_paged_qwen
from bobsphog.reference_model import load_reference_qwen


@dataclass(frozen=True)
class CapabilityProbe:
    name: str
    prompt: str
    expected_substring: str


DEFAULT_PROBES = (
    CapabilityProbe(
        "instruction",
        "Reply with exactly SAPPHIRE and nothing else.",
        "sapphire",
    ),
    CapabilityProbe(
        "factual",
        "What is the capital of France? Answer with only the city name.",
        "paris",
    ),
    CapabilityProbe(
        "arithmetic",
        "A shelf has 17 books. Nine are removed. How many remain? Answer only.",
        "8",
    ),
    CapabilityProbe(
        "translation",
        "Translate 'good morning' into Spanish. Answer only with the translation.",
        "buenos",
    ),
    CapabilityProbe(
        "science",
        "What process lets green plants convert light into chemical energy? Answer only.",
        "photosynthesis",
    ),
    CapabilityProbe(
        "code",
        "Complete this Python assignment so y is x squared. Return one line: y =",
        "x",
    ),
)


@dataclass(frozen=True)
class B1CapabilityConfig:
    checkpoint: str
    device: str = "cuda:0"
    cache_pages: int = 320
    max_new_tokens: int = 16
    probe_limit: int = len(DEFAULT_PROBES)
    autoregressive_probes: int = 0
    top_k: int = 5


@dataclass
class ReferenceTrace:
    probe: CapabilityProbe
    prompt_ids: Tensor
    sequence: Tensor
    generated_ids: Tensor
    logits: Tensor
    generated_text: str


def _stats_delta(before: ExpertCacheStats, after: ExpertCacheStats) -> dict[str, Any]:
    return {
        field_name: getattr(after, field_name) - getattr(before, field_name)
        for field_name in asdict(before)
    }


def compare_teacher_forced_logits(
    reference_logits: Tensor,
    paged_logits: Tensor,
    generated_ids: Tensor,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Measure distribution and greedy-decision parity for one generated trace."""

    if reference_logits.shape != paged_logits.shape:
        raise ValueError("reference and paged logits must have the same shape")
    if reference_logits.ndim != 2:
        raise ValueError("logits must have shape [steps, vocabulary]")
    if generated_ids.shape != reference_logits.shape[:1]:
        raise ValueError("generated_ids must contain one token per logit step")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    reference = reference_logits.float()
    paged = paged_logits.float()
    reference_log_probs = F.log_softmax(reference, dim=-1)
    paged_log_probs = F.log_softmax(paged, dim=-1)
    kl_per_step = (
        reference_log_probs.exp() * (reference_log_probs - paged_log_probs)
    ).sum(dim=-1)
    reference_top1 = reference.argmax(dim=-1)
    paged_top1 = paged.argmax(dim=-1)
    decision_matches = paged_top1.eq(reference_top1)
    generated_matches = paged_top1.eq(generated_ids.to(paged_top1.device))

    effective_top_k = min(top_k, reference.shape[-1])
    reference_top = reference.topk(effective_top_k, dim=-1).indices
    paged_top = paged.topk(effective_top_k, dim=-1).indices
    top_k_overlap = (
        reference_top.unsqueeze(-1).eq(paged_top.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
    )
    absolute_error = (reference - paged).abs()
    mismatch = (~decision_matches).nonzero(as_tuple=False)
    return {
        "steps": reference.shape[0],
        "mean_kl_per_token": float(kl_per_step.mean()),
        "max_kl_per_token": float(kl_per_step.max()),
        "mean_absolute_logit_error": float(absolute_error.mean()),
        "max_absolute_logit_error": float(absolute_error.max()),
        "reference_top1_agreement_fraction": float(decision_matches.float().mean()),
        "generated_token_agreement_fraction": float(generated_matches.float().mean()),
        "all_reference_top1_decisions_match": bool(decision_matches.all()),
        "all_generated_tokens_match": bool(generated_matches.all()),
        "mean_top_k_overlap_fraction": float(top_k_overlap.mean()),
        "first_reference_top1_mismatch_step": (
            None if mismatch.numel() == 0 else int(mismatch[0, 0])
        ),
        "paged_top1_ids": [int(token) for token in paged_top1],
    }


def _format_prompt(tokenizer: Any, prompt: str) -> Tensor:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(rendered, return_tensors="pt").input_ids[0]


def _checkpoint_bytes(checkpoint_root: Path) -> int:
    return sum(path.stat().st_size for path in checkpoint_root.glob("*.safetensors"))


def _collect_reference_traces(
    model: Any,
    tokenizer: Any,
    probes: tuple[CapabilityProbe, ...],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[list[ReferenceTrace], float]:
    traces: list[ReferenceTrace] = []
    started = perf_counter()
    for probe in probes:
        prompt_ids = _format_prompt(tokenizer, probe.prompt).to(device)
        with torch.inference_mode():
            generated = model.generate(
                prompt_ids.unsqueeze(0),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            ).squeeze(0)
        generated_ids = generated[prompt_ids.numel():]
        if generated_ids.numel() == 0:
            raise RuntimeError(f"reference generated no tokens for probe {probe.name}")
        teacher_input = generated[:-1].unsqueeze(0)
        with torch.inference_mode():
            reference_output = model(
                input_ids=teacher_input,
                use_cache=False,
                logits_to_keep=generated_ids.numel(),
            )
        traces.append(
            ReferenceTrace(
                probe=probe,
                prompt_ids=prompt_ids.cpu(),
                sequence=generated.cpu(),
                generated_ids=generated_ids.cpu(),
                logits=reference_output.logits[0].cpu(),
                generated_text=tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ),
            )
        )
    torch.cuda.synchronize(device)
    return traces, perf_counter() - started


def run_b1_capability(config: B1CapabilityConfig) -> dict[str, Any]:
    """Run a same-device full-versus-paged capability parity experiment."""

    if not torch.cuda.is_available():
        raise RuntimeError("B1 capability comparison requires CUDA or HIP")
    if config.cache_pages <= 0 or config.max_new_tokens <= 0:
        raise ValueError("cache_pages and max_new_tokens must be positive")
    if not 0 < config.probe_limit <= len(DEFAULT_PROBES):
        raise ValueError("probe_limit is outside the default probe suite")
    if not 0 <= config.autoregressive_probes <= config.probe_limit:
        raise ValueError("autoregressive_probes must be between zero and probe_limit")

    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    probes = DEFAULT_PROBES[: config.probe_limit]

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers is required for B1 capability comparison") from error
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_root,
        local_files_only=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    reference_model, reference_load = load_reference_qwen(
        checkpoint_root,
        device=device,
    )
    torch.cuda.synchronize(device)
    reference_allocated = torch.cuda.memory_allocated(device)
    reference_traces, reference_eval_seconds = _collect_reference_traces(
        reference_model,
        tokenizer,
        probes,
        device=device,
        max_new_tokens=config.max_new_tokens,
    )
    reference_peak = torch.cuda.max_memory_allocated(device)

    del reference_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    allocation_after_reference_free = torch.cuda.memory_allocated(device)

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
    paged_model, paged_load = load_paged_qwen(
        checkpoint_root,
        cache,
        device=device,
    )
    torch.cuda.synchronize(device)
    paged_allocated = torch.cuda.memory_allocated(device)

    probe_results: list[dict[str, Any]] = []
    paged_started = perf_counter()
    for trace in reference_traces:
        before = cache.snapshot()
        teacher_input = trace.sequence[:-1].unsqueeze(0).to(device)
        with torch.inference_mode():
            paged_output = paged_model(
                input_ids=teacher_input,
                use_cache=False,
                logits_to_keep=trace.generated_ids.numel(),
            )
        torch.cuda.synchronize(device)
        comparison = compare_teacher_forced_logits(
            trace.logits,
            paged_output.logits[0].cpu(),
            trace.generated_ids,
            top_k=config.top_k,
        )
        implied_text = tokenizer.decode(
            comparison.pop("paged_top1_ids"),
            skip_special_tokens=True,
        )
        expected = trace.probe.expected_substring.casefold()
        probe_results.append(
            {
                "name": trace.probe.name,
                "prompt": trace.probe.prompt,
                "expected_substring": trace.probe.expected_substring,
                "reference_text": trace.generated_text,
                "paged_teacher_forced_top1_text": implied_text,
                "reference_expected_pass": expected in trace.generated_text.casefold(),
                "paged_expected_pass": expected in implied_text.casefold(),
                "comparison": comparison,
                "cache": _stats_delta(before, cache.snapshot()),
            }
        )
        del paged_output

    autoregressive_matches = 0
    for probe_index, trace in enumerate(
        reference_traces[: config.autoregressive_probes]
    ):
        before = cache.snapshot()
        generated_started = perf_counter()
        with torch.inference_mode():
            paged_sequence = paged_model.generate(
                trace.prompt_ids.unsqueeze(0).to(device),
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            ).squeeze(0)
        torch.cuda.synchronize(device)
        sequence_match = torch.equal(paged_sequence.cpu(), trace.sequence)
        autoregressive_matches += int(sequence_match)
        paged_generated_ids = paged_sequence[trace.prompt_ids.numel():]
        probe_results[probe_index].update(
            {
                "paged_autoregressive_text": tokenizer.decode(
                    paged_generated_ids,
                    skip_special_tokens=True,
                ),
                "autoregressive_sequence_match": sequence_match,
                "autoregressive_seconds": perf_counter() - generated_started,
                "autoregressive_cache": _stats_delta(before, cache.snapshot()),
            }
        )
    paged_eval_seconds = perf_counter() - paged_started
    paged_peak = torch.cuda.max_memory_allocated(device)

    total_steps = sum(result["comparison"]["steps"] for result in probe_results)
    weighted_mean_kl = sum(
        result["comparison"]["mean_kl_per_token"]
        * result["comparison"]["steps"]
        for result in probe_results
    ) / total_steps
    matching_steps = sum(
        result["comparison"]["reference_top1_agreement_fraction"]
        * result["comparison"]["steps"]
        for result in probe_results
    )
    generated_matching_steps = sum(
        result["comparison"]["generated_token_agreement_fraction"]
        * result["comparison"]["steps"]
        for result in probe_results
    )
    reference_passes = sum(result["reference_expected_pass"] for result in probe_results)
    paged_passes = sum(result["paged_expected_pass"] for result in probe_results)
    result = {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "checkpoint_parameter_bytes": _checkpoint_bytes(checkpoint_root),
        "reference": {
            "load": asdict(reference_load),
            "allocated_bytes": reference_allocated,
            "peak_allocated_bytes": reference_peak,
            "evaluation_seconds": reference_eval_seconds,
        },
        "after_reference_free_allocated_bytes": allocation_after_reference_free,
        "paged": {
            "load": asdict(paged_load),
            "allocated_before_probes_bytes": paged_allocated,
            "peak_allocated_bytes": paged_peak,
            "expert_cache_capacity_bytes": cache.capacity_bytes,
            "evaluation_seconds": paged_eval_seconds,
            "final_cache_stats": asdict(cache.snapshot()),
        },
        "memory": {
            "paged_to_reference_peak_fraction": paged_peak / reference_peak,
            "peak_reduction_fraction": 1.0 - paged_peak / reference_peak,
        },
        "capability": {
            "probes": len(probe_results),
            "generated_steps": total_steps,
            "reference_expected_passes": reference_passes,
            "paged_expected_passes": paged_passes,
            "reference_top1_agreement_fraction": matching_steps / total_steps,
            "generated_token_agreement_fraction": (
                generated_matching_steps / total_steps
            ),
            "all_probe_decisions_match": all(
                result["comparison"]["all_reference_top1_decisions_match"]
                for result in probe_results
            ),
            "all_teacher_forced_generated_tokens_match": all(
                result["comparison"]["all_generated_tokens_match"]
                for result in probe_results
            ),
            "autoregressive_probes": config.autoregressive_probes,
            "autoregressive_sequence_matches": autoregressive_matches,
            "all_autoregressive_sequences_match": (
                autoregressive_matches == config.autoregressive_probes
            ),
            "mean_teacher_kl_per_token": weighted_mean_kl,
            "max_teacher_kl_per_token": max(
                result["comparison"]["max_kl_per_token"]
                for result in probe_results
            ),
        },
        "probe_results": probe_results,
    }
    cache.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-pages", type=int, default=320)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--probe-limit", type=int, default=len(DEFAULT_PROBES))
    parser.add_argument("--autoregressive-probes", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_b1_capability(B1CapabilityConfig(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
