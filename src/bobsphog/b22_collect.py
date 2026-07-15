"""Collect multi-prompt prefill and decode routing traces on resident Qwen."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from bobsphog.b1_throughput import _format_prompt
from bobsphog.reference_model import load_reference_qwen
from bobsphog.route_trace import ExpertRouteRecorder


@dataclass(frozen=True)
class B22CollectConfig:
    checkpoint: str
    prompts: str
    device: str = "cuda:0"
    output_tokens: int = 32
    limit: int | None = None


def load_prompt_corpus(path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).expanduser().read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("prompt corpus must be a non-empty JSON list")
    required = {"id", "domain", "split", "prompt"}
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"every prompt requires fields {sorted(required)!r}")
        record = {key: str(item[key]) for key in required}
        if record["id"] in seen:
            raise ValueError(f"duplicate prompt id: {record['id']}")
        if record["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split for prompt {record['id']}")
        if not record["prompt"].strip():
            raise ValueError(f"prompt {record['id']} is empty")
        seen.add(record["id"])
        records.append(record)
    return records


def _serialize_groups(groups: tuple[tuple[tuple[int, int], ...], ...]) -> list[list[int]]:
    return [[expert for _, expert in group] for group in groups]


def _collect_one(
    model: Any,
    tokenizer: Any,
    recorder: ExpertRouteRecorder,
    record: dict[str, str],
    *,
    device: torch.device,
    output_tokens: int,
) -> dict[str, Any]:
    prompt_ids = _format_prompt(tokenizer, record["prompt"])
    input_ids = prompt_ids.unsqueeze(0).to(device)
    torch.cuda.synchronize(device)
    started = perf_counter()
    recorder.begin()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    prefill_groups = recorder.end()
    selected_ids = [int(output.logits[0, -1].argmax())]
    past_key_values = output.past_key_values
    decode_traces: list[tuple[tuple[tuple[int, int], ...], ...]] = []

    for token_index in range(1, output_tokens):
        current_id = torch.tensor(
            [[selected_ids[-1]]], dtype=torch.long, device=device
        )
        attention_mask = torch.ones(
            (1, prompt_ids.numel() + token_index),
            dtype=torch.long,
            device=device,
        )
        recorder.begin()
        with torch.inference_mode():
            output = model(
                input_ids=current_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        decode_traces.append(recorder.end())
        selected_ids.append(int(output.logits[0, -1].argmax()))
        past_key_values = output.past_key_values

    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    return {
        **record,
        "prompt_tokens": prompt_ids.numel(),
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "selected_token_ids": selected_ids,
        "selected_text": tokenizer.decode(selected_ids, skip_special_tokens=True),
        "prefill_experts_by_layer": _serialize_groups(prefill_groups),
        "decode_experts_by_token_and_layer": [
            _serialize_groups(trace) for trace in decode_traces
        ],
    }


def run_b22_collect(config: B22CollectConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("B2.2 collection requires CUDA or HIP")
    if config.output_tokens < 2:
        raise ValueError("output_tokens must be at least two")
    if config.limit is not None and config.limit <= 0:
        raise ValueError("limit must be positive")
    prompts = load_prompt_corpus(config.prompts)
    if config.limit is not None:
        prompts = prompts[: config.limit]

    checkpoint_root = Path(config.checkpoint).expanduser().resolve()
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers is required for B2.2 collection") from error
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_root, local_files_only=True)
    model, load_summary = load_reference_qwen(checkpoint_root, device=device)
    recorder = ExpertRouteRecorder(model)
    try:
        traces = [
            _collect_one(
                model,
                tokenizer,
                recorder,
                record,
                device=device,
                output_tokens=config.output_tokens,
            )
            for record in prompts
        ]
    finally:
        recorder.close()
    return {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(device),
        "load": asdict(load_summary),
        "prompts": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run_b22_collect(B22CollectConfig(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
