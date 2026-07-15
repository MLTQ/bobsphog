"""Full-residency Qwen3.6 text-model loader for parity experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn


def checkpoint_key_to_reference_text_key(checkpoint_key: str) -> str | None:
    """Map full checkpoint tensors into the complete text-only model."""

    if checkpoint_key.startswith("model.visual."):
        return None
    if checkpoint_key.startswith("mtp."):
        return None
    language_prefix = "model.language_model."
    if checkpoint_key.startswith(language_prefix):
        return "model." + checkpoint_key.removeprefix(language_prefix)
    if checkpoint_key == "lm_head.weight":
        return checkpoint_key
    return None


@dataclass(frozen=True)
class ReferenceLoadSummary:
    loaded_tensors: int
    loaded_bytes: int
    touched_shards: int
    load_seconds: float


def load_reference_qwen(
    checkpoint_root: Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, ReferenceLoadSummary]:
    """Load every language tensor, including routed experts, onto one GPU."""

    if device.type != "cuda":
        raise ValueError("the Qwen reference loader requires a CUDA/HIP device")
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForCausalLM,
        )
    except ImportError as error:
        raise RuntimeError(
            "Transformers, Accelerate, and safetensors are required for Qwen loading"
        ) from error

    config = AutoConfig.from_pretrained(
        checkpoint_root,
        local_files_only=True,
    ).text_config
    with init_empty_weights():
        model = Qwen3_5MoeForCausalLM(config)

    index = json.loads((checkpoint_root / "model.safetensors.index.json").read_text())
    targets = dict(model.named_parameters()) | dict(model.named_buffers())
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for checkpoint_key, shard_name in index["weight_map"].items():
        target_key = checkpoint_key_to_reference_text_key(checkpoint_key)
        if target_key is None:
            continue
        if target_key not in targets:
            raise KeyError(f"checkpoint key maps to unknown model tensor: {target_key}")
        by_shard.setdefault(shard_name, []).append((checkpoint_key, target_key))

    started = perf_counter()
    loaded_tensors = 0
    loaded_bytes = 0
    for shard_name, mappings in by_shard.items():
        shard_path = checkpoint_root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard is not present: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for checkpoint_key, target_key in mappings:
                value = shard.get_tensor(checkpoint_key)
                set_module_tensor_to_device(
                    model,
                    target_key,
                    device,
                    value=value,
                    dtype=value.dtype,
                )
                loaded_tensors += 1
                loaded_bytes += value.numel() * value.element_size()

    meta_parameters = [
        name for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    ]
    if meta_parameters:
        raise RuntimeError(
            f"reference parameters remain on meta device: {meta_parameters[:5]}"
        )
    model.model.rotary_emb.to(device)
    model.eval().requires_grad_(False)
    return model, ReferenceLoadSummary(
        loaded_tensors=loaded_tensors,
        loaded_bytes=loaded_bytes,
        touched_shards=len(by_shard),
        load_seconds=perf_counter() - started,
    )
