"""Meta-device loader and cache-backed expert adapter for GLM-5.2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor, nn

from bobsphog.expert_cache import CudaExpertCache


def checkpoint_key_to_glm_key(checkpoint_key: str) -> str | None:
    """Return resident GLM keys and reject routed experts and MTP-only tensors."""

    if ".mlp.experts." in checkpoint_key:
        return None
    if checkpoint_key.startswith("model.layers.78."):
        return None
    if checkpoint_key.startswith("model.") or checkpoint_key == "lm_head.weight":
        return checkpoint_key
    return None


class PagedGlmExperts(nn.Module):
    """Drop-in GLM expert collection backed by the bounded exact-page cache."""

    def __init__(self, layer: int, cache: CudaExpertCache) -> None:
        super().__init__()
        self.layer = layer
        self.cache = cache

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        experts = tuple(int(value) for value in torch.unique(top_k_index).tolist())
        self.cache.schedule((self.layer, expert) for expert in experts)
        return self.cache.apply_routed(
            self.layer,
            hidden_states,
            top_k_index,
            top_k_weights,
        )


@dataclass(frozen=True)
class GlmScaffoldLoadSummary:
    loaded_tensors: int
    loaded_bytes: int
    ignored_tensors: int
    touched_shards: int
    load_seconds: float


def load_paged_glm(
    checkpoint_root: Path,
    cache: CudaExpertCache,
    *,
    device: torch.device,
) -> tuple[nn.Module, GlmScaffoldLoadSummary]:
    """Load GLM-5.2's resident scaffold while leaving routed experts file-backed."""

    if device.type != "cuda":
        raise ValueError("the GLM scaffold loader requires CUDA or HIP")
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaForCausalLM,
        )
    except ImportError as error:
        raise RuntimeError(
            "Transformers 5.12+, Accelerate, and safetensors are required for GLM loading"
        ) from error

    config = AutoConfig.from_pretrained(checkpoint_root, local_files_only=True)
    with init_empty_weights():
        model = GlmMoeDsaForCausalLM(config)
    sparse_layers = 0
    for layer_id, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "experts"):
            layer.mlp.experts = PagedGlmExperts(layer_id, cache)
            sparse_layers += 1
    if sparse_layers == 0:
        raise RuntimeError("GLM model contains no replaceable sparse expert modules")

    index = json.loads((checkpoint_root / "model.safetensors.index.json").read_text())
    targets = dict(model.named_parameters()) | dict(model.named_buffers())
    by_shard: dict[str, list[tuple[str, str]]] = {}
    ignored_tensors = 0
    for checkpoint_key, shard_name in index["weight_map"].items():
        target_key = checkpoint_key_to_glm_key(checkpoint_key)
        if target_key is None:
            ignored_tensors += 1
            continue
        if target_key not in targets:
            raise KeyError(f"checkpoint key maps to unknown GLM tensor: {target_key}")
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
        name for name, parameter in model.named_parameters() if parameter.device.type == "meta"
    ]
    if meta_parameters:
        raise RuntimeError(f"GLM scaffold parameters remain on meta device: {meta_parameters[:5]}")
    model.model.rotary_emb.to(device)
    model.eval().requires_grad_(False)
    return model, GlmScaffoldLoadSummary(
        loaded_tensors=loaded_tensors,
        loaded_bytes=loaded_bytes,
        ignored_tensors=ignored_tensors,
        touched_shards=len(by_shard),
        load_seconds=perf_counter() - started,
    )
