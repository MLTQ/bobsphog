"""File-backed expert access for GLM-5.2 checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from bobsphog.moe_checkpoint import ExpertSourceStats, ExpertWeights


@dataclass(frozen=True)
class GlmMoeSpec:
    num_layers: int
    num_mtp_layers: int
    sparse_layers: tuple[int, ...]
    num_experts: int
    experts_per_token: int
    hidden_size: int
    intermediate_size: int
    checkpoint_bytes: int

    @classmethod
    def from_files(cls, config_path: Path, index_path: Path) -> GlmMoeSpec:
        config = json.loads(config_path.read_text())
        index = json.loads(index_path.read_text())
        layer_types = tuple(config["mlp_layer_types"])
        num_layers = int(config["num_hidden_layers"])
        if len(layer_types) != num_layers:
            raise ValueError("mlp_layer_types must contain one entry per layer")
        sparse_layers = tuple(
            layer for layer, layer_type in enumerate(layer_types) if layer_type == "sparse"
        )
        if not sparse_layers:
            raise ValueError("checkpoint config contains no sparse MoE layers")
        return cls(
            num_layers=num_layers,
            num_mtp_layers=int(config.get("num_nextn_predict_layers", 0)),
            sparse_layers=sparse_layers,
            num_experts=int(config["n_routed_experts"]),
            experts_per_token=int(config["num_experts_per_tok"]),
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["moe_intermediate_size"]),
            checkpoint_bytes=int(index["metadata"]["total_size"]),
        )

    @property
    def expert_parameter_count(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    def expert_bytes(self, element_size: int = 2) -> int:
        if element_size <= 0:
            raise ValueError("element_size must be positive")
        return self.expert_parameter_count * element_size

    @property
    def routed_expert_bytes(self) -> int:
        return len(self.sparse_layers) * self.num_experts * self.expert_bytes()

    @property
    def estimated_causal_scaffold_upper_bound_bytes(self) -> int:
        checkpoint_expert_layers = len(self.sparse_layers) + self.num_mtp_layers
        return (
            self.checkpoint_bytes
            - checkpoint_expert_layers * self.num_experts * self.expert_bytes()
        )


class GlmSafetensorCheckpointIndex:
    """Resolve one GLM routed expert's three tensors to local shards."""

    def __init__(self, index_path: Path) -> None:
        payload = json.loads(index_path.read_text())
        self.root = index_path.parent
        self.total_size_bytes = int(payload["metadata"]["total_size"])
        self.weight_map: dict[str, str] = payload["weight_map"]

    @staticmethod
    def expert_tensor_names(layer: int, expert: int) -> tuple[str, str, str]:
        if layer < 0 or expert < 0:
            raise ValueError("layer and expert must be non-negative")
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        return (
            f"{prefix}.gate_proj.weight",
            f"{prefix}.up_proj.weight",
            f"{prefix}.down_proj.weight",
        )

    def shard_for(self, tensor_name: str) -> Path:
        try:
            shard_name = self.weight_map[tensor_name]
        except KeyError as error:
            raise KeyError(f"tensor {tensor_name!r} is absent from checkpoint index") from error
        return self.root / shard_name


class MappedGlmExpertSource:
    """Materialize one exact GLM expert without loading its neighboring experts."""

    def __init__(self, index: GlmSafetensorCheckpointIndex, spec: GlmMoeSpec) -> None:
        self.index = index
        self.spec = spec
        self.stats = ExpertSourceStats()
        self._sparse_layers = frozenset(spec.sparse_layers)

    def _read_tensor(self, tensor_name: str) -> Tensor:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("safetensors is required for mapped expert loading") from error

        shard_path = self.index.shard_for(tensor_name)
        if not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard is not present: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            return shard.get_tensor(tensor_name)

    def load(
        self,
        layer: int,
        expert: int,
        *,
        pin_memory: bool = False,
    ) -> ExpertWeights:
        if layer not in self._sparse_layers:
            raise IndexError("layer is not a sparse MoE layer")
        if not 0 <= expert < self.spec.num_experts:
            raise IndexError("expert is out of range")

        started = perf_counter()
        gate_name, up_name, down_name = self.index.expert_tensor_names(layer, expert)
        gate = self._read_tensor(gate_name)
        up = self._read_tensor(up_name)
        down = self._read_tensor(down_name)
        expected_up = (self.spec.intermediate_size, self.spec.hidden_size)
        expected_down = (self.spec.hidden_size, self.spec.intermediate_size)
        if tuple(gate.shape) != expected_up or tuple(up.shape) != expected_up:
            raise ValueError(
                "expert gate/up shapes do not match config: "
                f"gate={tuple(gate.shape)}, up={tuple(up.shape)}"
            )
        if tuple(down.shape) != expected_down:
            raise ValueError(
                f"expert down shape does not match config: down={tuple(down.shape)}"
            )
        if gate.dtype != up.dtype or gate.dtype != down.dtype:
            raise ValueError("expert tensors must share one dtype")

        gate_up = torch.empty(
            (2 * self.spec.intermediate_size, self.spec.hidden_size),
            dtype=gate.dtype,
            pin_memory=pin_memory,
        )
        gate_up[: self.spec.intermediate_size].copy_(gate)
        gate_up[self.spec.intermediate_size :].copy_(up)
        if pin_memory:
            down = down.pin_memory()
        else:
            down = down.clone()
        weights = ExpertWeights(gate_up=gate_up, down=down)
        self.stats.loads += 1
        self.stats.bytes_read += weights.parameter_bytes
        self.stats.load_seconds += perf_counter() - started
        return weights

    def describe(self) -> dict[str, Any]:
        return {
            "layers": self.spec.num_layers,
            "sparse_layers": len(self.spec.sparse_layers),
            "experts_per_layer": self.spec.num_experts,
            "experts_per_token": self.spec.experts_per_token,
            "expert_bytes_bfloat16": self.spec.expert_bytes(),
            "estimated_causal_scaffold_upper_bound_bytes": (
                self.spec.estimated_causal_scaffold_upper_bound_bytes
            ),
            "checkpoint_bytes": self.spec.checkpoint_bytes,
        }
