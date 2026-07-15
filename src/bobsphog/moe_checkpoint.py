"""Memory-mapped expert access for packed Qwen3.6 MoE checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class QwenMoeSpec:
    num_layers: int
    num_experts: int
    experts_per_token: int
    hidden_size: int
    intermediate_size: int
    checkpoint_bytes: int

    @classmethod
    def from_files(
        cls,
        config_path: Path,
        index_path: Path,
    ) -> QwenMoeSpec:
        config = json.loads(config_path.read_text())
        index = json.loads(index_path.read_text())
        text_config = config.get("text_config", config)
        return cls(
            num_layers=int(text_config["num_hidden_layers"]),
            num_experts=int(text_config["num_experts"]),
            experts_per_token=int(text_config["num_experts_per_tok"]),
            hidden_size=int(text_config["hidden_size"]),
            intermediate_size=int(text_config["moe_intermediate_size"]),
            checkpoint_bytes=int(index["metadata"]["total_size"]),
        )

    @property
    def expert_parameter_count(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    def expert_bytes(self, element_size: int = 2) -> int:
        if element_size <= 0:
            raise ValueError("element_size must be positive")
        return self.expert_parameter_count * element_size


class SafetensorCheckpointIndex:
    """Resolve packed expert tensors to local checkpoint shards."""

    def __init__(self, index_path: Path) -> None:
        payload = json.loads(index_path.read_text())
        self.root = index_path.parent
        self.total_size_bytes = int(payload["metadata"]["total_size"])
        self.weight_map: dict[str, str] = payload["weight_map"]

    @staticmethod
    def expert_tensor_names(layer: int) -> tuple[str, str]:
        if layer < 0:
            raise ValueError("layer must be non-negative")
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        return f"{prefix}.gate_up_proj", f"{prefix}.down_proj"

    def shard_for(self, tensor_name: str) -> Path:
        try:
            shard_name = self.weight_map[tensor_name]
        except KeyError as error:
            raise KeyError(f"tensor {tensor_name!r} is absent from checkpoint index") from error
        return self.root / shard_name

    def expert_shards(self, layer: int) -> tuple[Path, Path]:
        gate_up_name, down_name = self.expert_tensor_names(layer)
        return self.shard_for(gate_up_name), self.shard_for(down_name)


@dataclass(frozen=True)
class ExpertWeights:
    gate_up: Tensor
    down: Tensor

    @property
    def parameter_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.gate_up, self.down)
        )


@dataclass
class ExpertSourceStats:
    loads: int = 0
    bytes_read: int = 0
    load_seconds: float = 0.0


class MappedExpertSource:
    """Materialize individual experts through safetensors slice access."""

    def __init__(
        self,
        index: SafetensorCheckpointIndex,
        spec: QwenMoeSpec,
    ) -> None:
        self.index = index
        self.spec = spec
        self.stats = ExpertSourceStats()

    def _read_slice(self, tensor_name: str, expert: int) -> Tensor:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("safetensors is required for mapped expert loading") from error

        shard_path = self.index.shard_for(tensor_name)
        if not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard is not present: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            return shard.get_slice(tensor_name)[expert].clone()

    def load(
        self,
        layer: int,
        expert: int,
        *,
        pin_memory: bool = False,
    ) -> ExpertWeights:
        if not 0 <= layer < self.spec.num_layers:
            raise IndexError("layer is out of range")
        if not 0 <= expert < self.spec.num_experts:
            raise IndexError("expert is out of range")
        gate_up_name, down_name = self.index.expert_tensor_names(layer)
        started = perf_counter()
        gate_up = self._read_slice(gate_up_name, expert)
        down = self._read_slice(down_name, expert)
        expected_gate_up = (2 * self.spec.intermediate_size, self.spec.hidden_size)
        expected_down = (self.spec.hidden_size, self.spec.intermediate_size)
        if tuple(gate_up.shape) != expected_gate_up or tuple(down.shape) != expected_down:
            raise ValueError(
                "expert tensor shapes do not match config: "
                f"gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}"
            )
        if pin_memory:
            gate_up = gate_up.pin_memory()
            down = down.pin_memory()
        weights = ExpertWeights(gate_up=gate_up, down=down)
        self.stats.loads += 1
        self.stats.bytes_read += weights.parameter_bytes
        self.stats.load_seconds += perf_counter() - started
        return weights

    def describe(self) -> dict[str, Any]:
        return {
            "layers": self.spec.num_layers,
            "experts_per_layer": self.spec.num_experts,
            "experts_per_token": self.spec.experts_per_token,
            "expert_bytes_bfloat16": self.spec.expert_bytes(2),
            "checkpoint_bytes": self.spec.checkpoint_bytes,
        }
