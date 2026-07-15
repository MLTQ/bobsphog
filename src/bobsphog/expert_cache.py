"""Bounded CUDA cache and routed execution for exact MoE expert pages."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Iterable, Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from bobsphog.moe_checkpoint import ExpertWeights

ExpertKey = tuple[int, int]


class ExpertSource(Protocol):
    def load(
        self,
        layer: int,
        expert: int,
        *,
        pin_memory: bool = False,
    ) -> ExpertWeights: ...


@dataclass
class ExpertCacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_transferred: int = 0
    source_load_seconds: float = 0.0
    schedule_seconds: float = 0.0
    host_wait_seconds: float = 0.0


@dataclass
class CachedExpert:
    gate_up: Tensor
    down: Tensor
    parameter_bytes: int
    ready: torch.cuda.Event
    staging: ExpertWeights | None


class CudaExpertCache:
    """Keep an exact LRU subset of routed experts on one CUDA device."""

    def __init__(
        self,
        source: ExpertSource,
        *,
        device: torch.device,
        capacity_bytes: int,
        num_experts: int = 256,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("CudaExpertCache requires an available CUDA device")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.source = source
        self.device = device
        self.capacity_bytes = capacity_bytes
        self.num_experts = num_experts
        self.stats = ExpertCacheStats()
        self._cache: OrderedDict[ExpertKey, CachedExpert] = OrderedDict()
        self._cache_bytes = 0
        self._stream = torch.cuda.Stream(device=device)

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    def snapshot(self) -> ExpertCacheStats:
        return ExpertCacheStats(**asdict(self.stats))

    @staticmethod
    def _unique(keys: Iterable[ExpertKey]) -> tuple[ExpertKey, ...]:
        return tuple(dict.fromkeys(keys))

    def _reap_staging(self) -> None:
        for cached in self._cache.values():
            if cached.staging is not None and cached.ready.query():
                cached.staging = None

    def _evict_for(self, required_bytes: int, protected: set[ExpertKey]) -> None:
        while self._cache_bytes + required_bytes > self.capacity_bytes:
            victim = next((key for key in self._cache if key not in protected), None)
            if victim is None:
                raise RuntimeError("cache cannot fit requested expert working set")
            cached = self._cache.pop(victim)
            started = perf_counter()
            cached.ready.synchronize()
            self.stats.host_wait_seconds += perf_counter() - started
            self._cache_bytes -= cached.parameter_bytes
            self.stats.evictions += 1

    def schedule(self, keys: Iterable[ExpertKey]) -> tuple[ExpertKey, ...]:
        """Load cold experts and enqueue their H2D copies."""

        started = perf_counter()
        requested = self._unique(keys)
        protected = set(requested)
        self._reap_staging()
        missing: list[ExpertKey] = []
        for key in requested:
            self.stats.requests += 1
            if key in self._cache:
                self.stats.hits += 1
                self._cache.move_to_end(key)
            else:
                self.stats.misses += 1
                missing.append(key)

        loaded: list[tuple[ExpertKey, ExpertWeights]] = []
        for layer, expert in missing:
            load_started = perf_counter()
            weights = self.source.load(layer, expert, pin_memory=True)
            self.stats.source_load_seconds += perf_counter() - load_started
            loaded.append(((layer, expert), weights))
        requested_bytes = sum(weights.parameter_bytes for _, weights in loaded)
        resident_requested_bytes = sum(
            self._cache[key].parameter_bytes
            for key in requested
            if key in self._cache
        )
        if requested_bytes + resident_requested_bytes > self.capacity_bytes:
            raise ValueError("requested expert set exceeds cache capacity")

        for key, weights in loaded:
            self._evict_for(weights.parameter_bytes, protected)
            with torch.cuda.stream(self._stream):
                gate_up = weights.gate_up.to(self.device, non_blocking=True)
                down = weights.down.to(self.device, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(self._stream)
            self._cache[key] = CachedExpert(
                gate_up=gate_up,
                down=down,
                parameter_bytes=weights.parameter_bytes,
                ready=ready,
                staging=weights,
            )
            self._cache_bytes += weights.parameter_bytes
            self.stats.bytes_transferred += weights.parameter_bytes
        self.stats.schedule_seconds += perf_counter() - started
        return requested

    def wait(self, keys: Iterable[ExpertKey]) -> tuple[ExpertKey, ...]:
        requested = self._unique(keys)
        started = perf_counter()
        for key in requested:
            cached = self._cache.get(key)
            if cached is None:
                raise RuntimeError(f"expert {key!r} was not scheduled")
            cached.ready.synchronize()
            cached.staging = None
        self.stats.host_wait_seconds += perf_counter() - started
        return requested

    def apply(self, key: ExpertKey, inputs: Tensor) -> Tensor:
        cached = self._cache.get(key)
        if cached is None:
            raise RuntimeError(f"expert {key!r} was not scheduled")
        torch.cuda.current_stream(self.device).wait_event(cached.ready)
        self._cache.move_to_end(key)
        expert_inputs = inputs.to(cached.gate_up.dtype)
        gate, up = F.linear(expert_inputs, cached.gate_up).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, cached.down).to(inputs.dtype)

    def apply_routed(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
        activation: Callable[[Tensor], Tensor] = F.silu,
    ) -> Tensor:
        """Apply the exact Qwen expert accumulation for one layer."""

        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be flattened to [tokens, hidden]")
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(
            top_k_index,
            num_classes=self.num_experts,
        ).permute(2, 1, 0)
        for expert_tensor in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
            expert = int(expert_tensor[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert])
            current = hidden_states[token_idx]
            cached = self._cache.get((layer, expert))
            if cached is None:
                raise RuntimeError(f"expert {(layer, expert)!r} was not scheduled")
            torch.cuda.current_stream(self.device).wait_event(cached.ready)
            self._cache.move_to_end((layer, expert))
            expert_inputs = current.to(cached.gate_up.dtype)
            gate, up = F.linear(expert_inputs, cached.gate_up).chunk(2, dim=-1)
            current = F.linear(activation(gate) * up, cached.down)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def close(self) -> None:
        self._stream.synchronize()
        self._cache.clear()
        self._cache_bytes = 0
        torch.cuda.empty_cache()
