"""Future-aware expert retention and trace-driven prefetch controls."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from math import inf
from time import perf_counter
from typing import Iterable, Sequence

import torch

from bobsphog.expert_cache import CachedExpert, CudaExpertCache, ExpertKey


class OracleTraceMismatch(RuntimeError):
    """Raised when live routing diverges from the recorded oracle trace."""


class FutureUseOracle:
    """Track exact future group positions for Belady-style replacement."""

    def __init__(self, groups: Iterable[Iterable[ExpertKey]]) -> None:
        self.groups = tuple(tuple(dict.fromkeys(group)) for group in groups)
        self._future: dict[ExpertKey, deque[int]] = defaultdict(deque)
        for group_index, group in enumerate(self.groups):
            for key in group:
                self._future[key].append(group_index)
        self.cursor = 0

    def consume(self, requested: Sequence[ExpertKey]) -> None:
        requested_tuple = tuple(requested)
        if self.cursor >= len(self.groups):
            raise OracleTraceMismatch("live routing continued beyond the oracle trace")
        expected = self.groups[self.cursor]
        if requested_tuple != expected:
            raise OracleTraceMismatch(
                f"route mismatch at group {self.cursor}: "
                f"expected {expected!r}, observed {requested_tuple!r}"
            )
        for key in requested_tuple:
            positions = self._future[key]
            if not positions or positions[0] != self.cursor:
                raise RuntimeError("oracle future-use index is inconsistent")
            positions.popleft()
        self.cursor += 1

    def next_use(self, key: ExpertKey) -> float:
        positions = self._future.get(key)
        return float(positions[0]) if positions else inf

    def choose_victim(
        self,
        resident: Iterable[ExpertKey],
        protected: set[ExpertKey],
    ) -> ExpertKey:
        candidates = [key for key in resident if key not in protected]
        if not candidates:
            raise RuntimeError("oracle cache has no evictable page")
        return max(candidates, key=self.next_use)

    @property
    def complete(self) -> bool:
        return self.cursor == len(self.groups)


@dataclass
class OraclePrefetchStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_transferred: int = 0
    source_load_seconds: float = 0.0
    cuda_wait_seconds: float = 0.0
    total_seconds: float = 0.0

    def snapshot(self) -> OraclePrefetchStats:
        return OraclePrefetchStats(**asdict(self))


class OracleExpertCache(CudaExpertCache):
    """Execute exact experts with future-aware eviction and optional prefetch."""

    def __init__(
        self,
        *args: object,
        oracle_groups: Iterable[Iterable[ExpertKey]],
        expert_page_bytes: int,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if expert_page_bytes <= 0:
            raise ValueError("expert_page_bytes must be positive")
        self.oracle = FutureUseOracle(oracle_groups)
        self.expert_page_bytes = expert_page_bytes
        self.prefetch_stats = OraclePrefetchStats()
        self.pinned_keys: set[ExpertKey] = set()

    def set_pinned_keys(self, keys: Iterable[ExpertKey]) -> None:
        pinned = set(keys)
        if len(pinned) * self.expert_page_bytes > self.capacity_bytes:
            raise ValueError("pinned expert set exceeds cache capacity")
        self.pinned_keys = pinned

    def schedule(self, keys: Iterable[ExpertKey]) -> tuple[ExpertKey, ...]:
        requested = self._unique(keys)
        self.oracle.consume(requested)
        return super().schedule(requested)

    def _evict_for(self, required_bytes: int, protected: set[ExpertKey]) -> None:
        protected = protected | self.pinned_keys
        while self._cache_bytes + required_bytes > self.capacity_bytes:
            victim = self.oracle.choose_victim(self._cache, protected)
            cached = self._cache.pop(victim)
            started = perf_counter()
            cached.ready.synchronize()
            waited = perf_counter() - started
            self.stats.host_wait_seconds += waited
            self._cache_bytes -= cached.parameter_bytes
            self.stats.evictions += 1

    def prefetch_groups(
        self,
        groups: Iterable[Iterable[ExpertKey]],
    ) -> tuple[ExpertKey, ...]:
        """Load an upcoming atomic bundle without consuming demand requests."""

        normalized = tuple(self._unique(group) for group in groups)
        requested = self._unique(key for group in normalized for key in group)
        if len(requested) * self.expert_page_bytes > self.capacity_bytes:
            raise ValueError("prefetch bundle exceeds cache capacity")
        protected = set(requested)
        total_started = perf_counter()
        before_evictions = self.stats.evictions
        before_bytes = self.stats.bytes_transferred
        before_source = self.stats.source_load_seconds
        before_wait = self.stats.host_wait_seconds
        self.prefetch_stats.requests += len(requested)

        for group in normalized:
            self._reap_staging()
            missing: list[ExpertKey] = []
            for key in group:
                if key in self._cache:
                    self.prefetch_stats.hits += 1
                else:
                    self.prefetch_stats.misses += 1
                    missing.append(key)
            for layer, expert in missing:
                load_started = perf_counter()
                weights = self.source.load(layer, expert, pin_memory=True)
                self.stats.source_load_seconds += perf_counter() - load_started
                self._evict_for(weights.parameter_bytes, protected)
                with torch.cuda.stream(self._stream):
                    gate_up = weights.gate_up.to(self.device, non_blocking=True)
                    down = weights.down.to(self.device, non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record(self._stream)
                key = (layer, expert)
                self._cache[key] = CachedExpert(
                    gate_up=gate_up,
                    down=down,
                    parameter_bytes=weights.parameter_bytes,
                    ready=ready,
                    staging=weights,
                )
                self._cache_bytes += weights.parameter_bytes
                self.stats.bytes_transferred += weights.parameter_bytes
            # Bound pinned staging to one layer while retaining the whole token.
            self.wait(group)

        elapsed = perf_counter() - total_started
        self.prefetch_stats.evictions += self.stats.evictions - before_evictions
        self.prefetch_stats.bytes_transferred += (
            self.stats.bytes_transferred - before_bytes
        )
        self.prefetch_stats.source_load_seconds += (
            self.stats.source_load_seconds - before_source
        )
        self.prefetch_stats.cuda_wait_seconds += (
            self.stats.host_wait_seconds - before_wait
        )
        self.prefetch_stats.total_seconds += elapsed
        return requested
