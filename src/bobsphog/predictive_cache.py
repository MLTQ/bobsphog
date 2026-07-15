"""Pinned prompt-bundle cache with LRU-managed residual capacity."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

from bobsphog.expert_cache import CudaExpertCache, ExpertKey


class PinnedCudaExpertCache(CudaExpertCache):
    """Keep a predicted page bundle resident while applying LRU elsewhere."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.pinned_keys: set[ExpertKey] = set()

    def set_pinned_keys(self, keys: Iterable[ExpertKey]) -> None:
        pinned = set(keys)
        if len(pinned) * self.source.spec.expert_bytes() > self.capacity_bytes:
            raise ValueError("pinned expert set exceeds cache capacity")
        self.pinned_keys = pinned

    def _evict_for(self, required_bytes: int, protected: set[ExpertKey]) -> None:
        protected = protected | self.pinned_keys
        while self._cache_bytes + required_bytes > self.capacity_bytes:
            victim = next((key for key in self._cache if key not in protected), None)
            if victim is None:
                raise RuntimeError("cache has no unpinned page available for eviction")
            cached = self._cache.pop(victim)
            started = perf_counter()
            cached.ready.synchronize()
            self.stats.host_wait_seconds += perf_counter() - started
            self._cache_bytes -= cached.parameter_bytes
            self.stats.evictions += 1

    def prefetch_groups(
        self,
        groups: Iterable[Iterable[ExpertKey]],
    ) -> tuple[ExpertKey, ...]:
        """Synchronously materialize groups without recording live route calls."""

        normalized = tuple(self._unique(group) for group in groups)
        requested = self._unique(key for group in normalized for key in group)
        for group in normalized:
            CudaExpertCache.schedule(self, group)
            self.wait(group)
        return requested


class TracingPinnedCudaExpertCache(PinnedCudaExpertCache):
    """Pinned LRU cache that exposes demand-only route groups for parity checks."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._request_trace: list[tuple[ExpertKey, ...]] = []

    def schedule(self, keys: Iterable[ExpertKey]) -> tuple[ExpertKey, ...]:
        requested = self._unique(keys)
        self._request_trace.append(requested)
        return super().schedule(requested)

    def drain_request_trace(self) -> tuple[tuple[ExpertKey, ...], ...]:
        trace = tuple(self._request_trace)
        self._request_trace.clear()
        return trace
