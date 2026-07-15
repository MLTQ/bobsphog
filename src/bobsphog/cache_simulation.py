"""Trace-driven simulations for grouped expert-page cache requests."""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable, Sequence

from bobsphog.expert_cache import ExpertKey


@dataclass(frozen=True)
class CacheSimulationResult:
    policy: str
    capacity_pages: int
    request_groups: int
    requests: int
    hits: int
    misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def describe(self, page_bytes: int) -> dict[str, int | float | str]:
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        result = asdict(self)
        result["hit_rate"] = self.hit_rate
        result["bytes_transferred"] = self.misses * page_bytes
        return result


def _normalize_groups(
    groups: Iterable[Iterable[ExpertKey]],
) -> list[tuple[ExpertKey, ...]]:
    return [tuple(dict.fromkeys(group)) for group in groups]


def _validate(groups: Sequence[Sequence[ExpertKey]], capacity_pages: int) -> None:
    if capacity_pages <= 0:
        raise ValueError("capacity_pages must be positive")
    largest = max((len(group) for group in groups), default=0)
    if largest > capacity_pages:
        raise ValueError(
            f"capacity {capacity_pages} cannot fit a request group of {largest} pages"
        )


def simulate_grouped_lru(
    groups: Iterable[Iterable[ExpertKey]],
    capacity_pages: int,
) -> CacheSimulationResult:
    """Replay the production cache's protected-group LRU policy."""

    normalized = _normalize_groups(groups)
    _validate(normalized, capacity_pages)
    cache: OrderedDict[ExpertKey, None] = OrderedDict()
    requests = hits = misses = evictions = 0
    for group in normalized:
        protected = set(group)
        missing: list[ExpertKey] = []
        for key in group:
            requests += 1
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                misses += 1
                missing.append(key)
        for key in missing:
            while len(cache) >= capacity_pages:
                victim = next(
                    (candidate for candidate in cache if candidate not in protected),
                    None,
                )
                if victim is None:
                    raise RuntimeError("no evictable page despite validated group size")
                cache.pop(victim)
                evictions += 1
            cache[key] = None
        # Production execution touches every requested expert after scheduling.
        # Replaying that order matters when hits and misses are interleaved.
        for key in group:
            cache.move_to_end(key)
    return CacheSimulationResult(
        policy="grouped_lru",
        capacity_pages=capacity_pages,
        request_groups=len(normalized),
        requests=requests,
        hits=hits,
        misses=misses,
        evictions=evictions,
    )


def simulate_grouped_belady(
    groups: Iterable[Iterable[ExpertKey]],
    capacity_pages: int,
) -> CacheSimulationResult:
    """Compute the group-residency-aware offline-optimal replacement bound."""

    normalized = _normalize_groups(groups)
    _validate(normalized, capacity_pages)
    future: dict[ExpertKey, deque[int]] = defaultdict(deque)
    for group_index, group in enumerate(normalized):
        for key in group:
            future[key].append(group_index)

    cache: set[ExpertKey] = set()
    requests = hits = misses = evictions = 0
    for group_index, group in enumerate(normalized):
        protected = set(group)
        for key in group:
            positions = future[key]
            if not positions or positions[0] != group_index:
                raise RuntimeError("future-use index is inconsistent")
            positions.popleft()
            requests += 1
            if key in cache:
                hits += 1
            else:
                misses += 1

        missing = protected - cache
        while len(cache) + len(missing) > capacity_pages:
            candidates = cache - protected
            if not candidates:
                raise RuntimeError("no evictable page despite validated group size")
            victim = max(
                candidates,
                key=lambda key: future[key][0] if future[key] else inf,
            )
            cache.remove(victim)
            evictions += 1
        cache.update(missing)

    return CacheSimulationResult(
        policy="grouped_belady_offline_optimal",
        capacity_pages=capacity_pages,
        request_groups=len(normalized),
        requests=requests,
        hits=hits,
        misses=misses,
        evictions=evictions,
    )
