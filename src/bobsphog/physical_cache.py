"""Pinned-CPU page storage and a bounded asynchronous CUDA factor cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bobsphog.model import ToyTransformer
from bobsphog.paging import PagePlan

PageKey = tuple[str, int]


@dataclass
class CacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_transferred: int = 0
    schedule_seconds: float = 0.0
    host_wait_seconds: float = 0.0
    prefetch_seconds: float = 0.0


@dataclass(frozen=True)
class CachedFactors:
    left: Tensor
    right: Tensor
    parameter_bytes: int
    ready: torch.cuda.Event


class PhysicalPageCache:
    """Keep source factors pinned on CPU and exact hot copies in bounded VRAM."""

    def __init__(
        self,
        model: ToyTransformer,
        *,
        device: torch.device,
        capacity_bytes: int,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("PhysicalPageCache requires an available CUDA device")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.model = model
        self.device = device
        self.capacity_bytes = capacity_bytes
        self.stats = CacheStats()
        self._sources: dict[PageKey, nn.Module] = {}
        self._cache: OrderedDict[PageKey, CachedFactors] = OrderedDict()
        self._cache_bytes = 0

        model.eval().requires_grad_(False)
        model.to(device=device, dtype=dtype)
        for layer_id, layer in model.paged_layers().items():
            for page_id, page in enumerate(layer.pages):
                page.to("cpu")
                page.left = nn.Parameter(
                    page.left.detach().contiguous().pin_memory(),
                    requires_grad=False,
                )
                page.right = nn.Parameter(
                    page.right.detach().contiguous().pin_memory(),
                    requires_grad=False,
                )
                self._sources[(layer_id, page_id)] = page
            layer.set_page_provider(self)
        self._stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    @property
    def source_bytes(self) -> int:
        return sum(
            sum(parameter.numel() * parameter.element_size() for parameter in source.parameters())
            for source in self._sources.values()
        )

    def snapshot(self) -> CacheStats:
        return CacheStats(**asdict(self.stats))

    def _requested_keys(self, plan: PagePlan) -> tuple[PageKey, ...]:
        keys: list[PageKey] = []
        for layer_id, layer in self.model.paged_layers().items():
            keys.extend(
                (layer_id, page_id)
                for page_id in plan.selected(layer_id, layer.page_count)
            )
        return tuple(keys)

    def _source_bytes(self, key: PageKey) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self._sources[key].parameters()
        )

    def _evict_for(self, required_bytes: int, protected: set[PageKey]) -> None:
        while self._cache_bytes + required_bytes > self.capacity_bytes:
            victim = next((key for key in self._cache if key not in protected), None)
            if victim is None:
                raise RuntimeError("cache cannot fit requested working set")
            removed = self._cache.pop(victim)
            wait_started = perf_counter()
            removed.ready.synchronize()
            self.stats.host_wait_seconds += perf_counter() - wait_started
            self._cache_bytes -= removed.parameter_bytes
            self.stats.evictions += 1

    def schedule(self, plan: PagePlan) -> tuple[PageKey, ...]:
        """Enqueue missing page copies without waiting for them on the host."""

        started = perf_counter()
        requested = self._requested_keys(plan)
        requested_set = set(requested)
        requested_bytes = sum(self._source_bytes(key) for key in requested)
        if requested_bytes > self.capacity_bytes:
            raise ValueError("requested page set exceeds cache capacity")

        missing: list[PageKey] = []
        for key in requested:
            self.stats.requests += 1
            if key in self._cache:
                self.stats.hits += 1
                self._cache.move_to_end(key)
            else:
                self.stats.misses += 1
                missing.append(key)

        for key in missing:
            parameter_bytes = self._source_bytes(key)
            self._evict_for(parameter_bytes, requested_set)
            source = self._sources[key]
            with torch.cuda.stream(self._stream):
                left = source.left.detach().to(self.device, non_blocking=True)
                right = source.right.detach().to(self.device, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(self._stream)
            self._cache[key] = CachedFactors(left, right, parameter_bytes, ready)
            self._cache_bytes += parameter_bytes
            self.stats.bytes_transferred += parameter_bytes
        self.stats.schedule_seconds += perf_counter() - started
        return requested

    def wait(self, plan: PagePlan) -> tuple[PageKey, ...]:
        """Wait on the host until every requested page copy has completed."""

        requested = self._requested_keys(plan)
        started = perf_counter()
        for key in requested:
            factors = self._cache.get(key)
            if factors is None:
                raise RuntimeError(f"page {key!r} was not scheduled")
            factors.ready.synchronize()
        self.stats.host_wait_seconds += perf_counter() - started
        return requested

    def prepare(self, plan: PagePlan) -> tuple[PageKey, ...]:
        """Schedule a plan and synchronously make it ready for execution."""

        started = perf_counter()
        requested = self.schedule(plan)
        self.wait(plan)
        self.stats.prefetch_seconds += perf_counter() - started
        return requested

    def apply(self, layer_id: str, page_id: int, inputs: Tensor) -> Tensor:
        key = (layer_id, page_id)
        factors = self._cache.get(key)
        if factors is None:
            raise RuntimeError(f"page {key!r} was not scheduled")
        torch.cuda.current_stream(self.device).wait_event(factors.ready)
        self._cache.move_to_end(key)
        return F.linear(F.linear(inputs, factors.right), factors.left)

    def close(self) -> None:
        self._stream.synchronize()
        for layer in self.model.paged_layers().values():
            layer.set_page_provider(None)
        self._cache.clear()
        self._cache_bytes = 0
        torch.cuda.empty_cache()
