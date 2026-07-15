"""Background host staging for predicted expert pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Condition, Thread
from time import perf_counter
from typing import Iterable

from bobsphog.expert_cache import ExpertKey, ExpertSource
from bobsphog.moe_checkpoint import ExpertWeights


@dataclass
class StagedSourceStats:
    requested_pages: int = 0
    staged_pages: int = 0
    staged_bytes: int = 0
    staged_hits: int = 0
    direct_loads: int = 0
    duplicate_loads: int = 0
    background_seconds: float = 0.0
    foreground_wait_seconds: float = 0.0

    def snapshot(self) -> StagedSourceStats:
        return StagedSourceStats(**asdict(self))


class AsyncStagedExpertSource:
    """Read predicted pages into host RAM while foreground inference proceeds."""

    def __init__(self, base: ExpertSource) -> None:
        self.base = base
        self.spec = getattr(base, "spec", None)
        self.stats = StagedSourceStats()
        self._condition = Condition()
        self._ready: dict[ExpertKey, ExpertWeights] = {}
        self._inflight: set[ExpertKey] = set()
        self._consumed: set[ExpertKey] = set()
        self._thread: Thread | None = None
        self._wait_for_inflight = False
        self._cancelled = False

    def start(self, keys: Iterable[ExpertKey]) -> None:
        ordered = tuple(dict.fromkeys(keys))
        if self._thread is not None:
            raise RuntimeError("background staging has already started")
        self.stats.requested_pages = len(ordered)
        self._thread = Thread(
            target=self._stage,
            args=(ordered,),
            name="bobsphog-expert-stage",
            daemon=True,
        )
        self._thread.start()

    def _stage(self, keys: tuple[ExpertKey, ...]) -> None:
        started = perf_counter()
        try:
            for key in keys:
                with self._condition:
                    if self._cancelled:
                        break
                    if key in self._consumed or key in self._ready:
                        continue
                    self._inflight.add(key)
                weights = self.base.load(*key, pin_memory=False)
                with self._condition:
                    self._inflight.discard(key)
                    if key not in self._consumed:
                        self._ready[key] = weights
                        self.stats.staged_pages += 1
                        self.stats.staged_bytes += weights.parameter_bytes
                    self._condition.notify_all()
        finally:
            self.stats.background_seconds = perf_counter() - started
            with self._condition:
                self._condition.notify_all()

    def set_wait_for_inflight(self, enabled: bool) -> None:
        self._wait_for_inflight = enabled

    @staticmethod
    def _pin(weights: ExpertWeights, requested: bool) -> ExpertWeights:
        if not requested or weights.gate_up.is_pinned():
            return weights
        return ExpertWeights(
            gate_up=weights.gate_up.pin_memory(),
            down=weights.down.pin_memory(),
        )

    def load(
        self,
        layer: int,
        expert: int,
        *,
        pin_memory: bool = False,
    ) -> ExpertWeights:
        key = (layer, expert)
        wait_started = perf_counter()
        with self._condition:
            while (
                self._wait_for_inflight
                and key in self._inflight
                and key not in self._ready
            ):
                self._condition.wait()
            self.stats.foreground_wait_seconds += perf_counter() - wait_started
            ready = self._ready.pop(key, None)
            if ready is not None:
                self._consumed.add(key)
                self.stats.staged_hits += 1
                return self._pin(ready, pin_memory)
            duplicate = key in self._inflight
            self._consumed.add(key)
            self.stats.direct_loads += 1
            if duplicate:
                self.stats.duplicate_loads += 1
        return self.base.load(layer, expert, pin_memory=pin_memory)

    def finish(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def close(self) -> None:
        self.finish()
        with self._condition:
            self._ready.clear()
            self._inflight.clear()
