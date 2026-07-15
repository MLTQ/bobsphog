"""Page-selection plans and logical execution traces."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Mapping


@dataclass(frozen=True)
class PageEvent:
    """One paged layer's logical residency during a forward pass."""

    layer_id: str
    selected_pages: tuple[int, ...]
    base_bytes: int
    page_bytes: tuple[int, ...]

    @property
    def selected_page_bytes(self) -> int:
        return sum(self.page_bytes)


@dataclass
class PagingTrace:
    """Collect page events for instrumentation and debugging."""

    events: list[PageEvent] = field(default_factory=list)

    def record(self, event: PageEvent) -> None:
        self.events.append(event)

    @property
    def selected_page_bytes(self) -> int:
        return sum(event.selected_page_bytes for event in self.events)

    @property
    def paged_layer_base_bytes(self) -> int:
        return sum(event.base_bytes for event in self.events)

    @property
    def selected_page_count(self) -> int:
        return sum(len(event.selected_pages) for event in self.events)


@dataclass(frozen=True)
class PagePlan:
    """Resolve layer IDs to selected logical page IDs."""

    selections: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    default: Literal["all", "base"] = "all"

    @classmethod
    def full(cls) -> PagePlan:
        return cls(default="all")

    @classmethod
    def base_only(cls) -> PagePlan:
        return cls(default="base")

    @classmethod
    def uniform_prefix(
        cls,
        page_counts: Mapping[str, int],
        pages_per_layer: int,
    ) -> PagePlan:
        if pages_per_layer < 0:
            raise ValueError("pages_per_layer must be non-negative")
        return cls(
            selections={
                layer_id: tuple(range(min(page_count, pages_per_layer)))
                for layer_id, page_count in page_counts.items()
            },
            default="base",
        )

    @classmethod
    def random_dropout(
        cls,
        page_counts: Mapping[str, int],
        dropout_rate: float,
        *,
        seed: int,
    ) -> PagePlan:
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between zero and one")
        generator = random.Random(seed)
        return cls(
            selections={
                layer_id: tuple(
                    page_id
                    for page_id in range(page_count)
                    if generator.random() >= dropout_rate
                )
                for layer_id, page_count in page_counts.items()
            },
            default="base",
        )

    def selected(self, layer_id: str, page_count: int) -> tuple[int, ...]:
        selected = self.selections.get(layer_id)
        if selected is None:
            return tuple(range(page_count)) if self.default == "all" else ()
        if len(selected) != len(set(selected)):
            raise ValueError(f"page plan for {layer_id!r} contains duplicates")
        if any(page_id < 0 or page_id >= page_count for page_id in selected):
            raise IndexError(f"page plan for {layer_id!r} contains an invalid page ID")
        return selected
