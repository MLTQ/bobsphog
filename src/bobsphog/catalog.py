"""Stable global indexing and plan construction for logical weight pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from bobsphog.model import ToyTransformer
from bobsphog.paging import PagePlan


@dataclass(frozen=True)
class PageRef:
    global_id: int
    layer_id: str
    page_id: int
    parameter_bytes: int

    @property
    def name(self) -> str:
        return f"{self.layer_id}#{self.page_id}"


class PageCatalog:
    """Map stable global page IDs to per-layer page plans."""

    def __init__(self, refs: tuple[PageRef, ...]) -> None:
        if tuple(ref.global_id for ref in refs) != tuple(range(len(refs))):
            raise ValueError("page global IDs must be contiguous and ordered")
        self.refs = refs

    @classmethod
    def from_model(cls, model: ToyTransformer) -> PageCatalog:
        raw: list[tuple[int, str, int, int]] = []
        for layer_order, (layer_id, layer) in enumerate(model.paged_layers().items()):
            for page_id, parameter_bytes in enumerate(layer.page_parameter_bytes):
                raw.append((page_id, layer_id, layer_order, parameter_bytes))
        raw.sort(key=lambda item: (item[0], item[2]))
        refs = tuple(
            PageRef(
                global_id=global_id,
                layer_id=layer_id,
                page_id=page_id,
                parameter_bytes=parameter_bytes,
            )
            for global_id, (page_id, layer_id, _, parameter_bytes) in enumerate(raw)
        )
        return cls(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def validate_ids(self, global_ids: Iterable[int]) -> tuple[int, ...]:
        ids = tuple(global_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("global page IDs must be unique")
        if any(global_id < 0 or global_id >= len(self) for global_id in ids):
            raise IndexError("global page ID is out of range")
        return ids

    def plan(self, global_ids: Iterable[int]) -> PagePlan:
        ids = self.validate_ids(global_ids)
        selections: dict[str, list[int]] = {}
        for global_id in ids:
            ref = self.refs[global_id]
            selections.setdefault(ref.layer_id, []).append(ref.page_id)
        return PagePlan(
            selections={
                layer_id: tuple(sorted(page_ids))
                for layer_id, page_ids in selections.items()
            },
            default="base",
        )

    def resident_mask(
        self,
        global_ids: Iterable[int],
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        ids = self.validate_ids(global_ids)
        mask = torch.zeros(len(self), dtype=torch.float32, device=device)
        if ids:
            mask[list(ids)] = 1.0
        return mask

    def static_prefix(self, budget: int) -> tuple[int, ...]:
        if not 0 <= budget <= len(self):
            raise ValueError("budget must be between zero and page count")
        return tuple(range(budget))

    def selected_bytes(self, global_ids: Iterable[int]) -> int:
        return sum(self.refs[index].parameter_bytes for index in self.validate_ids(global_ids))

    def names(self, global_ids: Iterable[int]) -> list[str]:
        return [self.refs[index].name for index in self.validate_ids(global_ids)]
