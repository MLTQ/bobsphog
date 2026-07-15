"""Capture per-layer routed-expert sets from a full-resident Qwen model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from bobsphog.expert_cache import ExpertKey


class ExpertRouteRecorder:
    """Attach non-mutating hooks to Qwen expert modules and capture one forward."""

    def __init__(self, model: Any) -> None:
        layers = model.model.layers
        self.num_layers = len(layers)
        self._active = False
        self._records: list[tuple[int, Tensor]] = []
        self._handles = [
            layer.mlp.experts.register_forward_pre_hook(self._make_hook(layer_index))
            for layer_index, layer in enumerate(layers)
        ]

    def _make_hook(self, layer: int):
        def hook(module: Any, args: tuple[Any, ...]) -> None:
            del module
            if not self._active:
                return
            if len(args) < 2 or not isinstance(args[1], Tensor):
                raise RuntimeError("expert hook did not receive top-k indices")
            self._records.append((layer, args[1].detach()))

        return hook

    def begin(self) -> None:
        if self._active:
            raise RuntimeError("route capture is already active")
        self._records.clear()
        self._active = True

    def end(self) -> tuple[tuple[ExpertKey, ...], ...]:
        if not self._active:
            raise RuntimeError("route capture is not active")
        self._active = False
        observed_layers = [layer for layer, _ in self._records]
        expected_layers = list(range(self.num_layers))
        if observed_layers != expected_layers:
            raise RuntimeError(
                f"expected one ordered expert call per layer, observed {observed_layers!r}"
            )
        return tuple(
            tuple(
                (layer, int(expert))
                for expert in torch.unique(top_k_index).cpu().tolist()
            )
            for layer, top_k_index in self._records
        )

    def close(self) -> None:
        self._active = False
        self._records.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

