from dataclasses import dataclass

import torch

from bobsphog.moe_checkpoint import ExpertWeights
from bobsphog.staged_source import AsyncStagedExpertSource


@dataclass
class FakeSource:
    loads: list[tuple[int, int, bool]]

    def load(self, layer, expert, *, pin_memory=False):
        self.loads.append((layer, expert, pin_memory))
        value = float(layer * 10 + expert)
        return ExpertWeights(
            gate_up=torch.full((2, 2), value),
            down=torch.full((2, 2), value),
        )


def test_background_stage_serves_exact_foreground_weights() -> None:
    base = FakeSource([])
    source = AsyncStagedExpertSource(base)
    source.start([(0, 1), (0, 2)])
    source.finish()

    weights = source.load(0, 2)

    assert weights.gate_up[0, 0].item() == 2.0
    assert source.stats.requested_pages == 2
    assert source.stats.staged_pages == 2
    assert source.stats.staged_hits == 1
    assert source.stats.direct_loads == 0


def test_unpredicted_page_uses_direct_source_load() -> None:
    base = FakeSource([])
    source = AsyncStagedExpertSource(base)
    source.start([])
    source.finish()

    source.load(1, 3)

    assert source.stats.direct_loads == 1
    assert base.loads == [(1, 3, False)]
