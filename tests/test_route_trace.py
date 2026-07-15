from types import SimpleNamespace

import torch
from torch import nn

from bobsphog.route_trace import ExpertRouteRecorder


class FakeExperts(nn.Module):
    def forward(self, hidden, top_k_index, top_k_weights):
        del top_k_index, top_k_weights
        return hidden


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = FakeExperts()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([FakeLayer(), FakeLayer()]))


def test_route_recorder_captures_sorted_unique_experts_by_layer() -> None:
    model = FakeModel()
    recorder = ExpertRouteRecorder(model)
    hidden = torch.randn(3, 4)
    weights = torch.ones(3, 2)

    recorder.begin()
    model.model.layers[0].mlp.experts(
        hidden, torch.tensor([[7, 2], [2, 5], [7, 5]]), weights
    )
    model.model.layers[1].mlp.experts(
        hidden, torch.tensor([[9, 1], [1, 9], [4, 1]]), weights
    )
    result = recorder.end()
    recorder.close()

    assert result == (
        ((0, 2), (0, 5), (0, 7)),
        ((1, 1), (1, 4), (1, 9)),
    )

