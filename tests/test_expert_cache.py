import pytest
import torch
from torch.nn import functional as F

from bobsphog.expert_cache import CudaExpertCache
from bobsphog.moe_checkpoint import ExpertWeights


class FakeExpertSource:
    def __init__(self, weights):
        self.weights = weights

    def load(self, layer, expert, *, pin_memory=False):
        weights = self.weights[(layer, expert)]
        gate_up = weights.gate_up.clone()
        down = weights.down.clone()
        if pin_memory:
            gate_up = gate_up.pin_memory()
            down = down.pin_memory()
        return ExpertWeights(gate_up, down)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_expert_cache_matches_direct_routed_execution() -> None:
    torch.manual_seed(72)
    device = torch.device("cuda:0")
    weights = {
        (0, expert): ExpertWeights(
            gate_up=torch.randn(8, 6),
            down=torch.randn(6, 4),
        )
        for expert in (3, 9)
    }
    source = FakeExpertSource(weights)
    capacity = sum(value.parameter_bytes for value in weights.values())
    cache = CudaExpertCache(source, device=device, capacity_bytes=capacity)
    keys = ((0, 3), (0, 9))
    cache.schedule(keys)
    hidden = torch.randn(3, 6, device=device)
    top_k_index = torch.tensor([[3, 9], [9, 3], [3, 9]], device=device)
    top_k_weights = torch.tensor(
        [[0.7, 0.3], [0.8, 0.2], [0.55, 0.45]],
        device=device,
    )

    expected = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for position in range(2):
            expert = int(top_k_index[token, position])
            page = weights[(0, expert)]
            gate, up = F.linear(hidden[token], page.gate_up.to(device)).chunk(2)
            output = F.linear(F.silu(gate) * up, page.down.to(device))
            expected[token] += output * top_k_weights[token, position]
    actual = cache.apply_routed(0, hidden, top_k_index, top_k_weights)
    torch.testing.assert_close(actual, expected)
    cold = cache.snapshot()
    cache.schedule(keys)
    warm = cache.snapshot()

    assert cold.misses == 2
    assert warm.hits - cold.hits == 2
