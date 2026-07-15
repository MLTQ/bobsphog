import torch

from bobsphog.glm_model import PagedGlmExperts, checkpoint_key_to_glm_key


def test_checkpoint_key_to_glm_key_filters_nonresident_tensors() -> None:
    assert checkpoint_key_to_glm_key("model.embed_tokens.weight") == (
        "model.embed_tokens.weight"
    )
    assert checkpoint_key_to_glm_key(
        "model.layers.3.self_attn.q_a_proj.weight"
    ) == "model.layers.3.self_attn.q_a_proj.weight"
    assert checkpoint_key_to_glm_key("lm_head.weight") == "lm_head.weight"
    assert checkpoint_key_to_glm_key(
        "model.layers.3.mlp.experts.7.down_proj.weight"
    ) is None
    assert checkpoint_key_to_glm_key("model.layers.78.self_attn.q_a_proj.weight") is None
    assert checkpoint_key_to_glm_key("unexpected.weight") is None


def test_paged_glm_experts_exposes_routes_to_optional_cache_observer() -> None:
    class ObservingCache:
        def __init__(self) -> None:
            self.observed = None

        def observe_routes(self, layer, top_k_index):
            self.observed = (layer, top_k_index)

        def schedule(self, keys):
            return tuple(keys)

        def apply_routed(self, layer, hidden_states, top_k_index, top_k_weights):
            return hidden_states

    cache = ObservingCache()
    experts = PagedGlmExperts(3, cache)  # type: ignore[arg-type]
    hidden = torch.randn(2, 4)
    indices = torch.tensor([[1, 2], [2, 3]])
    weights = torch.full((2, 2), 0.5)

    output = experts(hidden, indices, weights)

    assert cache.observed == (3, indices)
    assert output is hidden
