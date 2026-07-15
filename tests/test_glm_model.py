from bobsphog.glm_model import checkpoint_key_to_glm_key


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
