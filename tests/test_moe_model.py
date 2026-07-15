from bobsphog.moe_model import checkpoint_key_to_text_key


def test_checkpoint_key_to_text_key_filters_nonresident_tensors() -> None:
    assert checkpoint_key_to_text_key("model.language_model.norm.weight") == (
        "model.norm.weight"
    )
    assert checkpoint_key_to_text_key(
        "model.language_model.layers.3.self_attn.q_proj.weight"
    ) == "model.layers.3.self_attn.q_proj.weight"
    assert checkpoint_key_to_text_key("lm_head.weight") == "lm_head.weight"
    assert checkpoint_key_to_text_key(
        "model.language_model.layers.0.mlp.experts.down_proj"
    ) is None
    assert checkpoint_key_to_text_key("model.visual.pos_embed.weight") is None
    assert checkpoint_key_to_text_key("mtp.fc.weight") is None
    assert checkpoint_key_to_text_key("unexpected.weight") is None
