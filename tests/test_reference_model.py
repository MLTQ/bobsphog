from bobsphog.reference_model import checkpoint_key_to_reference_text_key


def test_reference_mapping_includes_experts_but_excludes_non_text_weights() -> None:
    assert checkpoint_key_to_reference_text_key(
        "model.language_model.layers.0.mlp.experts.down_proj"
    ) == "model.layers.0.mlp.experts.down_proj"
    assert checkpoint_key_to_reference_text_key(
        "model.language_model.layers.3.self_attn.q_proj.weight"
    ) == "model.layers.3.self_attn.q_proj.weight"
    assert checkpoint_key_to_reference_text_key("lm_head.weight") == "lm_head.weight"
    assert checkpoint_key_to_reference_text_key("model.visual.pos_embed.weight") is None
    assert checkpoint_key_to_reference_text_key("mtp.fc.weight") is None
    assert checkpoint_key_to_reference_text_key("unexpected.weight") is None
