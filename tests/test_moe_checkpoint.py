import json

import pytest

from bobsphog.moe_checkpoint import (
    MappedExpertSource,
    QwenMoeSpec,
    SafetensorCheckpointIndex,
)


def _write_metadata(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 40,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "hidden_size": 2048,
                    "moe_intermediate_size": 512,
                }
            }
        )
    )
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 71_903_645_408},
                "weight_map": {
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": "one.safetensors",
                    "model.language_model.layers.0.mlp.experts.down_proj": "two.safetensors",
                },
            }
        )
    )
    return config_path, index_path


def test_qwen_moe_spec_and_expert_shards(tmp_path) -> None:
    config_path, index_path = _write_metadata(tmp_path)
    spec = QwenMoeSpec.from_files(config_path, index_path)
    index = SafetensorCheckpointIndex(index_path)

    assert spec.num_layers == 40
    assert spec.experts_per_token == 8
    assert spec.expert_parameter_count == 3_145_728
    assert spec.expert_bytes() == 6_291_456
    assert index.expert_shards(0) == (
        tmp_path / "one.safetensors",
        tmp_path / "two.safetensors",
    )


def test_mapped_source_rejects_invalid_coordinates_before_read(tmp_path) -> None:
    config_path, index_path = _write_metadata(tmp_path)
    spec = QwenMoeSpec.from_files(config_path, index_path)
    source = MappedExpertSource(SafetensorCheckpointIndex(index_path), spec)

    with pytest.raises(IndexError):
        source.load(-1, 0)
    with pytest.raises(IndexError):
        source.load(0, 256)
    with pytest.raises(ValueError):
        SafetensorCheckpointIndex.expert_tensor_names(-1)
