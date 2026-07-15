import json

import pytest
import torch

from bobsphog.glm_checkpoint import (
    GlmMoeSpec,
    GlmSafetensorCheckpointIndex,
    MappedGlmExpertSource,
)


def _write_metadata(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "num_hidden_layers": 4,
                "num_nextn_predict_layers": 1,
                "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
                "n_routed_experts": 256,
                "num_experts_per_tok": 8,
                "hidden_size": 6144,
                "moe_intermediate_size": 2048,
            }
        )
    )
    names = GlmSafetensorCheckpointIndex.expert_tensor_names(1, 7)
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1_506_659_919_872},
                "weight_map": {name: f"part-{position}.safetensors" for position, name in enumerate(names)},
            }
        )
    )
    return config_path, index_path


def test_glm_spec_reports_sparse_layout_and_page_size(tmp_path) -> None:
    config_path, index_path = _write_metadata(tmp_path)

    spec = GlmMoeSpec.from_files(config_path, index_path)

    assert spec.sparse_layers == (1, 2, 3)
    assert spec.expert_parameter_count == 37_748_736
    assert spec.expert_bytes() == 75_497_472
    assert spec.routed_expert_bytes == 3 * 256 * 75_497_472
    assert spec.num_mtp_layers == 1
    assert spec.estimated_causal_scaffold_upper_bound_bytes == (
        spec.checkpoint_bytes - 4 * 256 * 75_497_472
    )


def test_glm_index_resolves_individual_expert_tensors(tmp_path) -> None:
    _, index_path = _write_metadata(tmp_path)
    index = GlmSafetensorCheckpointIndex(index_path)
    names = index.expert_tensor_names(1, 7)

    assert names[0] == "model.layers.1.mlp.experts.7.gate_proj.weight"
    assert index.shard_for(names[2]) == tmp_path / "part-2.safetensors"
    with pytest.raises(ValueError):
        index.expert_tensor_names(-1, 0)


def test_mapped_glm_source_packs_gate_then_up(monkeypatch, tmp_path) -> None:
    config_path, index_path = _write_metadata(tmp_path)
    spec = GlmMoeSpec.from_files(config_path, index_path)
    spec = GlmMoeSpec(
        num_layers=spec.num_layers,
        num_mtp_layers=spec.num_mtp_layers,
        sparse_layers=spec.sparse_layers,
        num_experts=spec.num_experts,
        experts_per_token=spec.experts_per_token,
        hidden_size=3,
        intermediate_size=2,
        checkpoint_bytes=spec.checkpoint_bytes,
    )
    source = MappedGlmExpertSource(GlmSafetensorCheckpointIndex(index_path), spec)
    tensors = {
        "gate_proj.weight": torch.full((2, 3), 1.0),
        "up_proj.weight": torch.full((2, 3), 2.0),
        "down_proj.weight": torch.full((3, 2), 3.0),
    }

    monkeypatch.setattr(
        source,
        "_read_tensor",
        lambda name: tensors[name.rsplit(".", 2)[-2] + ".weight"],
    )
    weights = source.load(1, 7)

    torch.testing.assert_close(weights.gate_up[:2], tensors["gate_proj.weight"])
    torch.testing.assert_close(weights.gate_up[2:], tensors["up_proj.weight"])
    torch.testing.assert_close(weights.down, tensors["down_proj.weight"])
    assert source.stats.bytes_read == weights.parameter_bytes


def test_glm_spec_rejects_misaligned_layer_types(tmp_path) -> None:
    config_path, index_path = _write_metadata(tmp_path)
    config = json.loads(config_path.read_text())
    config["mlp_layer_types"] = ["dense"]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="one entry per layer"):
        GlmMoeSpec.from_files(config_path, index_path)
