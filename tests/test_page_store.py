import json

import pytest
import torch

from bobsphog.page_store import (
    DATA_NAME,
    ContiguousExpertSource,
    PageStoreMetadata,
    write_metadata,
)


def _metadata() -> PageStoreMetadata:
    return PageStoreMetadata(
        format_version=1,
        dtype="bfloat16",
        num_layers=2,
        num_experts=3,
        experts_per_token=1,
        hidden_size=2,
        intermediate_size=1,
        checkpoint_bytes=1000,
        page_elements=6,
        page_bytes=12,
        total_pages=6,
        data_bytes=72,
        config_sha256="config",
        checkpoint_index_sha256="index",
    )


def _write_store(tmp_path) -> None:
    metadata = _metadata()
    pages = [
        torch.arange(6, dtype=torch.float32).add(10 * page).to(torch.bfloat16)
        for page in range(metadata.total_pages)
    ]
    data = torch.cat(pages).view(torch.uint8).numpy().tobytes()
    (tmp_path / DATA_NAME).write_bytes(data)
    write_metadata(tmp_path, metadata)


def test_contiguous_source_loads_fixed_offset_page(tmp_path) -> None:
    _write_store(tmp_path)
    source = ContiguousExpertSource(tmp_path)

    weights = source.load(1, 1)

    assert weights.gate_up.shape == (2, 2)
    assert weights.down.shape == (2, 1)
    assert weights.gate_up.flatten().tolist() == [40.0, 41.0, 42.0, 43.0]
    assert weights.down.flatten().tolist() == [44.0, 45.0]
    assert weights.parameter_bytes == 12
    assert source.stats.loads == 1
    assert source.stats.bytes_read == 12


def test_contiguous_source_rejects_invalid_coordinates(tmp_path) -> None:
    _write_store(tmp_path)
    source = ContiguousExpertSource(tmp_path)

    with pytest.raises(IndexError):
        source.load(-1, 0)
    with pytest.raises(IndexError):
        source.load(0, 3)


def test_contiguous_source_rejects_truncated_data(tmp_path) -> None:
    metadata = _metadata()
    (tmp_path / DATA_NAME).write_bytes(b"\0" * (metadata.data_bytes - 2))
    write_metadata(tmp_path, metadata)

    with pytest.raises(ValueError, match="wrong size"):
        ContiguousExpertSource(tmp_path)


def test_metadata_rejects_inconsistent_page_geometry(tmp_path) -> None:
    payload = {**_metadata().__dict__, "page_bytes": 10}
    (tmp_path / "metadata.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="page_bytes"):
        ContiguousExpertSource(tmp_path)


def test_contiguous_source_rejects_checkpoint_hash_mismatch(tmp_path) -> None:
    _write_store(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("different config")
    (checkpoint / "model.safetensors.index.json").write_text("different index")

    with pytest.raises(ValueError, match="config hash"):
        ContiguousExpertSource(tmp_path, expected_checkpoint=checkpoint)
