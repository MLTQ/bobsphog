"""Build and read fixed-offset contiguous Qwen MoE expert-page stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from bobsphog.moe_checkpoint import (
    ExpertSourceStats,
    ExpertWeights,
    QwenMoeSpec,
    SafetensorCheckpointIndex,
)


FORMAT_VERSION = 1
METADATA_NAME = "metadata.json"
DATA_NAME = "experts.bf16"
PARTIAL_DATA_NAME = "experts.bf16.partial"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PageStoreMetadata:
    """Self-describing geometry and checkpoint identity for one page store."""

    format_version: int
    dtype: str
    num_layers: int
    num_experts: int
    experts_per_token: int
    hidden_size: int
    intermediate_size: int
    checkpoint_bytes: int
    page_elements: int
    page_bytes: int
    total_pages: int
    data_bytes: int
    config_sha256: str
    checkpoint_index_sha256: str

    @classmethod
    def from_checkpoint(cls, checkpoint_root: Path) -> PageStoreMetadata:
        config_path = checkpoint_root / "config.json"
        index_path = checkpoint_root / "model.safetensors.index.json"
        spec = QwenMoeSpec.from_files(config_path, index_path)
        total_pages = spec.num_layers * spec.num_experts
        page_elements = spec.expert_parameter_count
        page_bytes = spec.expert_bytes(2)
        return cls(
            format_version=FORMAT_VERSION,
            dtype="bfloat16",
            num_layers=spec.num_layers,
            num_experts=spec.num_experts,
            experts_per_token=spec.experts_per_token,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            checkpoint_bytes=spec.checkpoint_bytes,
            page_elements=page_elements,
            page_bytes=page_bytes,
            total_pages=total_pages,
            data_bytes=total_pages * page_bytes,
            config_sha256=_sha256(config_path),
            checkpoint_index_sha256=_sha256(index_path),
        )

    @classmethod
    def read(cls, store_root: Path) -> PageStoreMetadata:
        path = store_root / METADATA_NAME
        if not path.is_file():
            raise FileNotFoundError(f"page-store metadata is absent: {path}")
        metadata = cls(**json.loads(path.read_text()))
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"unsupported page-store format {self.format_version}")
        if self.dtype != "bfloat16":
            raise ValueError(f"unsupported page-store dtype {self.dtype!r}")
        dimensions = (
            self.num_layers,
            self.num_experts,
            self.experts_per_token,
            self.hidden_size,
            self.intermediate_size,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("page-store dimensions must be positive")
        expected_elements = 3 * self.hidden_size * self.intermediate_size
        expected_pages = self.num_layers * self.num_experts
        if self.page_elements != expected_elements:
            raise ValueError("page_elements disagrees with model dimensions")
        if self.page_bytes != 2 * expected_elements:
            raise ValueError("page_bytes disagrees with BF16 model dimensions")
        if self.total_pages != expected_pages:
            raise ValueError("total_pages disagrees with model dimensions")
        if self.data_bytes != self.total_pages * self.page_bytes:
            raise ValueError("data_bytes disagrees with page geometry")

    def to_spec(self) -> QwenMoeSpec:
        return QwenMoeSpec(
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            experts_per_token=self.experts_per_token,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            checkpoint_bytes=self.checkpoint_bytes,
        )


def write_metadata(store_root: Path, metadata: PageStoreMetadata) -> Path:
    """Atomically publish metadata after its data file is complete."""

    metadata.validate()
    store_root.mkdir(parents=True, exist_ok=True)
    destination = store_root / METADATA_NAME
    temporary = destination.with_suffix(".json.partial")
    temporary.write_text(json.dumps(asdict(metadata), indent=2) + "\n")
    temporary.replace(destination)
    return destination


class ContiguousExpertSource:
    """Load exact experts from one memory-mapped, fixed-offset BF16 file."""

    def __init__(
        self,
        store_root: str | Path,
        expected_spec: QwenMoeSpec | None = None,
        expected_checkpoint: str | Path | None = None,
    ) -> None:
        self.root = Path(store_root).expanduser().resolve()
        self.metadata = PageStoreMetadata.read(self.root)
        self.spec = self.metadata.to_spec()
        if expected_spec is not None:
            expected_geometry = (
                expected_spec.num_layers,
                expected_spec.num_experts,
                expected_spec.experts_per_token,
                expected_spec.hidden_size,
                expected_spec.intermediate_size,
            )
            actual_geometry = (
                self.spec.num_layers,
                self.spec.num_experts,
                self.spec.experts_per_token,
                self.spec.hidden_size,
                self.spec.intermediate_size,
            )
            if actual_geometry != expected_geometry:
                raise ValueError("page-store geometry differs from checkpoint")
        if expected_checkpoint is not None:
            checkpoint_root = Path(expected_checkpoint).expanduser().resolve()
            actual_config_hash = _sha256(checkpoint_root / "config.json")
            actual_index_hash = _sha256(
                checkpoint_root / "model.safetensors.index.json"
            )
            if actual_config_hash != self.metadata.config_sha256:
                raise ValueError("page-store config hash differs from checkpoint")
            if actual_index_hash != self.metadata.checkpoint_index_sha256:
                raise ValueError("page-store index hash differs from checkpoint")
        data_path = self.root / DATA_NAME
        if not data_path.is_file():
            raise FileNotFoundError(f"page-store data is absent: {data_path}")
        actual_bytes = data_path.stat().st_size
        if actual_bytes != self.metadata.data_bytes:
            raise ValueError(
                "page-store data has the wrong size: "
                f"expected={self.metadata.data_bytes}, actual={actual_bytes}"
            )
        self._data = torch.from_file(
            str(data_path),
            shared=False,
            size=self.metadata.total_pages * self.metadata.page_elements,
            dtype=torch.bfloat16,
        )
        self.stats = ExpertSourceStats()

    def _page_index(self, layer: int, expert: int) -> int:
        if not 0 <= layer < self.spec.num_layers:
            raise IndexError("layer is out of range")
        if not 0 <= expert < self.spec.num_experts:
            raise IndexError("expert is out of range")
        return layer * self.spec.num_experts + expert

    def load(
        self,
        layer: int,
        expert: int,
        *,
        pin_memory: bool = False,
    ) -> ExpertWeights:
        started = perf_counter()
        page_index = self._page_index(layer, expert)
        offset = page_index * self.metadata.page_elements
        page = self._data.narrow(0, offset, self.metadata.page_elements).clone()
        if pin_memory:
            page = page.pin_memory()
        gate_elements = 2 * self.spec.intermediate_size * self.spec.hidden_size
        gate_up = page[:gate_elements].view(
            2 * self.spec.intermediate_size,
            self.spec.hidden_size,
        )
        down = page[gate_elements:].view(
            self.spec.hidden_size,
            self.spec.intermediate_size,
        )
        weights = ExpertWeights(gate_up=gate_up, down=down)
        self.stats.loads += 1
        self.stats.bytes_read += weights.parameter_bytes
        self.stats.load_seconds += perf_counter() - started
        return weights

    def describe(self) -> dict[str, Any]:
        return {
            "format_version": self.metadata.format_version,
            "data_path": str(self.root / DATA_NAME),
            "layers": self.spec.num_layers,
            "experts_per_layer": self.spec.num_experts,
            "experts_per_token": self.spec.experts_per_token,
            "expert_bytes_bfloat16": self.metadata.page_bytes,
            "total_pages": self.metadata.total_pages,
            "data_bytes": self.metadata.data_bytes,
            "checkpoint_bytes": self.spec.checkpoint_bytes,
        }


def _write_tensor(output: Any, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    output.write(contiguous.view(torch.uint8).numpy().tobytes())


def build_page_store(
    checkpoint: str | Path,
    store: str | Path,
) -> dict[str, Any]:
    """Create or resume one layer-major, expert-minor exact page store."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to build a page store") from error

    checkpoint_root = Path(checkpoint).expanduser().resolve()
    store_root = Path(store).expanduser().resolve()
    store_root.mkdir(parents=True, exist_ok=True)
    metadata = PageStoreMetadata.from_checkpoint(checkpoint_root)
    index = SafetensorCheckpointIndex(
        checkpoint_root / "model.safetensors.index.json"
    )
    final_path = store_root / DATA_NAME
    partial_path = store_root / PARTIAL_DATA_NAME

    if final_path.exists():
        if final_path.stat().st_size != metadata.data_bytes:
            raise ValueError("existing page-store data has the wrong size")
        write_metadata(store_root, metadata)
        return {
            "status": "already_complete",
            "store": str(store_root),
            **metadata_summary(metadata),
        }
    if partial_path.exists() and partial_path.stat().st_size % metadata.page_bytes:
        raise ValueError("partial page-store size is not page aligned")
    completed_pages = (
        partial_path.stat().st_size // metadata.page_bytes
        if partial_path.exists()
        else 0
    )
    if completed_pages > metadata.total_pages:
        raise ValueError("partial page store is larger than the expected data")

    started = perf_counter()
    with partial_path.open("ab") as output:
        for layer in range(metadata.num_layers):
            first_expert = max(0, completed_pages - layer * metadata.num_experts)
            if first_expert >= metadata.num_experts:
                continue
            gate_name, down_name = index.expert_tensor_names(layer)
            gate_path, down_path = index.expert_shards(layer)
            if not gate_path.is_file() or not down_path.is_file():
                raise FileNotFoundError(f"checkpoint shards for layer {layer} are absent")
            layer_started = perf_counter()
            with safe_open(gate_path, framework="pt", device="cpu") as gate_shard:
                with safe_open(down_path, framework="pt", device="cpu") as down_shard:
                    gate_slice = gate_shard.get_slice(gate_name)
                    down_slice = down_shard.get_slice(down_name)
                    for expert in range(first_expert, metadata.num_experts):
                        gate_up = gate_slice[expert]
                        down = down_slice[expert]
                        expected_gate = (
                            2 * metadata.intermediate_size,
                            metadata.hidden_size,
                        )
                        expected_down = (
                            metadata.hidden_size,
                            metadata.intermediate_size,
                        )
                        if tuple(gate_up.shape) != expected_gate:
                            raise ValueError(
                                f"layer {layer} expert {expert} gate shape is invalid"
                            )
                        if tuple(down.shape) != expected_down:
                            raise ValueError(
                                f"layer {layer} expert {expert} down shape is invalid"
                            )
                        _write_tensor(output, gate_up)
                        _write_tensor(output, down)
            output.flush()
            os.fsync(output.fileno())
            completed_pages = (layer + 1) * metadata.num_experts
            print(
                f"wrote layer {layer + 1}/{metadata.num_layers} "
                f"in {perf_counter() - layer_started:.2f}s",
                file=sys.stderr,
                flush=True,
            )

    actual_bytes = partial_path.stat().st_size
    if actual_bytes != metadata.data_bytes:
        raise RuntimeError(
            f"page-store build ended at {actual_bytes} of {metadata.data_bytes} bytes"
        )
    partial_path.replace(final_path)
    write_metadata(store_root, metadata)
    return {
        "status": "built",
        "store": str(store_root),
        "build_seconds": perf_counter() - started,
        **metadata_summary(metadata),
    }


def metadata_summary(metadata: PageStoreMetadata) -> dict[str, Any]:
    return {
        "total_pages": metadata.total_pages,
        "page_bytes": metadata.page_bytes,
        "data_bytes": metadata.data_bytes,
        "checkpoint_bytes": metadata.checkpoint_bytes,
    }


def inspect_page_store(store: str | Path) -> dict[str, Any]:
    source = ContiguousExpertSource(store)
    return source.describe()


def evict_page_store(store: str | Path) -> dict[str, Any]:
    """Request eviction of clean store pages for a controlled cold-read trial."""

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("this platform does not expose POSIX_FADV_DONTNEED")
    source = ContiguousExpertSource(store)
    data_path = source.root / DATA_NAME
    del source
    with data_path.open("rb") as data:
        os.posix_fadvise(data.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return {
        "status": "eviction_requested",
        "data_path": str(data_path),
        "data_bytes": data_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build or resume a page store")
    build.add_argument("--checkpoint", required=True)
    build.add_argument("--store", required=True)
    inspect = subparsers.add_parser("inspect", help="validate and describe a store")
    inspect.add_argument("--store", required=True)
    evict = subparsers.add_parser(
        "evict", help="request OS-cache eviction before a cold-read trial"
    )
    evict.add_argument("--store", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_page_store(args.checkpoint, args.store)
    elif args.command == "inspect":
        result = inspect_page_store(args.store)
    else:
        result = evict_page_store(args.store)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
