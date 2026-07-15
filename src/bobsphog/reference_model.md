# `reference_model.py`

## Purpose

Loads the complete Qwen3.6 language model onto one CUDA/HIP device. It is the
full-residency control for measuring numerical and capability parity against
the paged implementation without loading vision or MTP tensors.

## Components

### `checkpoint_key_to_reference_text_key`

- **Does**: Maps language and output-head checkpoint names into the text-only
  Transformers namespace while retaining routed-expert tensors.

### `ReferenceLoadSummary`

- **Does**: Reports the full reference loader's tensor, byte, shard, and timing
  totals.

### `load_reference_qwen`

- **Does**: Builds the text model on the meta device and streams every language
  tensor, including all routed experts, directly into the target device.
- **Interacts with**: `run_b1_capability` in `b1_capability.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Capability control | Routed experts are resident and exact | Filtering `.mlp.experts.*` |
| Memory comparison | Vision and MTP weights remain excluded | Loading the multimodal wrapper |
| Numerical parity | Checkpoint dtypes are preserved | Implicit dtype conversion |
| Load safety | Missing shards fail before evaluation | Network fallback or partial loading |

## Notes

This loader intentionally favors a transparent control path over low-memory
loading. It should only run where the full text model and temporary loader
workspace fit together.
