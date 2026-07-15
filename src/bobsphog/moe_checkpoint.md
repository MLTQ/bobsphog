# `moe_checkpoint.py`

## Purpose

Defines the cold-storage contract for Qwen3.6-35B-A3B without constructing the
full model. It reads checkpoint metadata, maps each routed expert to its packed
safetensor slices, and materializes only one requested expert at a time.

## Components

### `QwenMoeSpec`

- **Does**: Extracts the language-model shape and checkpoint byte count from
  `config.json` and `model.safetensors.index.json`.
- **Interacts with**: Cache sizing and validation code.

### `SafetensorCheckpointIndex`

- **Does**: Resolves a tensor name to its shard and constructs the two packed
  expert tensor names for a given transformer layer.
- **Interacts with**: Hugging Face sharded checkpoint layout.

### `MappedExpertSource`

- **Does**: Uses safetensors slice access to copy one expert's packed gate/up
  and down weights into CPU tensors, optionally pinned for asynchronous H2D
  transfer.
- **Interacts with**: The future bounded CUDA expert cache.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Expert cache | `load(layer, expert)` returns only that expert | Loading the full packed tensor |
| Qwen3.6 layout | Gate/up shape is `[2I, H]`, down shape is `[H, I]` | Different expert packing order |
| Cold residency | Source shards remain files and are opened read-only | Eager checkpoint deserialization |
| Metrics | `bytes_read` counts materialized tensor bytes | Counting whole mapped shard size |

## Notes

Qwen3.6-35B-A3B stores 256 experts per layer in two packed tensors. At BF16,
one expert has `3 * 2048 * 512` parameters, or 6 MiB. Safetensors slicing lets
the source copy that expert without allocating the other 255 experts.

