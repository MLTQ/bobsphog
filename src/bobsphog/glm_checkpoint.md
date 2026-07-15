# `glm_checkpoint.py`

## Purpose

Defines the cold-storage contract for GLM-5.2. It identifies sparse layers and
materializes one exact routed expert from the checkpoint's three per-expert
safetensor entries without loading neighboring experts.

## Components

### `GlmMoeSpec`

- **Does**: Reads model geometry, sparse-layer IDs, checkpoint bytes, and exact
  BF16 expert/scaffold estimates from config and index metadata.

### `GlmSafetensorCheckpointIndex`

- **Does**: Maps `(layer, expert)` to gate, up, and down tensor names and shards.

### `MappedGlmExpertSource`

- **Does**: Reads three exact expert tensors, validates their shapes/dtypes, and
  packs gate/up into the cache contract used by `CudaExpertCache`.
- **Interacts with**: `ExpertWeights` in `moe_checkpoint.py` and
  `CudaExpertCache` in `expert_cache.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| GLM cache | `load(layer, expert)` returns packed gate/up plus down | Tensor order or shapes |
| Cold residency | Only the requested expert tensors materialize | Loading an entire shard or expert layer |
| Loader sizing | Sparse layers come from `mlp_layer_types` | Treating dense layers as pageable |
| Metrics | One BF16 expert is exactly 72 MiB for GLM-5.2 | Dtype or model geometry changes |

## Notes

The official GLM-5.2 checkpoint stores experts individually rather than in the
packed 3D tensors used by the Transformers runtime. Packing occurs one requested
expert at a time in pinned host memory.
