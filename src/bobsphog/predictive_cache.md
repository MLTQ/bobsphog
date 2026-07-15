# `predictive_cache.py`

## Purpose

Implements the B2.2 deployment policy: a prompt-conditioned expert bundle is
pinned for the decode epoch while all remaining expert pages share a bounded
LRU cache.

## Components

### `PinnedCudaExpertCache`

- **Does**: Excludes predicted keys from LRU eviction and synchronously
  prefetches the bundle in layer groups.
- **Rationale**: Separates sequence-level prediction from token-level demand
  paging without changing exact expert computation.
- **Constraint**: The pinned set and the largest simultaneous non-pinned layer
  request must fit together.

### `TracingPinnedCudaExpertCache`

- **Does**: Records demand route groups while excluding explicit prefetch calls.
- **Rationale**: Live B2.2 runs can verify that the forced token path reproduces
  the resident-model route exactly.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Mapped source | Exposes `spec.expert_bytes()` | Sources without checkpoint spec metadata |
| Paged Qwen | Calls `schedule` once per MoE layer | Fused cross-layer scheduling |
| Live runner | Prefetch groups are layer-bounded | One monolithic group larger than cache |

## Notes

Prefetch is synchronous in this control. It measures whether prediction reduces
demand faults before asynchronous overlap or superpage I/O is introduced.
