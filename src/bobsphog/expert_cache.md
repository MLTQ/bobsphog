# `expert_cache.py`

## Purpose

Implements a bounded CUDA LRU for exact routed Qwen expert pages. Cold expert
weights are materialized one at a time by `MappedExpertSource`, staged in pinned
CPU memory, and copied on a dedicated CUDA stream.

## Components

### `ExpertCacheStats`

- **Does**: Separates source-load time from CUDA scheduling, transfer bytes,
  hits, misses, evictions, and explicit host waits.

### `CudaExpertCache`

- **Does**: Resolves a requested set of `(layer, expert)` keys, protects the
  active set during LRU eviction, retains pinned staging until asynchronous
  copies complete, and executes cached SwiGLU experts.
- **Interacts with**: `MappedExpertSource` and Qwen top-k router outputs.
- **Loading dynamics**: Sources that expose a fixed `spec.expert_bytes()` are
  streamed one page at a time. Each disk read overlaps the prior asynchronous
  device copy, and completed staging tensors are reaped between pages. This
  avoids materializing an entire high-fanout layer in pinned host memory.

### `apply_routed`

- **Does**: Reproduces the reference Transformers expert loop over the experts
  hit by a flattened token batch, including router weights and `index_add_`.
- **Dtype contract**: Casts fallback FP32 DeltaNet states to the BF16 expert
  weight dtype for the two matrix products, then accumulates contributions in
  the incoming residual dtype, matching Qwen's decorated expert kernel.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Router adapter | Every routed key is scheduled before execution | Implicit blocking page faults |
| Async transfer | Pinned source tensors live until their readiness event completes | Dropping staging immediately |
| Numerical parity | Gate/up packing and weighted accumulation match Qwen | Fused/reordered reductions |
| Capacity | One requested working set fits in `capacity_bytes` | Silent oversubscription |

## Notes

Disk-to-pinned-memory loading remains synchronous, while the preceding device
copy can progress on the cache stream. A background source prefetcher remains a
future optimization.
