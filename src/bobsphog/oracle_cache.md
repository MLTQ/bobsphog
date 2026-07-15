# `oracle_cache.py`

## Purpose

Provides the deliberately clairvoyant cache used by B2.1 to measure the upper
bound from perfect route knowledge. It changes retention and load timing while
executing the original exact expert tensors and router decisions.

## Components

### `FutureUseOracle`

- **Does**: Validates each live atomic layer request against a recorded trace,
  consumes the current occurrence, and exposes every resident page's next use.
- **Rationale**: Evicting the page used furthest in the future implements the
  group-aware Belady control in the live CUDA cache.

### `OraclePrefetchStats`

- **Does**: Separates proactive requests, hits, misses, evictions, bytes, source
  loading, CUDA waits, and elapsed time from demand-cache counters.

### `OracleExpertCache`

- **Does**: Overrides LRU eviction with future-aware replacement and optionally
  preloads a complete upcoming token bundle without consuming demand requests.
- **Interacts with**: `CudaExpertCache` for exact expert execution and
  `MappedExpertSource` for file-backed page materialization.
- **Rationale**: Token bundles are protected atomically during prefetch, then
  transferred one layer at a time to bound pinned host staging.

### `set_pinned_keys`

- **Does**: Marks an oracle-selected prompt or sequence bundle as non-evictable
  while transient layer pages share the remaining capacity.
- **Rationale**: Separates the complete decode union from one-use prefill pages
  in the prompt-level oracle control.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| B2.1 runner | Live request groups exactly match the recorded trace | Independent sampling or reordered routing |
| Oracle eviction | Current trace occurrence is consumed before eviction | Calling base `schedule` directly |
| Token prefetch | One token's unique pages fit in cache | Capacity below 320 Qwen pages |
| Prompt bundle | Pinned union plus the largest active layer fits | Oversubscribed pinned set |
| Capability control | Expert math remains inherited and exact | Approximate or reordered expert execution |

## Notes

This is an experimental upper bound, not a deployable online policy. Prefetch is
synchronous in B2.1; the benchmark separately reports serial wall time and the
ideal pipeline time if those measured prefetch operations were overlapped with
the preceding token's computation.
