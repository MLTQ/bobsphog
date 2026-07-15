# `physical_cache.py`

## Purpose

Implements the first real A5 residency runtime: exact optional factors live in
pinned host memory and only a bounded prompt working set occupies CUDA memory.

## Components

### `CacheStats`

- **Does**: Counts requests, hits, misses, evictions, transferred bytes, and
  prefetch wall time.

### `PhysicalPageCache`

- **Does**: Moves the resident model to CUDA, offloads source pages to pinned
  CPU tensors, binds itself as page provider, and owns an LRU GPU cache.
- **Interacts with**: `PagedLinear.set_page_provider` and `PagePlan`.

### `PhysicalPageCache.schedule`

- **Does**: Resolves a prompt plan, evicts unprotected LRU pages, launches
  non-blocking copies on a dedicated CUDA stream, records one readiness event
  per page, and returns without a host-side synchronization.

### `PhysicalPageCache.wait` and `prepare`

- **Does**: `wait` provides an explicit host barrier. `prepare` preserves the
  original synchronous convenience contract by composing `schedule` and
  `wait`.

### `PhysicalPageCache.apply`

- **Does**: Adds a device-stream dependency on the requested page's readiness
  event, then executes cached factors without materializing a dense matrix.
  The host can therefore enqueue model compute while page copies are in flight.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A5 benchmark | Source pages are physically on pinned CPU after construction | Leaving sources in VRAM |
| Model forward | Every selected page was scheduled before execution | Automatic page-fault semantics |
| Metrics | `bytes_transferred` counts exact H2D factor bytes | Compression or staging changes |
| Cache policy | Requested pages are protected during preparation | Evicting the active working set |

## Notes

- This is inference-only; attaching the cache disables gradients.
- Scheduling order follows transformer execution order, allowing transfers for
  later layers to overlap compute in earlier layers.
- Eviction waits if a victim's transfer is still in flight, preventing reuse of
  storage whose asynchronous copy has not completed.
- CUDA reserved memory may stay high because of allocator caching; physical
  comparisons use `memory_allocated` after `empty_cache`.
