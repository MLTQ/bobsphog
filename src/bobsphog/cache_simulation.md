# `cache_simulation.py`

## Purpose

Replays expert-page request traces without loading tensors. The simulations
separate routing locality from storage and GPU timing and quantify how much of
the observed transfer volume is caused by cache policy.

## Components

### `CacheSimulationResult`

- **Does**: Stores request, hit, miss, and eviction counts and derives hit rate
  and transferred bytes for a known page size.

### `simulate_grouped_lru`

- **Does**: Reproduces the production cache's LRU order while protecting every
  page requested by the current layer from eviction, including the expert-touch
  order applied after scheduling.
- **Rationale**: Expert requests for one layer form an atomic working set; all
  pages must coexist until that layer executes.

### `simulate_grouped_belady`

- **Does**: Evicts the non-protected page whose next request is furthest in the
  future.
- **Rationale**: This is an offline-optimal replacement lower bound on misses,
  while still respecting atomic layer working sets. It is not an implementable
  online policy or a claim that transfer latency can always be hidden.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| B2 analysis | Request groups preserve execution order | Sorting groups |
| Layer execution | Capacity fits the largest atomic group | Partial-group simulation |
| Transfer estimate | Every miss transfers one equal-size page | Variable page sizes |
