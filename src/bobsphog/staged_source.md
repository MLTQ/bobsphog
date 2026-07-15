# `staged_source.py`

## Purpose

Provides the first B2.3 transfer-overlap mechanism. Predicted exact expert pages
are materialized into host RAM in a background thread while the normal GPU
cache executes prompt prefill, then foreground promotion can consume staged
weights without repeating safetensor reads.

## Components

### `StagedSourceStats`

- **Does**: Separates staged hits, unavoidable direct reads, duplicate races,
  bytes staged, background duration, and foreground wait.

### `AsyncStagedExpertSource`

- **Does**: Wraps an expert source with one ordered background staging pass and
  a thread-safe foreground `load` interface.
- **Rationale**: Host staging leaves the entire GPU expert cache available to
  prefill, unlike pinning the decode bundle before prefill.
- **Race policy**: Foreground demand does not wait during prefill; it performs a
  direct read and marks the background copy disposable. During final promotion,
  callers may wait for an already-inflight page rather than duplicate it.
- **Lifecycle**: `cancel` stops admission of new pages after the current read;
  this prevents decode-only staging from continuing after the response ends.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CUDA expert cache | `load(layer, expert, pin_memory=...)` returns exact weights | Different source protocol |
| B2.3 live runner | `start` is called once and `close` joins the worker | Multiple staging epochs |
| Memory bound | One predicted bundle may occupy host RAM | Unbounded multi-prompt staging |
| Exactness | Staged and direct paths return identical tensors | Transforming staged weights |

## Notes

This stage overlaps NVMe/page-cache to CPU materialization, not final H2D copy.
GPU-side staging and multiple source workers should be attempted only after this
control demonstrates useful overlap without storage contention.
