# `paging.py`

## Purpose

Defines logical page-selection inputs and execution traces independently of the
neural architecture. Physical CPU/GPU movement will build on these contracts.

## Components

### `PageEvent`

- **Does**: Records selected IDs and logical bytes for one paged layer call.
- **Interacts with**: `PagedLinear.forward` in `decomposition.py`.

### `PagingTrace`

- **Does**: Accumulates events and summarizes selected page count and bytes.
- **Interacts with**: `ToyTransformer.forward` callers and the smoke experiment.

### `PagePlan`

- **Does**: Resolves stable layer IDs into page selections.
- **Interacts with**: Every paged MLP projection in `model.py`.
- **Rationale**: A data-only plan makes full, base-only, prefix, random-dropout,
  and future retriever policies comparable through the same model path.

### `PagePlan.random_dropout`

- **Does**: Samples reproducible structured whole-page dropout.
- **Interacts with**: Future multi-budget training loops.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `model.py` | Missing selections follow `default` | Default resolution semantics |
| Tests and experiments | `uniform_prefix` selects SVD pages from index zero | Reordering page priorities |
| Metrics | Trace byte totals describe logical selected tensors | Redefining byte accounting |

## Notes

Logical page bytes are not yet measured peak memory: all parameters still live
in the model. Keeping that distinction explicit prevents simulated residency
from being reported as a hardware result.
