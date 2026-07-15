# `smoke.py`

## Purpose

Runs a deterministic, CPU-friendly check of how toy-model output approaches the
full page set as logical resident bytes increase.

## Components

### `run_smoke`

- **Does**: Evaluates base-only through full uniform-prefix plans and reports
  teacher KL plus parameter-byte accounting.
- **Interacts with**: `ToyTransformer`, `PagePlan`, and `PagingTrace`.

### `main`

- **Does**: Exposes the experiment as the `bobsphog-smoke` JSON CLI.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Human and CI smoke checks | Returns one row per prefix budget | Result schema or budget order |
| Future plotting code | JSON fields have stable meanings | Renaming metric keys |

## Notes

The reported residency is logical, not measured process memory. This command is
an implementation sanity check, not evidence that physical paging is faster.
