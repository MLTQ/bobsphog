# `a3_evaluation.py`

## Purpose

Runs the decisive A3 equal-page-budget comparison on held-out examples for each
domain.

## Components

### `compare_selectors`

- **Does**: Compares random subsets, static SVD order, calibration-label greedy
  oracle, and label-free learned greedy selection against base/full controls.
- **Interacts with**: `PageCatalog`, oracle, retriever, and synthetic task.
- **Rationale**: The oracle is selected on a separate calibration batch, while
  all policy quality is measured on the same held-out evaluation batch.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `a3.py` | Rows contain equal-budget metrics and selected page names | Result schema changes |
| Scientific comparison | All non-base/full selectors use exactly `budget_pages` | Unequal page counts |
| Oracle interpretation | Calibration and evaluation batches use different seeds | Selecting oracle on evaluation labels |

## Notes

All current pages are the same byte size. Random results average several sets;
static, oracle, and learned results each use one deterministic set per seed.
