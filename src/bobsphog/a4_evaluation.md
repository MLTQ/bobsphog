# `a4_evaluation.py`

## Purpose

Compares independent, relationship-aware, learned, and direct-oracle page sets
on held-out compositional examples at identical page counts.

## Components

### `compare_bundle_selectors`

- **Does**: Evaluates random, static SVD, calibrated singleton, calibrated graph
  bundle, learned, learned-plus-graph, and oracle policies.
- **Interacts with**: Domain calibration graph, A3 estimator, and held-out task
  batches.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A4 report | Every non-control policy uses `budget_pages` pages | Unequal selection sizes |
| Scientific protocol | Graph/oracle calibration and evaluation seeds differ | Evaluation-label leakage |
| Relationship test | Singleton and graph policies share singleton utilities | Different calibration bases |

## Notes

The learned policies inspect evaluation inputs but not targets. Calibrated
singleton/graph/oracle policies use labels from a separate same-domain batch.
