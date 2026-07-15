# `oracle.py`

## Purpose

Implements the expensive, label-aware A3 upper bound: greedily add the page that
most reduces measured calibration loss, then evaluate that fixed set elsewhere.

## Components

### `SelectionMetrics` / `evaluate_selection`

- **Does**: Reports answer loss and accuracy for one global page set.
- **Interacts with**: `PageCatalog.plan` and `SyntheticBatch`.

### `GreedySelection` / `greedy_oracle_selection`

- **Does**: Exhaustively evaluates every remaining page at each selection step.
- **Rationale**: Establishes whether useful sparse working sets exist before
  judging a learned retriever.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A3 evaluation | Oracle uses calibration labels, never deployment claims | Removing label dependence disclosure |
| Learned-selector comparison | Greedy choices use true marginal loss | Replacing direct model evaluations |
| Budget comparison | Exactly `budget` distinct pages are selected | Early stopping or repeated pages |

## Notes

Greedy selection is not a globally optimal combinatorial oracle and may miss
page synergy. It is an attainable upper-bound baseline, not a deployable policy.
