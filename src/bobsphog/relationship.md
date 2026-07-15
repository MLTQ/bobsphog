# `relationship.py`

## Purpose

Builds the first “holographicity” index: a sparse graph of signed pair
interactions measured on calibration data, plus selectors that form bundles.

## Components

### `SparseRelationshipGraph`

- **Does**: Stores singleton utilities and sparse symmetric interaction edges.
- **Interacts with**: Calibrated and learned graph-guided selectors.

### `build_relationship_graph`

- **Does**: Measures every singleton, evaluates pairs among the strongest
  singleton candidates, and retains each page's strongest absolute interactions.
- **Rationale**: A sparse top-neighbor graph avoids an $N^2$ resident index.

### `graph_guided_learned_selection`

- **Does**: Adds calibration-graph synergy to the learned counterfactual score
  after each page selection.
- **Interacts with**: A3 estimator, current hidden query, and resident mask.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A4 evaluation | Edge weights are excess pair utility over singleton sum | Synergy definition changes |
| Graph routing | Graph is built only from calibration batches | Evaluation-label leakage |
| Equal-budget policies | Selectors return unique global page IDs | Duplicate or variable-count bundles |

## Notes

Positive edges represent superadditive complementarity; negative edges represent
redundancy or substitution and penalize loading overlapping pages. The graph is
domain-conditioned through an explicit domain choice and uses labels offline.
