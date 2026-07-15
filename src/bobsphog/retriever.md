# `retriever.py`

## Purpose

Defines and trains the first deployable-style A3 selector: a small network that
predicts the marginal value of fetching an omitted page.

## Components

### `CounterfactualUtilityEstimator`

- **Does**: Combines projected query state, candidate embedding, and mean
  resident-page embedding to predict raw loss improvement.
- **Interacts with**: `UtilityExamples` and `PageCatalog` masks.

### `train_utility_estimator`

- **Does**: Fits standardized utility regression and reports held-out RMSE,
  correlation, and improvement-sign accuracy.

### `learned_greedy_selection`

- **Does**: Recomputes the resident query after each chosen page and greedily
  selects the highest predicted remaining utility.
- **Interacts with**: `ToyTransformer.hidden_states` and estimator scores.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A3 evaluation | Estimator input never includes evaluation labels | Adding targets to features |
| Utility training | Forward outputs raw loss-improvement units | Returning normalized values |
| Learned selector | Exactly `budget` unique pages are selected | Early stopping or duplicates |

## Notes

The resident set is summarized by a mean embedding, which cannot represent all
higher-order interactions. This is deliberately the smallest useful estimator.
