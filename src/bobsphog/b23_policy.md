# `b23_policy.py`

## Purpose

Implements B2.3 confidence calibration and adaptive prompt-bundle sizing. It
predicts incremental decode-fault savings over the warm post-prefill LRU state,
then chooses no pinning or a fixed candidate budget using a conservative lower
confidence bound and an exposed-prefetch cost.

The evaluated physical policy is lazy retention: predicted pages are not
preloaded, so confidence labels include only the effect of protecting a page
after its first genuine request.

## Components

### `route_groups` / `simulate_decode_misses`

- **Does**: Converts a route example into physical request groups and replays
  pinned prefetch, prefill, and decode through the production-equivalent cache.
- **Interacts with**: `simulate_phased_pinned_lru` in `cache_simulation.py`.

### `confidence_features`

- **Does**: Computes deployable signals from prefill similarity, neighbor
  agreement, predicted score mass, prefill coverage, and neighbor fault savings.
- **Constraint**: Never reads the query's decode target when constructing
  features.

### `fit_ridge` / `RidgeCalibrator`

- **Does**: Fits a standardized ridge model that predicts decode-fault savings
  in page units.
- **Rationale**: The small transparent calibrator is auditable before a neural
  controller is justified.

### `run_b23_policy`

- **Does**: Builds leave-one-out training labels, chooses ridge strength on the
  validation split, calibrates an absolute-error confidence bound, and evaluates
  adaptive budgets only on held-out test prompts.
- **Reports**: A confidence/cost sensitivity grid and an oracle adaptive upper
  bound so conservatism can be separated from predictor ranking quality.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| B2.2 corpus | Ordered prefill/decode routes and fixed splits | Missing route groups |
| Confidence validity | Query decode targets are used only after budget choice | Target-derived features |
| Adaptive policy | Candidate budgets start with zero and stay below cache capacity | Removing no-prefetch control |
| Live B2.3 | Chosen budget and method map to equal per-layer page selection | Different page quota semantics |

## Notes

Lazy pinning defaults `exposed_prefetch_cost_per_page` to zero because it moves
no speculative page; lost transient capacity is already reflected in simulated
misses. Nonzero values model adding optional background staging, where `0.10`
means only one tenth of a demand-fault cost remains exposed after overlap.
B2.3 reports sensitivity rather than treating that systems value as learned.
The default sweep extends to 2,520 pages, leaving a 40-page atomic safety margin
inside the 2,560-page cache.
