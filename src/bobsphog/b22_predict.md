# `b22_predict.py`

## Purpose

Evaluates whether prefill routing predicts held-out decode working sets before a
learned neural controller is introduced. It provides random, static-frequency,
prefill-reuse, nearest-neighbor, conditional relationship-index, and per-prompt
oracle baselines at equal page budgets.

## Components

### `RouteExample` / `load_trace_examples`

- **Does**: Converts serialized routes into per-layer prefill indicators,
  decode request counts, and unique-union targets.

### `select_equal_layer_budget`

- **Does**: Applies a deterministic near-equal page quota to every layer.
- **Rationale**: Each Qwen layer requests eight experts per decode token; equal
  quotas prevent a global score from starving less frequent layers.

### `PredictorSuite`

- **Does**: Fits global request frequencies and a per-layer
  `P(decode expert | prefill expert)` relationship tensor, and scores held-out
  prompts with six baselines.
- **Rationale**: The conditional tensor is the first concrete implementation of
  the proposed sparse parameter-relationship index.

### `evaluate_selection`

- **Does**: Reports unique-union recall/precision, request-weighted hit
  fraction, and unavoidable late unique pages/bytes for one fixed bundle.

### `simulate_pinned_bundle_lru`

- **Does**: Replays the ordered decode route with the predicted bundle pinned
  and the remainder of a fixed-capacity cache managed by LRU.
- **Rationale**: Converts static route recall into estimated physical page
  faults and transfer bytes under the same 2,560-page budget used by B2/B2.1.
- **Conservative assumption**: The transient LRU begins empty after bundle
  loading; useful non-pinned pages left by prefill are not credited.
- **Transfer accounting**: Reports both decode-fault bytes and cold total bytes
  including the one-time predicted-bundle prefetch. These are distinct because
  a large contiguous prefetch may be faster than repeated demand faults even
  when it moves more total data.

### `tune_hyperparameters`

- **Does**: Chooses conditional-prior strength and nearest-neighbor count using
  validation request-hit fraction only.

### `run_b22_predict`

- **Does**: Fits on 40 training prompts and reports validation and held-out test
  aggregates by budget, method, and domain.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Corpus loader | Every prompt has fixed layer count and valid expert IDs | Ragged or out-of-range traces |
| Equal comparison | All methods receive the identical per-layer page quota | Method-specific budgets |
| Test validity | Hyperparameters are selected only on validation prompts | Tuning against test metrics |
| Live follow-up | Selected pages map directly to `(layer, expert)` cache keys | Cross-layer page encoding |

## Notes

Late unique pages remain a lower bound. The pinned-bundle LRU replay estimates
repeated faults but does not model overlapped I/O or kernel time. A live run is
still required to measure wall-clock latency and exact output directly.
