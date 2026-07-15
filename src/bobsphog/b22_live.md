# `b22_live.py`

## Purpose

Runs the best held-out B2.2 route predictor through the exact file-backed Qwen
model. It is the bridge from offline route recall to measured page faults,
latency, peak VRAM, output parity, and route parity.

## Design

1. Fit predictor state on the 40 training prompts.
2. Tune only on the eight validation prompts.
3. Select a fixed, equal-per-layer bundle for one held-out test prompt.
4. Pin and prefetch that bundle into a 2,560-page physical cache.
5. Execute the resident-model token path through exact paged experts.
6. Verify every live route group and every top-1 decision against the recorded
   full-resident trace.

The default is the validation-selected three-nearest-neighbor predictor with a
2,048-page bundle, leaving 512 pages for ordinary LRU demand traffic.

`--prefetch-phase` compares two schedules:

- `before_prefill` makes the predicted bundle available immediately but limits
  prefill to the residual cache;
- `after_prefill` gives prefill the full LRU cache, then synchronously reshapes
  and pins the working set before decode.

The latter is a scheduling control for prefill thrash. Its prefetch pause is
included in end-to-end time but not TTFT because the first token already exists.

When training traces and deployment use different numerical backends,
`--execution-trace` supplies a B2 route captured on the deployment GPU. The
bundle is still predicted only from the ROCm training corpus, while route,
token, fault, and quality parity are judged against the CUDA-native control.
Both backend-specific offline scores are retained in the result.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| B2.2 corpus | Full prompt text, token IDs, prefill routes, and ordered decode routes | Removing trace fields |
| Optional B2 trace | Prompt text exactly matches the corpus case | Comparing different prompts across backends |
| Predictor | Training/validation/test split is fixed before fitting | Tuning on held-out test routes |
| Physical cache | Bundle plus largest unpinned atomic layer group fits | Oversized bundle |
| Parity gate | Forced token path reproduces exact top-1 and expert routes | Approximate expert kernels |

Route failures report the first divergent layer, missing and extra keys, and
whether the difference is ordering-only. This keeps cross-backend drift
diagnosable without weakening the parity gate.

## Interpretation

This is a synchronous prefetch control. A speedup demonstrates that route
prediction is useful before implementing background I/O. Failure can be split
into predictor error, cache churn, or prefetch overhead because each component
is reported separately.
