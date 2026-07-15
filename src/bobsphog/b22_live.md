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
  and pins the working set before decode;
- `background_host` materializes predicted pages into host RAM concurrently
  with full-cache prefill, then promotes the staged bundle for decode;
- `lazy_background_decode` preserves full-cache prefill, protects predicted
  pages only after their first demand load, and stages opportunistically during
  decode without a synchronous promotion barrier;
- `lazy_pin_decode` applies the same demand-time retention without speculative
  host reads, isolating cache-policy value from storage contention.

The eager after-prefill and background-host schedules are controls for prefill
thrash. Their final promotion pause is included in end-to-end time but not TTFT
because the first token already exists. Background staging reports direct reads
and duplicate races separately.

The lazy schedule is the causal B2.3 path for the current predictor, whose
bundle is not known until prefill routing exists. It trades some first-use faults
for zero bundle-loading delay, starts staging exactly once after prefill, and
cancels unused staging when decode ends.

The pin-only schedule is the clean physical control for deciding whether a
prediction helps residency before adding any transfer mechanism.

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
| Background source | One host-staging epoch is joined before teardown | Reusing the source across prompts |
| Parity gate | Forced token path reproduces exact top-1 and expert routes | Approximate expert kernels |

Route failures report the first divergent layer, missing and extra keys, and
whether the difference is ordering-only. This keeps cross-backend drift
diagnosable without weakening the parity gate.

## Interpretation

The before/after phases are synchronous controls; `background_host` is the first
real overlap path. Failure can be split into predictor error, cache churn,
background source contention, final promotion, or decode demand because each is
reported separately.
