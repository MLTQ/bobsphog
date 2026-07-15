# B2 results: cache scaling helps, sequence-aware retention matters more

## Result

B2 swept the exact Qwen3.6-35B-A3B expert cache on the RTX 4090 while every
capacity followed the same 16-token output path. Increasing the cache from 320
to 2,560 six-MiB pages:

- increased measured peak CUDA allocation from 6.49 to 19.62 GiB;
- reduced decode misses from 3,116 to 1,296;
- reduced decode transfer from 19.60 to 8.15 GB, or **58.4%**;
- raised warm steady decode from 0.527 to 0.740 token/s, or approximately
  **40.5%**; and
- preserved the exact output tokens, top-1 predictions, and routing trace.

Dynamic residency therefore produces a real memory/rate frontier. Cache size
alone is not enough, however: even the 2,560-page run is far below the 2 token/s
B2 performance gate and the full-resident Strix control's 6.79 token/s.

## Experiment design

The benchmark formats the same 28-token prompt used by B1.6, performs one real
prefill, then manually decodes 15 additional forwards to produce 16 output
tokens. Every forward is synchronized for timing and preserves Qwen's real
KV/recurrent state.

The first capacity establishes the greedy token path. All later capacities are
forced along those token IDs while their own top-1 predictions are recorded.
This prevents generation divergence from confounding routing and speed.

Two reciprocal four-point passes bracket operating-system file-cache effects:

1. 2,048 → 1,280 → 640 → 320 pages, which favors the smaller capacities with
   later host-cache warmth.
2. 320 → 640 → 1,280 → 2,048 pages after the first pass, which favors the larger
   capacities.

A final 2,560-page run tests the original upper endpoint after verifying 4090
headroom. Algorithmic misses and bytes are deterministic across passes; elapsed
source loading remains sensitive to host cache and shard locality.

## Measured frontier

Token-rate ranges below contain both reciprocal passes. The 2,560-page point is
a single warm measurement.

| Expert pages | Page cache (GiB) | Peak CUDA (GiB) | Decode hit rate | Decode transfer (GB) | Decode token/s |
|-------------:|-----------------:|----------------:|----------------:|---------------------:|---------------:|
| 320 | 1.875 | 6.49 | 35.1% | 19.60 | 0.527–0.539 |
| 640 | 3.750 | 8.37 | 43.0% | 17.22 | 0.522–0.567 |
| 1,280 | 7.500 | 12.12 | 57.1% | 12.95 | 0.534–0.608 |
| 2,048 | 12.000 | 16.62 | 66.4% | 10.15 | 0.588–0.605 |
| 2,560 | 15.000 | 19.62 | 73.0% | 8.15 | 0.740 |

The larger cache produces a monotonic miss and transfer reduction, but token
rate is noisy below 2,560 pages because individual safetensor slices are loaded
synchronously and host-cache state dominates per-page latency. From the warm
320-page control to 2,560 pages, source-loading time fell only 30.9% despite a
58.4% byte reduction.

TTFT remained between 35.3 and 39.1 seconds. Prefill touches 3,108 distinct
expert pages once each, so every tested cache incurs the same 3,108 compulsory
prefill misses. More reactive capacity cannot improve this; prompt-time
prediction and overlapping prefetch are required.

## Working-set locality

Each one-token decode forward requests exactly 320 pages: eight experts in each
of 40 layers. Across all 15 decode forwards, the cumulative union is only 2,140
pages, or 12.54 GiB. A 2,048-page cache can physically hold **95.7%** of this
sequence's entire decode union.

That strong sequence-level locality is not visible one token at a time. The
previous token's page set recalls only 37.6% of the next token's pages on
average, ranging from 12.2% to 56.2% across transitions. Per-layer mean recall
ranges from 3.6% in layer 0 to 56.2% in layer 20.

The implication is favorable to the spongiform design:

> The useful unit is a prompt-conditioned multi-token page bundle, not the last
> token's 320-page working set.

## LRU versus an offline-optimal cache

The captured router trace was replayed through the production group-protected
LRU policy and a group-aware Belady policy that knows the complete future. The
simulation preserves the requirement that all experts for one layer coexist.
Production LRU and simulated LRU miss counts match exactly at every capacity.

Because prefill's 3,108 misses are compulsory, the table subtracts that phase
and reports decode-only policy opportunity:

| Pages | Reactive LRU misses | Offline-optimal misses | Avoidable LRU misses | Optimal transfer (GB) |
|------:|--------------------:|-----------------------:|---------------------:|----------------------:|
| 320 | 3,116 | 2,226 | 28.6% | 14.00 |
| 640 | 2,737 | 1,550 | 43.4% | 9.75 |
| 1,280 | 2,058 | 910 | 55.8% | 5.73 |
| 2,048 | 1,613 | 739 | 54.2% | 4.65 |
| 2,560 | 1,296 | 739 | 43.0% | 4.65 |

At 2,048 pages, future-aware retention cuts predicted decode transfer from
10.15 to 4.65 GB. At 2,560 pages the complete 2,140-page decode union fits, but
reactive LRU still reloads pages because one-time prefill pages initially occupy
the wrong slots. The remaining problem is page selection and retention, not
physical capacity.

Offline optimal is an upper bound, not an online implementation. It does,
however, define a concrete target for a prompt-conditioned page predictor.

## Capability control

Every capacity produced the same token IDs and route trace:

```text
The sky appears blue due to a phenomenon called **Rayleigh scattering**.

Here
```

All forced-path runs had 100% top-1 agreement with the reference capacity. B2
therefore changes only residency and timing; no approximation, quantization, or
expert substitution was introduced.

## Gate decision

| Gate | Result | Decision |
|------|--------|----------|
| Exact capability parity | Exact IDs, top-1, and routes | Pass |
| Physical memory/rate frontier | 6.49–19.62 GiB, monotonic miss reduction | Pass |
| Steady decode > 2 token/s | Best 0.740 token/s | Fail |
| TTFT < 10 seconds | Best 35.3 seconds | Fail |
| Source-load reduction ≥ 80% | 30.9% measured at 2,560 pages | Fail |

B2 validates dynamic cache scaling and, more importantly, proves that
sequence-level working-set prediction has enough locality to pursue. It does
not yet make out-of-core execution fast enough.

## Recommended next experiment

Proceed to an oracle-retention/prefetch control before training a predictor:

1. Use the recorded future trace to protect or pre-stage the offline-optimal
   prompt-conditioned bundle.
2. Measure how much of the predicted 4.65 GB decode-I/O bound converts into real
   token rate when pages are fetched before their layer executes.
3. Add background source reads, pinned staging, and contiguous expert
   superpages so transfer can overlap compute.
4. Replace oracle knowledge with a prompt/early-hidden-state predictor and
   report recall, excess bytes, late page faults, and quality.

This ordering separates two questions: whether the 4090 can exploit a correct
working set, and whether the model can predict that working set online.

## Artifacts

- `outputs/b2-4090.json`: descending-order pass.
- `outputs/b2-4090-warm-ascending.json`: reciprocal warm pass with corrected
  exact trace simulation.
- `outputs/b2-4090-2560.json`: warm 2,560-page endpoint.

