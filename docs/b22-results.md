# B2.2 results: prompt routes predict decode pages, scheduling is now the bottleneck

## Outcome

B2.2 establishes that a prompt-conditioned expert working set is predictable
on held-out prompts and useful in the exact 4090 page cache. A simple
three-nearest-neighbor predictor is the strongest practical baseline. At a
2,048-page bundle in a 2,560-page cache it:

- covers **70.54% of decode requests** across 16 held-out prompts;
- raises simulated effective cache hits from **72.88% to 84.53%**;
- reduces live decode faults by **3.3% to 39.5%** on three deliberately
  bracketed prompts;
- speeds exact decode by **1.59x to 2.31x** in the before-prefill runs; and
- preserves **100% token and route parity** with CUDA-native execution.

The result also rejects two tempting assumptions. The first coactivation
relationship tensor is weaker than nearest-neighbor retrieval, and pinning the
largest possible bundle is not optimal because pinned pages displace useful
transient cache capacity.

End-to-end latency is not yet consistently better. Synchronous bundle loading
and prefill/cache interference consume the decode savings on two of the three
live prompts. The next problem is scheduling and confidence-gated bundle size,
not proving that a predictable working set exists.

## Corpus and split

The full Qwen3.6-35B-A3B checkpoint was traced on the 128 GiB Strix Halo ROCm
host so all experts could remain resident:

| Property | Value |
|----------|------:|
| Prompts | 64 |
| Domains | 8 |
| Train / validation / test | 40 / 8 / 16 |
| Output tokens per prompt | 32 |
| Decode forwards per prompt | 31 |
| Layers / experts per layer / top-k | 40 / 256 / 8 |
| Decode page requests per prompt | 9,920 |
| Mean held-out decode union | 2,671.5 pages |

The domains are science, coding, math, writing, history, planning, language,
and mixed tasks. Split assignments were fixed in the prompt file before route
collection. Hyperparameters were selected on validation only.

## Predictors

All methods receive an equal per-layer quota:

- **Random:** deterministic random scores.
- **Global frequency:** training-set decode request frequency.
- **Prefill reuse:** global frequency plus pages active during this prefill.
- **Nearest neighbor:** request-weighted routes from the most similar training
  prefills; validation chose three neighbors.
- **Conditional coactivation:** the first explicit relationship index,
  estimating decode-page value from active prefill pages.
- **Oracle request frequency:** the held-out prompt's true future request
  counts; this is an upper bound, not a deployable policy.

At the selected 2,048-page budget:

| Method | Decode request hit | Union recall | Effective cache hit | Simulated faults |
|--------|-------------------:|-------------:|--------------------:|-----------------:|
| Pure 2,560-page LRU, no pinning | 0.00% | 0.00% | 72.88% | 2,690.0 |
| Random | 20.16% | 20.23% | 58.54% | 4,112.9 |
| Global frequency | 56.99% | 45.11% | 79.84% | 1,999.5 |
| Prefill reuse | 67.08% | 52.22% | 83.94% | 1,592.8 |
| **Three nearest neighbors** | **70.54%** | **52.55%** | **84.53%** | **1,535.0** |
| Conditional coactivation | 60.73% | 46.79% | 80.96% | 1,888.6 |
| Oracle future frequency | 93.06% | 76.10% | 93.44% | 650.8 |

Random pinning performs worse than unpinned LRU because it reserves most of the
cache for irrelevant pages. This is an important negative control: any speedup
comes from route information, not merely from preloading weights.

## Budget tradeoff

The predicted bundle and transient cache share one fixed 2,560-page allocation.
The best effective-hit budgets from the sweep were:

| Method | Best pinned budget | Effective hit | Decode faults |
|--------|-------------------:|--------------:|--------------:|
| Global frequency | 1,536 | 80.88% | 1,896.9 |
| Prefill reuse | 1,792 | 84.12% | 1,575.8 |
| **Nearest neighbor** | **2,048** | **84.53%** | **1,535.0** |
| Conditional coactivation | 1,792 | 81.59% | 1,825.9 |
| Oracle future frequency | 2,560 | 96.83% | 314.8 |

Increasing nearest-neighbor pinning from 2,048 to 2,304 pages raises direct
request coverage from 70.54% to 73.89%, but effective cache hits fall to 83.55%
because the transient LRU shrinks from 512 to 256 pages. Dynamic residency must
optimize the complete cache, not only predictor recall.

Cold total bytes, including the one-time bundle prefetch, are not lower than
pure LRU for the practical predictors. The approach can still win because one
ordered prompt prefetch is cheaper than many small synchronous page faults, but
that benefit depends on storage layout and overlap.

## Cross-backend parity

The ROCm and CUDA routers are numerically close but not bit-identical. On the
first attempted CUDA prefill, layer 1 selected 121 of the same 123 experts, with
two missing and three additional unique experts. The parity gate correctly
rejected that run.

The corrected protocol keeps predictor fitting entirely on the ROCm corpus but
captures each live prompt's execution route and greedy token path with the
ordinary CUDA LRU cache. The predicted bundle is then evaluated against that
CUDA-native trace. This tests cross-backend predictor transfer while retaining
exact deployment-backend token and route gates.

## Live 4090 validation

Three held-out cases bracket the offline predictor range: the weakest case,
one near the median, and the strongest. The table below uses the
`before_prefill` schedule, which synchronously loads and pins the bundle before
the prompt forward.

| Prompt | CUDA request hit | LRU -> predicted faults | Fault reduction | LRU -> predicted decode | Speedup | LRU -> predicted end-to-end |
|--------|-----------------:|-------------------------:|----------------:|------------------------:|--------:|----------------------------:|
| science-07 | 58.89% | 2,223 -> 2,149 | 3.3% | 1.015 -> 1.615 tok/s | 1.59x | 61.67 -> 74.63 s |
| language-08 | 73.70% | 1,797 -> 1,431 | 20.4% | 1.294 -> 2.985 tok/s | 2.31x | 56.94 -> 37.06 s |
| history-08 | 79.83% | 1,887 -> 1,141 | 39.5% | 1.288 -> 2.203 tok/s | 1.71x | 60.55 -> 65.83 s |

Mean decode speedup is 1.87x and mean fault reduction is 21.1%. Summed
end-to-end time is approximately flat: 177.52 seconds predicted versus 179.15
seconds for LRU. The median predictor case improves end-to-end latency by
34.9%, but the low-confidence case should not have loaded a 2,048-page bundle.

All six live predicted schedules—before- and after-prefill for all three
prompts—have:

- 100% forced-path top-1 agreement;
- 100% expert-route agreement against the CUDA-native trace; and
- the same approximately 19.62 GiB peak CUDA allocation as the LRU control.

For the three CUDA cases, the offline replay predicted 2,179, 1,461, and 1,183
faults. Hardware measured 2,149, 1,431, and 1,141, errors of 1.4%, 2.1%, and
3.6%. The simulator is therefore suitable for fast policy iteration.

## Prefetch-phase control

`after_prefill` gives the prompt forward the full LRU cache and only then
reshapes it into the predicted decode bundle. It produces the exact same decode
fault counts as before-prefill pinning, confirming the cache model. However,
the synchronous reshape adds 9.4–11.7 seconds after the first token and loses
end-to-end latency on all three prompts.

The before-prefill schedule can instead starve prefill: only 512 transient pages
remain while the prompt touches 3,000+ experts. Its timing varies sharply with
the Linux file cache. NVMe source-load time, rather than expert arithmetic,
dominates both schedules. Decode-rate numbers should therefore be treated as
exploratory systems measurements; exact fault counts and parity are the robust
result.

## Decision

B2.2 passes the predictor gate:

| Gate | Result | Decision |
|------|--------|----------|
| Held-out predictor beats static frequency | 70.54% vs 56.99% request hit | Pass |
| Held-out predictor beats prefill reuse | 70.54% vs 67.08% | Pass |
| Relationship index beats simple retrieval | 60.73% vs 70.54% | Fail |
| Cache benefit survives live execution | 3.3%–39.5% fewer faults | Pass |
| Offline cache model predicts hardware | 1.4%–3.6% fault error | Pass |
| Exact token and route parity | 100% on six runs | Pass |
| End-to-end speedup is reliable | Prompt dependent; aggregate near flat | Fail |

## Recommended next experiment: B2.3 adaptive overlapped prefetch

1. Calibrate predicted benefit and choose 0–2,048 pinned pages per prompt;
   low-confidence prompts such as science-07 should retain more LRU capacity or
   skip prefetch entirely.
2. Prefetch predicted pages concurrently with prefill instead of before or
   after it. Use the approximately 4 GiB of remaining 4090 headroom as a bounded
   staging pool, then promote pages into the decode cache as layers complete.
3. Pack frequently co-requested experts into contiguous superpages and separate
   storage-read time from H2D-copy time.
4. Warm or explicitly control the host page cache and alternate policy order
   for latency comparisons.
5. Run the resulting policy across all 16 held-out prompts before training a
   neural controller.

The coactivation tensor should not be scaled up in its current form. A learned
set predictor can wait until the adaptive nearest-neighbor policy establishes a
clean transfer/latency target.

## Artifacts

- `outputs/b22-strix-corpus.json`: 64-prompt resident ROCm route corpus.
- `outputs/b22-predictor-results.json`: complete validation/test predictor
  metrics and per-domain breakdowns.
- `outputs/b22-cuda-baseline-*.json`: CUDA-native LRU traces for live cases.
- `outputs/b22-live-4090-*.json`: before-prefill pinned-bundle runs.
- `outputs/b22-live-4090-*-after-prefill.json`: post-prefill scheduling controls.
