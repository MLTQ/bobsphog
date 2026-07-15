# A6 results: learned prompt plans drive physical paging

## Question

A6 closes the loop between the learned A3 retriever and the A5 CUDA runtime:

> Can the resident skeleton choose a complete prompt working set before any
> optional factor is loaded, and can that learned plan then execute from a
> bounded physical cache without changing the selected model's output?

The selector runs the base-only student once, freezes the resulting prompt
query, and greedily scores omitted pages while updating only its compact
resident-set embedding. Unlike the earlier adaptive A3 selector, it never
executes a newly selected page to choose the next one. The complete plan is
therefore known before transfer begins.

## Environment and training

The seed-61 run used the RTX 4090, PyTorch 2.11.0, CUDA 13.0, and the two-domain
modulo-ten task. The model contains 28 equal rank-4 pages. Training used 400
teacher steps, 400 structured-page-dropout student steps, 9,216 counterfactual
training labels, and 500 estimator steps.

The utility estimator achieved:

- validation correlation: 0.382;
- validation RMSE: 0.666 loss units; and
- utility-sign accuracy: 88.2%.

The lower correlation cautions against treating its exact utility magnitudes as
well calibrated, but its ranking signal is sufficient for this integration run.

## Held-out learned-plan quality

Each row is one held-out batch of 128 examples. The learned plan uses eight of
28 optional pages.

| Domain | Base only | Learned 8-page plan | Full 28 pages | Warmed selection time |
|--------|----------:|--------------------:|--------------:|----------------------:|
| Addition | 38.1% | 84.0% | 99.8% | 3.61 ms |
| Multiplication | 51.8% | 95.3% | 100.0% | 3.56 ms |

This experiment uses the deployable base-query selector, so these values should
not be compared directly with A3's adaptive selector. The important result is
that the base state alone recovers a useful query-specific bundle before
optional page residency.

The two plans share six pages and differ in two. That is evidence of a shared
computational core plus a small domain-conditioned fringe for this seed—not a
claim that these pages correspond to clean human-readable arithmetic experts.

## Physical cache execution

All 28 source pages moved to pinned CPU memory. CUDA cache capacity was fixed at
exactly one eight-page plan (12 KiB in this toy), and page copies were scheduled
without a global host barrier.

| Event | Hits | Misses | Evictions | H2D bytes | End-to-end | Max logit error |
|-------|-----:|-------:|----------:|----------:|-----------:|----------------:|
| Cold addition plan | 0 | 8 | 0 | 12,288 | 1.737 ms | 0 |
| Switch to multiplication | 6 | 2 | 2 | 3,072 | 1.292 ms | 0 |

The domain switch reuses exactly the six overlapping pages and transfers only
the two differing pages. Both physical outputs are bit-identical to the same
selected plans evaluated while all factors were originally resident, and their
task accuracies are unchanged.

## What A6 establishes

The prototype now demonstrates the complete intended control loop at toy scale:

1. run a coherent resident skeleton;
2. form a prompt-conditioned retrieval query;
3. choose the entire bounded working set without candidate execution;
4. schedule exact page factors from pinned host memory;
5. overlap copies with layer execution using per-page CUDA events;
6. reuse shared pages and evict only the domain-specific difference; and
7. reproduce the selected resident model exactly.

Together with A5, this separates two claims: A6 proves learned-control
integration, while the scaled A5 run proves meaningful physical memory savings
and a 31.5% reduction in cold-cycle latency from overlap.

## Limitations

- A6 pages are only 1.5 KiB, so its transfer timings are dominated by launch and
  Python overhead; use A5 for systems conclusions.
- Selection performs eight serial argmax decisions. Batched top-$k$, beam/bundle
  selection, or a direct set predictor should reduce its 3.6 ms control cost.
- Quality is reported on one batch per domain and one seed. A multi-seed base-
  query comparison against static, random, adaptive, and oracle plans remains
  necessary.
- The relationship graph is not yet part of the physical plan builder.
- The cold tier is pinned RAM, not NVMe, and no autoregressive KV-cache workload
  is included.
- The model is synthetic and tiny; this does not establish useful language-model
  behavior.

## Next gate

The toy mechanics are complete enough to begin Stage B. The recommended next
step is a checkpoint adapter for a permissively licensed 1–3B decoder model:
keep attention and a low-rank FFN base resident, construct executable residual
pages from the pretrained weights, first validate exact/approximate
reconstruction, and only then fine-tune the pages and retriever. A small static
prefix and random selector should remain as hard baselines throughout.
