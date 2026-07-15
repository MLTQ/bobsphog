# B2.1 results: the right prompt-level bundle restores resident-speed decode

## Result

B2.1 converted the trace-only Belady bound into a live exact CUDA cache and
then moved the same future pages ahead of execution. On the RTX 4090:

- live future-aware retention matched the offline-optimal miss count exactly at
  1,280, 2,048, and 2,560 pages;
- a prefetched token incurs zero demand faults and executes at 7.31–8.56
  token/s;
- one-token lookahead still reaches only 1.03–1.18 token/s because source
  loading is much slower than expert compute; and
- preloading and pinning the complete prompt-conditioned decode union produces
  **zero decode I/O and 8.72 token/s** at a 19.62 GiB CUDA peak.

The final control preserves the exact B2 output while allocating 29.3% of the
71.9 GB checkpoint. This is the strongest pretrained validation so far of the
spongiform design: a query-specific submodel can decode at resident-model speed
without loading the global expert reservoir.

## Experiment design

All controls use the same 28-token prompt, 16 selected output tokens, and 640
atomic per-layer request groups captured in B2. No policy predicts routes; it is
given the recorded future deliberately to establish a systems upper bound.

Three live policies were measured:

1. **Oracle retention** executes normal demand faults but evicts the resident
   page used furthest in the future.
2. **Oracle token prefetch** synchronously stages all 320 pages required by the
   next token before timing that forward. Serial time is measured directly; a
   one-token-overlap estimate pipelines each prefetch with the preceding
   token's compute.
3. **Oracle prompt union** deduplicates all future decode pages by layer,
   preloads the 2,140-page union, pins it through prefill, and then decodes with
   no page movement.

Every run forces the same selected token IDs while independently checking its
top-1 predictions and exact live route against the trace.

## Live oracle retention

| Cache pages | Peak CUDA (GiB) | LRU decode misses | Oracle misses | LRU token/s | Oracle token/s |
|------------:|----------------:|------------------:|--------------:|------------:|---------------:|
| 1,280 | 12.12 | 2,058 | 910 | 0.534 | 0.918 |
| 2,048 | 16.62 | 1,613 | 739 | 0.605 | 0.953 |
| 2,560 | 19.62 | 1,296 | 739 | 0.740 | 0.944 |

The measured demand misses equal the grouped-Belady simulation at every
capacity. At 2,048 pages, selecting the correct victims cuts decode transfer
from 10.15 to 4.65 GB and raises decode by approximately 58%.

Increasing capacity from 2,048 to 2,560 does not reduce the 739 oracle misses.
Those pages are compulsory for this decode because they were never retained
from prefill. Capacity is no longer the limiting factor; the pages must move
earlier.

## Token-prefetch control

| Cache pages | Prefetched decode (GB) | Compute-ready token/s | Serial token/s | Ideal one-token overlap |
|------------:|-----------------------:|----------------------:|---------------:|------------------------:|
| 1,280 | 5.79 | 8.56 | 0.929 | 1.034 |
| 2,048 | 4.65 | 7.31 | 1.020 | 1.175 |
| 2,560 | 4.65 | 8.36 | 0.995 | 1.120 |

Every timed forward has zero demand misses. The compute-ready rate demonstrates
that the cache lookup and exact expert execution themselves are not slow.

At 2,048 pages, source loading and transfer take 12.65 seconds across 15 decode
forwards, while the actual forwards take only 2.05 seconds. One-token lookahead
cannot hide an I/O stage roughly six times longer than compute. The effective
expert-page source rate is approximately 0.4 GB/s, far below ordinary sequential
NVMe bandwidth because the current loader performs thousands of individual
safetensor slice operations.

Reaching 2 token/s with the same 0.31 GB/token physical demand requires at least
about 0.62 GB/s of sustained page delivery, plus enough lookahead to smooth
per-token variation. This is plausible, but requires contiguous superpages,
parallel reads, and multi-token rather than one-token scheduling.

## Prompt-union control

The 2,140-page decode union occupies 12.54 GiB. Its largest simultaneous prefill
layer adds only 130 transient pages, so both fit inside a 2,560-page cache.

| Phase | Pages loaded | Transfer (GB) | Time |
|-------|-------------:|--------------:|-----:|
| Prompt-union preload | 2,140 | 13.46 | 19.86 s |
| Prefill residual faults | 1,707 | 10.74 | 25.26 s |
| Decode | **0** | **0** | 1.72 s |

Prompt preload plus residual prefill loads exactly 3,847 pages—the same total
miss count as the full offline-optimal trace. It changes *when* useful pages are
loaded without adding physical page traffic.

| Metric | 2,560-page reactive LRU | Prompt-union oracle |
|--------|-------------------------:|--------------------:|
| TTFT | 37.79 s | 45.12 s |
| Decode rate | 0.740 token/s | **8.723 token/s** |
| Decode transfer | 8.15 GB | **0 GB** |
| 16-token latency | 58.05 s | **46.84 s** |
| End-to-end output rate | 0.276 token/s | **0.342 token/s** |

Despite 7.33 seconds of extra startup, the 16-token request is 19.3% faster
end-to-end. The startup investment breaks even after approximately **5.9 decode
forwards**, or around the seventh output token. Longer generations receive the
full 8.72 token/s benefit.

The prompt-union run transfers 24.20 GB across preload, prefill, and decode,
versus 27.71 GB for reactive LRU. Thus the improvement is not achieved by doing
more I/O up front; it performs 12.6% less total transfer and arranges it before
the latency-sensitive decode loop.

## Capability and memory controls

All policies and capacities had:

- 100% top-1 agreement on the forced path;
- identical selected token IDs and decoded text;
- exact agreement with all 640 recorded route groups; and
- the same CUDA peak as the corresponding B2 capacity.

The prompt-union result uses 19.62 GiB peak CUDA allocation, or 29.3% of
checkpoint bytes. No approximation, quantization, expert merging, or output
repair is involved.

## Gate decision

| Gate | Result | Decision |
|------|--------|----------|
| Live oracle equals offline miss bound | Exact at all capacities | Pass |
| Exact capability parity | 100% top-1 and route agreement | Pass |
| Compute-ready decode > 2 token/s | 7.31–8.72 token/s | Pass |
| Prompt bundle eliminates decode I/O | 0 misses, 0 bytes | Pass |
| 16-token request beats reactive LRU | 19.3% lower latency | Pass |
| TTFT < 10 seconds | 45.12 seconds | Fail |
| One-token overlap > 2 token/s | Best 1.18 token/s | Fail |

B2.1 proves the target working set is executable and fast. The remaining
research problem is predicting that bundle from the prompt or early hidden
state; the remaining systems problem is loading it concurrently and
contiguously during prefill.

## Recommended next experiment

Proceed to B2.2 as an offline predictor study before adding more cache runtime
complexity:

1. Capture per-layer prefill and 32–128-token decode unions for a diverse prompt
   corpus.
2. Split prompts by task/domain into train, validation, and held-out test sets.
3. Build a simple conditional coactivation/relationship index from prefill
   routes to later decode pages before training a neural predictor.
4. Select a fixed-budget prompt bundle and report union recall, precision,
   excess bytes, late faults, and simulated token rate.
5. Run the best held-out predictor through the live pinned cache.

In parallel, replace individual safetensor slices with contiguous expert
superpages and background source workers. Prediction and transfer should be
evaluated separately, then combined only after each beats its oracle-normalized
baseline.

## Artifacts

- `outputs/b21-4090-2048-retention.json`: live 2,048-page Belady control.
- `outputs/b21-4090-2048-prefetch.json`: 2,048-page token-prefetch control.
- `outputs/b21-4090-endpoints.json`: 1,280/2,560 retention and token prefetch.
- `outputs/b21-4090-2560-prompt-union.json`: zero-fault prompt-union result.

