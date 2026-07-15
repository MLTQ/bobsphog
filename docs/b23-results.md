# B2.3 results: lazy query-conditioned retention works; speculative I/O does not

## Outcome

B2.3 separates three ideas that had been conflated in eager prompt prefetch:

1. predicting pages likely to matter;
2. protecting those pages after they become resident; and
3. reading predicted pages before they are demanded.

The first two work. The third does not yet work on the 4090's current
safetensor/NVMe path.

The best policy is **lazy query-conditioned retention**:

- prompt prefill receives the full 2,560-page LRU cache;
- the predictor marks 2,520 candidate pages but does not load them;
- a marked page becomes non-evictable only after a real decode request loads it;
- the remaining 40-page margin can hold every eight-expert atomic layer group;
- the marked set is discarded at the next prompt or topic epoch.

Across all 16 held-out prompts this reduces simulated decode faults from 30,165
to 26,938, a **10.7% reduction with zero speculative page transfers**. It is
within 30 faults of the per-prompt oracle adaptive budget.

On three live CUDA prompts it reduces faults by 7.4%–15.0%, preserves exact
tokens and routes, and matches the phased simulator exactly. End-to-end time is
still approximately 4% worse in the warm-order comparison because individual
safetensor fault cost is unstable and dominates the saved fault count.

## Corrected confidence target

B2.2's decode-only cold-cache replay overstated the value of pinning. B2.3 now
replays:

1. ordinary unpinned prompt prefill;
2. the resulting warm physical cache state;
3. decode with query-selected pages protected only after first demand.

This model exactly predicts all three live 2,520-page fault counts.

Confidence features use only information available before the query's decode
target is examined:

- nearest-prefill similarity, mean, variance, and margin;
- agreement among the three nearest training routes;
- predicted request-score mass inside each candidate budget;
- overlap between the candidate set and prompt-prefill pages;
- neighbor fault savings under the same physical cache policy; and
- prompt route density.

Forty leave-one-out training prompts provide savings labels. Ridge strength and
the absolute-error confidence bound are selected on the eight validation
prompts. Held-out test routes are read only after budget choice.

| Calibration metric | Result |
|--------------------|-------:|
| Selected ridge alpha | 100 |
| Validation savings RMSE | 46.72 pages |
| 80th-percentile absolute error | 59.09 pages |

For lazy retention, unused candidate pages consume neither I/O nor VRAM. Their
only cost appears if they are eventually requested and then protected instead
of another page. That opportunity cost is already present in the replay label.
Consequently all 16 held-out prompts have positive conservative utility at the
largest safe budget.

## Held-out budget frontier

| Candidate pages | Decode faults | Fault savings |
|----------------:|--------------:|--------------:|
| 0 / ordinary LRU | 30,165 | 0 |
| 1,024 | 28,905 | 1,260 |
| 1,536 | 28,100 | 2,065 |
| 1,792 | 27,760 | 2,405 |
| 2,048 | 27,429 | 2,736 |
| 2,304 | 27,096 | 3,069 |
| 2,400 | 27,037 | 3,128 |
| 2,480 | 26,950 | 3,215 |
| **2,520** | **26,938** | **3,227** |

The oracle chooses 2,520 pages for 13 prompts and slightly smaller budgets for
three, saving 3,257 faults. The calibrated policy's fixed 2,520-page decision
therefore captures **99.1% of oracle savings**. Dynamic budget variation adds
almost nothing for these 32-token, single-topic prompts once transfer is lazy.

This does not make confidence unnecessary. It becomes important when:

- candidates are eagerly staged;
- responses are long enough for most marked pages to become resident;
- the prompt changes topic during decode; or
- multiple sequences compete for one cache.

With a modeled exposed staging cost of 0.05 fault-equivalents per candidate,
the controller selects zero pages for two prompts and a mixture of 1,792–2,520
for the rest. At 0.10 it correctly rejects speculative staging for every prompt.

## Physical schedules tested

### Eager before-prefill

Loads and pins the complete bundle before prompt computation. It can make
decode fast, but constrains prefill to the small residual cache and has a large
synchronous TTFT cost.

### Synchronous after-prefill

Preserves full-cache prefill, then pauses to reshape the cache. The 9–12 second
promotion barrier consumes the decode savings.

### Background host staging during prefill

Reads predicted pages into host RAM concurrently with prefill, then promotes
them. This is exact but counterproductive because both foreground and
background readers contend for the same safetensor/NVMe path.

For `language-08`:

| Background budget | TTFT | Promotion pause | End-to-end |
|------------------:|-----:|----------------:|-----------:|
| 1,024 | 47.38 s | 5.25 s | 73.07 s |
| 2,048 | 51.22 s | 10.17 s | 76.18 s |

The ordinary warm LRU control completes in 50.15 seconds. Making I/O
asynchronous did not make it free; it slowed the foreground reads.

### Lazy background decode staging

Starts staging after prefill while marked pages become protected on first use.
It removes the promotion barrier, but storage contention remains. Across the
three prompts it helps one and hurts two relative to warm LRU.

### Lazy pin only

Performs no speculative read. It is the clean retention control and the B2.3
winner.

## Live 4090 lazy-retention result

The 2,520-page policy was evaluated against freshly repeated warm-order LRU
controls. Every run uses the CUDA-native greedy token path.

| Prompt | LRU faults | Lazy faults | Reduction | LRU decode | Lazy decode | LRU E2E | Lazy E2E |
|--------|-----------:|------------:|----------:|-----------:|------------:|--------:|---------:|
| science-07 | 2,223 | 2,058 | 7.4% | 1.609 tok/s | 1.327 tok/s | 45.06 s | 47.94 s |
| language-08 | 1,797 | 1,528 | 15.0% | 1.486 tok/s | 1.600 tok/s | 50.15 s | 51.62 s |
| history-08 | 1,887 | 1,667 | 11.7% | 1.444 tok/s | 1.434 tok/s | 46.53 s | 48.04 s |

Combined faults fall from 5,907 to 5,253, an **11.1% physical reduction**.
The simulator predicts 2,058, 1,528, and 1,667 misses respectively: exact
agreement with hardware.

All runs retain:

- 100% forced-path top-1 agreement;
- 100% expert-route agreement;
- the same approximately 19.62 GiB peak CUDA allocation; and
- no speculative page bytes.

TTFT changes even though lazy pinning is not active until after prefill. That is
direct evidence that single-run latency is dominated by Linux page-cache and
checkpoint-read state. Fault counts and parity are the authoritative B2.3
metrics; token rate will become meaningful only after the storage source is
controlled.

## Decision

| Gate | Result | Decision |
|------|--------|----------|
| Confidence labels include warm prefill state | Exact live miss prediction | Pass |
| Held-out lazy retention reduces faults | 10.7% across 16 prompts | Pass |
| Policy approaches oracle budget savings | 99.1% | Pass |
| Live exactness | 100% token and route parity | Pass |
| Live fault reduction | 7.4%–15.0% | Pass |
| Same-source background staging helps | Foreground contention dominates | Fail |
| Fault savings reliably improve latency | Aggregate approximately 4% slower | Fail |

## Recommended next experiment: B2.4 contiguous page source

The next bottleneck is the physical representation of cold pages, not route
prediction.

1. Pack expert pages into a persistent contiguous page store with stable
   offsets and long-lived file handles.
2. Group frequently co-requested pages into sequential superpages.
3. Separate storage-read timing from host-to-device timing and cache both
   independently.
4. Repeat policies in randomized AB/BA order with an explicitly warmed or
   flushed source state.
5. Reintroduce background prefetch only when it uses spare bandwidth or an
   independent RAM-backed tier.
6. Add epoch reset and topic-shift tests before multi-sequence cache sharing.

B2.3's lazy candidate page table should remain the retention baseline for that
work.

## Artifacts

- `outputs/b23-policy-results.json`: confidence calibration, sensitivity grid,
  static frontier, and held-out choices.
- `outputs/b23-warm-baseline-*.json`: repeated warm-order LRU controls.
- `outputs/b23-live-4090-*-lazy-pin-2520.json`: final live retention runs.
- `outputs/b23-live-4090-*-lazy-1024.json`: lazy background staging controls.
- `outputs/b23-live-4090-language-08-background*.json`: prefill-overlap failure
  controls.
