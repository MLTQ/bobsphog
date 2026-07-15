# B1.6 results: exact paging token rate

## Bottom line

Exact expert paging has two very different speed regimes in the current
prototype:

1. With weights effectively warm in unified/RAM-backed memory on Strix Halo,
   paging decodes at **2.77 tokens/s**, 2.45 times slower than the 6.79 tokens/s
   full-resident control.
2. With reactive file-backed sourcing on the RTX 4090, paging decodes at
   **0.62 tokens/s**, or 1.61 seconds per token, with a 39.4-second TTFT.

The first regime is a plausible quality/memory trade. The second is not yet an
acceptable interactive runtime. It is a baseline for B2, not a finished result.

## Controlled workload

All paths received the same 28-token formatted prompt and executed exactly 16
output tokens through Qwen's real KV/recurrent-state decode. The full and paged
Strix paths followed the same token IDs. The independent 4090 and RTX 2070
paths also produced the identical text:

```text
The sky appears blue due to a phenomenon called **Rayleigh scattering**.

Here
```

Model-load time is reported separately and excluded from TTFT and token-rate
calculations. Each decode step is accelerator-synchronized before timing.

## Token-rate results

| Metric | Strix full | Strix paged | RTX 4090 paged | RTX 2070 paged |
|--------|-----------:|------------:|---------------:|---------------:|
| Model load | 15.86 s | 1.32 s | 4.07 s | 4.88 s |
| TTFT / 28-token prefill | 3.36 s | 5.55 s | 39.40 s | 65.20 s |
| Steady decode | 6.79 tok/s | 2.77 tok/s | 0.62 tok/s | 0.43 tok/s |
| Mean decode latency | 0.147 s/tok | 0.361 s/tok | 1.613 s/tok | 2.335 s/tok |
| P95 decode latency | 0.164 s | 0.480 s | 3.590 s | 3.723 s |
| End-to-end time, 16 tokens | 5.57 s | 10.96 s | 63.59 s | 100.22 s |
| End-to-end output rate | 2.87 tok/s | 1.46 tok/s | 0.252 tok/s | 0.160 tok/s |
| Peak allocation | 69.46 GB | 7.04 GB | 6.97 GB | 6.97 GB |

The authoritative same-device slowdown is the Strix comparison:

- TTFT is **1.65x** slower;
- steady decode latency is **2.45x** higher; and
- 16-token end-to-end latency is **1.97x** higher.

The cross-device Strix-full versus 4090-paged ratio is not an architectural
apples-to-apples comparison. It does, however, describe the user-visible
deployment baseline: the current 4090 path is roughly 11 times slower in both
steady decode and short-response end-to-end latency than the full Strix
control.

## Why the 4090 is slow

The 4090 paged run requested 7,908 expert pages and moved 39.16 GB for one
28-token prompt plus 16 output tokens.

| Phase | Misses | Hits | Bytes transferred | Source-load time | Wall time |
|-------|-------:|-----:|------------------:|-----------------:|----------:|
| Prefill | 3,108 | 0 | 19.55 GB | 37.28 s | 39.40 s |
| Decode | 3,116 | 1,684 | 19.60 GB | 21.12 s | 24.19 s |
| Total | 6,224 | 1,684 | 39.16 GB | 58.40 s | 63.59 s |

Source loading consumes **91.8% of end-to-end time**. During decode, the cache
hit rate is only 35.1%, and each output step transfers an average of 1.31 GB.
The residual time after subtracting synchronous source loading is about 5.2
seconds for the whole run. This is not a perfect counterfactual because transfer
and compute are not yet overlapped, but it shows that expert computation itself
is not the primary bottleneck.

The Strix paged path moved almost the same 39.23 GB but loaded it in 6.08
seconds. That run benefited from unified memory and a warm OS page cache after
the full-reference load, so its 2.45x slowdown is the warm-memory regime rather
than a cold-NVMe result.

## Is the slowdown worth it?

Today:

- **On Strix Halo:** arguably yes for memory-constrained or batch workloads.
  A 2.45x decode penalty buys an 89.9% peak-memory reduction with exact observed
  outputs.
- **On the 4090:** not yet for interactive use. A 39-second TTFT and 0.62 tok/s
  steady rate are too slow unless the alternative is not running the model at
  all.

The 4090 experiment deliberately uses only 320 pages, or about 2.01 GB of
expert cache, leaving much of the 24 GB GPU unused. B2 should measure whether
spending more of that memory on 640, 1,280, and roughly 2,048 pages sharply
reduces misses and transfer amplification.

For a first practical target, reaching 2 tok/s on the 4090 requires reducing
decode time from 24.19 seconds to 7.5 seconds for these 15 steps. If non-source
time stays near 3.07 seconds, synchronous page-loading time must fall by about
79%. Matching the 2.77 tok/s warm Strix paged rate requires roughly an 89%
reduction in source-loading time.

## B2 performance gates

B2 should treat the following as explicit gates rather than merely reporting a
better cache hit rate:

1. TTFT below 10 seconds for this prompt.
2. Steady decode above 2 tok/s on the 4090.
3. At least 80% less synchronous source-load time.
4. Exact output parity with the B1.5 capability suite.
5. A measured memory/rate frontier across cache sizes, not one arbitrary point.

## Reproducibility

- Checkpoint revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Cache budget: 320 experts / 2,013,265,920 bytes
- Output length: 16 fixed decode tokens
- Strix raw result: `outputs/b1-throughput-strix.json`
- RTX 4090 raw result: `outputs/b1-throughput-4090.json`
- RTX 2070 raw result: `outputs/b1-throughput-2070.json`

Run the same-device control with:

```bash
bobsphog-b1-throughput \
  --checkpoint /path/to/Qwen3.6-35B-A3B \
  --mode both \
  --output-tokens 16
```
