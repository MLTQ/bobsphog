# A5 results: physical CUDA page residency

## Environment

The first physical benchmark ran remotely on:

- Arch Linux;
- NVIDIA GeForce RTX 4090;
- PyTorch 2.11.0 with CUDA 13.0; and
- exact float16 low-rank factors.

The benchmark model uses two 2,048-channel transformer layers, 8,192-channel
FFNs, rank-128 resident factors, and fifteen rank-128 optional pages per FFN
matrix. There are 60 optional pages totaling 150 MiB.

The model is randomly factorized because A5 isolates systems behavior. It avoids
an expensive large dense SVD while exercising the same `PagedLinear` execution
contract used by trained models.

## Physical layout

After construction:

- attention, embeddings, normalization, output head, and base factors remain on
  the 4090;
- all optional source factors move to pinned CPU memory;
- a bounded LRU cache owns exact CUDA copies of the selected pages; and
- a prompt plan is scheduled on a dedicated CUDA stream; and
- each page records a CUDA readiness event that the compute stream waits on only
  when that layer consumes it.

Unlike A1–A4 logical residency, these measurements use CUDA
`memory_allocated` after physical offload and allocator cleanup.

## Memory and transfer curve

| Cached pages | Page bytes | CUDA allocation | Savings vs fully resident | Cold prepare | Warm forward |
|-------------:|-----------:|----------------:|--------------------------:|-------------:|-------------:|
| 4 | 10 MiB | 94.5 MiB | 59.7% | 0.87 ms | 0.87 ms |
| 8 | 20 MiB | 104.5 MiB | 55.4% | 1.58 ms | 0.99 ms |
| 16 | 40 MiB | 124.5 MiB | 46.9% | 2.91 ms | 1.22 ms |

The fully resident model occupies 234.5 MiB and the offloaded skeleton occupies
84.5 MiB. Each additional selected page contributes its exact factor bytes to
measured CUDA allocation.

Cold pinned-memory transfer throughput is approximately 11.5–13.8 GiB/s across
this sweep. Transfer time and allocation scale approximately linearly with
working-set bytes.

## Correctness and cache behavior

All three working-set sizes produced zero maximum absolute logit difference
between:

1. the selected pages executed while every model page was CUDA-resident; and
2. the same selected pages copied from pinned CPU sources into the bounded cache.

For the eight-page run:

- cold preparation: 8 misses, 0 hits, 20 MiB transferred;
- immediate reuse: 8 hits, 0 misses, 0 bytes transferred;
- switch to a disjoint set: 8 misses, 8 evictions, 20 MiB transferred; and
- return to the original set: 8 misses, 8 evictions, 20 MiB transferred.

Warm cached forward time is essentially the same as fully resident selected-page
execution: 0.99 ms versus 1.01 ms. The provider indirection adds no measurable
steady-state penalty at this scale.

## Asynchronous overlap

A repeated eight-page run compared two equivalent cold cycles. Both started
with the disjoint working set cached, fetched the same 20 MiB primary set, ran
the same forward, and ended with a device synchronization.

| Cold path | End-to-end time | Host wait inside cache | Logit error |
|-----------|----------------:|-----------------------:|------------:|
| Synchronous prepare + forward | 2.342 ms | 0.736 ms | 0 |
| Schedule + event-gated forward | 1.605 ms | 0.011 ms | 0 |

The nonblocking path removed 0.737 ms, or **31.5%**, from observed cold-cycle
latency. Copies for later pages remain queued on the transfer stream while the
default stream starts attention and consumes earlier pages as they become
ready. This is genuine transfer/compute overlap: the timer includes host
scheduling and the final CUDA synchronization.

## Interpretation

A5 demonstrates the missing physical claim from earlier stages:

> Exact executable pages can remain in host memory while a bounded accelerator
> cache holds only the current working set, reducing measured CUDA allocation
> without changing model output.

The result also quantifies the systems tradeoff. A cold 20 MiB working set costs
about 1.5 ms to prepare synchronously. Event-gated execution hides roughly half
of that exposed transfer cost in this model, while a cache hit costs no
transfer. Prompt-level set stability and accurate prefetch therefore matter
more than raw page execution overhead.

## Limitations

- Sources reside in pinned CPU memory, not NVMe.
- The scaled benchmark model is random rather than a pretrained LLM.
- The learned selector is tested separately in A6; this scaled A5 model still
  uses static working sets.
- Cache capacity is measured in exact bytes, but pages happen to be equal-sized
  in this benchmark.
- CUDA reserved-memory behavior is allocator dependent; reported comparisons use
  live allocated tensor bytes.
- The overlap figure is one model shape and one GPU, not yet a throughput curve
  across layer depths, page sizes, or concurrent requests.

## Next gate

A6 connects base-query selection to the event-gated cache. The next systems
gate is to repeat these measurements on a small pretrained checkpoint and
report prefill/decode latency under stable prompts and topic shifts.
