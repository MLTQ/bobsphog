# B2.4 results: contiguous pages improve the source, but fault count is not yet throughput

## Outcome

B2.4 replaces per-expert safetensors slicing with a versioned fixed-offset page
store. The storage intervention succeeds:

- all 10,240 Qwen3.6-35B-A3B experts are packed into one 60 GiB BF16 file;
- every page has a stable 6 MiB offset;
- the complete file is memory mapped, while only requested pages are cloned and
  pinned;
- ordinary LRU and B2.3 lazy retention use the same exact source;
- sampled pages at `(0, 0)`, `(17, 203)`, and `(39, 255)` are bit-for-bit equal
  to both original checkpoint tensors; and
- all live lazy runs retain 100% token and expert-route parity.

The controlled policy result is more nuanced. Lazy retention still removes
**11.1% of decode faults**, but across 12 cold-source trials it is effectively
latency neutral: aggregate end-to-end time is 247.32 seconds versus 246.44 for
ordinary LRU. The contiguous layout makes the underlying source substantially
faster, but random cold-page cost is variable enough that counting fewer pages
does not yet predict wall time.

## Page-store format and build

The store is layer-major and expert-minor:

$$
p(\ell,e)=\ell N_e+e,
\qquad
o(\ell,e)=p(\ell,e)B_p.
$$

Each raw BF16 page concatenates `gate_up_proj` and `down_proj`:

$$
B_p=3d_{\text{model}}d_{\text{expert}}\cdot2
=6{,}291{,}456\text{ bytes}.
$$

For 40 layers and 256 experts per layer:

| Property | Value |
|----------|------:|
| Pages | 10,240 |
| Page size | 6 MiB |
| Data file | 64,424,509,440 bytes (60 GiB) |
| Original checkpoint | 71,903,645,408 bytes |
| Build time | 193.44 s |

The builder writes a page-aligned partial file, fsyncs after each layer, resumes
at the first incomplete expert, validates final size, then atomically publishes
the data filename and hashed metadata. Runtime initialization validates format,
geometry, and exact file size before mapping.

## Controlled protocol

Three held-out prompts were tested on the RTX 4090:

- `language-08`;
- `science-07`; and
- `history-08`.

Every run used 32 output tokens, a 2,560-page physical GPU cache, and the same
contiguous source. Lazy retention used the validated 2,520-page predicted set
but loaded no speculative pages. For each domain the order was ABBA:

1. ordinary LRU;
2. lazy retention;
3. lazy retention; and
4. ordinary LRU.

Immediately before every run, `POSIX_FADV_DONTNEED` requested eviction of the
store's clean OS-cache pages. This is store-specific best-effort Linux advice,
not a global cache flush. Model construction is excluded from inference timing,
as in B1–B2.3.

## Cold-source policy results

Values are means of two runs per cell.

| Prompt | Policy | Decode faults | Source read time | Decode rate | End-to-end |
|--------|--------|--------------:|-----------------:|------------:|-----------:|
| language | LRU | 1,797 | 33.69 s | 2.702 tok/s | 41.21 s |
| language | lazy | 1,528 | 32.58 s | 2.788 tok/s | 40.18 s |
| science | LRU | 2,223 | 34.75 s | 2.391 tok/s | 42.44 s |
| science | lazy | 2,058 | 34.35 s | 2.340 tok/s | 42.56 s |
| history | LRU | 1,887 | 32.16 s | 2.417 tok/s | 39.57 s |
| history | lazy | 1,667 | 33.17 s | 2.345 tok/s | 40.92 s |

Aggregated over all six runs per policy:

| Metric | LRU | Lazy | Lazy change |
|--------|----:|-----:|------------:|
| Decode faults per prompt | 1,969 | 1,751 | **-11.1%** |
| Mean source-load time | 33.53 s | 33.37 s | -0.5% |
| Aggregate decode rate | 2.494 tok/s | 2.472 tok/s | -0.9% |
| Total end-to-end time | 246.44 s | 247.32 s | +0.4% |
| Peak CUDA allocation | 19.62 GiB | 19.62 GiB | unchanged |
| Token / route parity | control | 100% / 100% | pass |

The fault reduction is real and deterministic, but the latency effect is below
run-to-run storage variance. In particular, a page avoided by lazy retention is
not necessarily a page that would have incurred an exposed disk read: readahead,
NVMe request merging, and incidental OS-cache residency make page costs unequal.

## Hot-source control

After warming the language prompt's working set, an ABBA control gave:

| Policy | End-to-end | Decode rate | Source-load time |
|--------|-----------:|------------:|-----------------:|
| LRU | 12.86 s | 5.328 tok/s | 5.83 s |
| Lazy | 12.30 s | 5.631 tok/s | 5.40 s |

Lazy is 4.4% faster end-to-end and 5.7% faster in decode here. This establishes
that retention can become throughput when page service cost is stable, but one
hot prompt is not evidence of a general production speedup.

## Storage-source effect

The closest prior safetensors LRU controls totaled 141.74 seconds across the
same three prompts. The cold contiguous LRU means total 123.22 seconds, a
directional **13.1% reduction**, while aggregate decode rate rises from 1.510 to
2.494 tokens/s. Source-load time falls from 118.26 to 100.60 seconds.

This comparison is not as strong as the B2.4 ABBA policy comparison: the old
controls were single runs and did not use explicit store eviction. It supports
the claim that fixed-offset storage is useful, not a precise 13.1% production
speedup estimate.

## Decision

| Gate | Result | Decision |
|------|--------|----------|
| Exact fixed-offset store builds and resumes | 60 GiB / 10,240 pages | Pass |
| Stored expert equality | Bit-for-bit sampled equality | Pass |
| Same source for both policies | Yes | Pass |
| Lazy fault reduction survives | 11.1% | Pass |
| Exact tokens and routes | 100% | Pass |
| Peak VRAM remains bounded | 19.62 GiB | Pass |
| Cold fault savings improve aggregate rate | -0.9% decode rate | Fail |
| Cold fault savings improve aggregate E2E | +0.4% time | Fail |
| Hot controlled prompt improves | 4.4% E2E reduction | Promising, insufficient |

B2.4 validates the storage abstraction and materially improves the page source.
It does **not** yet validate “saved faults = saved time” for cold random reads.

## Recommended next experiment: B2.5 cost-aware superpages

Keep the contiguous source and lazy retention, then make retrieval aware of the
physical read schedule:

1. measure per-page major faults, block I/O, and service latency rather than
   only source-call duration;
2. reorder or duplicate pages so co-requested layer groups occupy contiguous
   superpages;
3. issue batched `preadv`/io_uring reads into a reusable pinned host pool;
4. score candidates by expected exposed I/O cost saved, not page count alone;
5. compare 6 MiB pages with 12–48 MiB superpages under the same ABBA cold and
   hot protocols; and
6. reintroduce asynchronous prefetch only after the reader exposes batched,
   cancellable requests and separate foreground/background queues.

The key new target is correlation between predicted saved cost and measured
latency. A policy should not advance merely because it reduces logical misses.

## Artifacts

- `outputs/b24-page-store-build.json`: store geometry and build duration.
- `outputs/b24-controlled-summary.json`: all controlled rows and aggregate
  calculations.
- `outputs/b24-*-{lru,lazy}-cold{1,2}.json`: the 12 cold-source ABBA runs.
- `outputs/b24-language-{lru,lazy}-a{1,2}.json`: hot language ABBA control.
- `src/bobsphog/page_store.py`: builder, validator, mapped source, and eviction
  command.
