# B1 results: Qwen3.6-35B-A3B runs out of core

## Result

The full Qwen3.6-35B-A3B language model executed on one RTX 4090 without ever
materializing its 32.2B routed-expert parameters in host RAM or VRAM.

The loader constructed the text model on the meta device, replaced every packed
expert reservoir with a cache-backed shell, loaded 613 non-expert tensors from
all 26 shards, and left 10,240 routed experts as safetensor slices on NVMe.

This is the first pretrained demonstration of the core spongiform claim:

> A 71.9 GB checkpoint can execute with a small resident scaffold plus a bounded
> query-conditioned subset of exact weight pages.

## Residency

The one-token test uses at most eight experts in each of 40 layers. Cache
capacity was fixed to exactly 320 experts.

| Component | Bytes | GiB |
|-----------|------:|----:|
| File-backed checkpoint | 71,903,645,408 | 66.97 |
| Resident language scaffold | 4,896,747,520 | 4.56 |
| Expert cache capacity | 2,013,265,920 | 1.875 |
| Measured cold peak CUDA allocation | 6,952,478,208 | 6.48 |

Peak CUDA allocation is **9.67% of checkpoint bytes**, a 90.33% reduction. The
peak is almost exactly scaffold plus selected expert pages; no hidden full-
expert allocation occurs.

Each expert page is one BF16 gate/up slice and one BF16 down slice:

$$
3(2048)(512)(2\text{ bytes})=6{,}291{,}456\text{ bytes}=6\text{ MiB}.
$$

## Cold and warm replay

The reported final run used the one-token input `Hello`. The OS file cache had
been warmed by checkpoint download and development runs; “cold” below means the
CUDA expert cache was empty, not that the NVMe page cache was forcibly dropped.

| Path | Hits | Misses | H2D bytes | Latency | Logit difference |
|------|-----:|-------:|----------:|--------:|-----------------:|
| Empty expert cache | 0 | 320 | 2,013,265,920 | 1.623 s | — |
| Identical replay | 320 | 0 | 0 | 242.8 ms | 0 |

The cold path spent 610 ms loading expert slices and 726 ms inside source plus
CUDA scheduling. Earlier dtype-correct runs ranged to 2.55 seconds as host-cache
state varied. The warm replay is bit-identical and roughly 6.7 times faster.

The model's top continuations were coherent:

```text
Hello -> ",", " everyone", "!", " all", " there"
```

This is not a formal quality comparison with an official engine, but it is a
useful end-to-end checksum that embeddings, hybrid attention/DeltaNet layers,
routers, shared experts, routed experts, normalization, and output projection
are connected in the intended order.

## Decode-time churn

The warm pass also created Qwen's real recurrent/KV state. Its argmax token was
`,`; advancing that token produced a new 320-page routed working set:

| Transition | Hits | Misses | Evictions | H2D bytes | Latency |
|------------|-----:|-------:|----------:|----------:|--------:|
| `Hello` → `,` | 75 | 245 | 245 | 1,541,406,720 | 4.929 s |

Only **23.4%** of expert pages were reused across this token transition. The
next-token path spent 4.67 seconds loading source slices. This is the most
important negative result in B1:

> Exact reactive page faults are feasible, but a cache sized to one token is not
> a viable decoding policy when expert routing changes substantially.

The behavior validates the original emphasis on prompt-level prediction,
relationship-aware prefetch, and a cache holding a coherent future working set.
It also suggests that “eight active experts per layer” understates the physical
working set required over an autoregressive window.

## Engineering observations

- Preserving checkpoint tensor dtypes matters. Letting the meta model's default
  FP32 dtype control loading doubled scaffold allocation from 4.90 GB to 9.79
  GB and increased cold latency. Explicit BF16/FP32 preservation fixed both.
- The installed Transformers build lacks the optional Flash Linear Attention
  and causal-conv1d fast path, so 243 ms warm latency is a correctness baseline,
  not an optimized throughput result.
- Safetensors expert slicing materializes exactly 6 MiB per request; packed
  256-expert tensors are never eagerly loaded.
- Pinned staging tensors remain alive until their H2D readiness events complete,
  avoiding asynchronous source-lifetime bugs.
- The first source access has substantial initialization cost; steady accesses
  depend strongly on OS page-cache state and shard locality.

## Next experiment

B2 should measure the cumulative expert union across 8–32 decoded tokens with
cache capacities of 320, 640, 1,280, and 2,560 pages. It should report:

- per-token and per-layer overlap;
- unique-page working-set growth;
- LRU hit rate and transferred bytes;
- quality and latency with exact routing;
- an oracle next-token prefetch upper bound; and
- a predictor using current router/hidden state to fetch later-layer experts.

The immediate systems optimization is background NVMe loading plus contiguous
expert superpages. The immediate modeling optimization is sequence-level
working-set prediction. Both can now be evaluated against a real 35B/3B-active
model rather than the toy fixture.

