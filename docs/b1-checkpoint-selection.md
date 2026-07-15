# B1 checkpoint selection: Qwen3.6-35B-A3B

## Decision

The first meaningful pretrained target is
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
not the earlier dense Qwen2.5-1.5B candidate.

The dense 1–3B stage remains useful as a cheap compatibility fixture, but it no
longer carries the main research claim. Qwen3.6 is the first target that is both
too large for the 24 GB accelerator and sparse enough to expose an executable
query-specific working set.

## Why this model fits the hypothesis

The official model reports:

- 35B total parameters and 3B activated parameters;
- 40 language layers;
- 256 routed experts per layer;
- eight routed experts plus one shared expert active per token;
- hidden size 2,048 and expert intermediate size 512;
- 30 Gated DeltaNet layers and 10 full-attention layers; and
- an Apache-2.0 license.

The Hugging Face checkpoint stores each layer's routed experts in two packed
tensors:

```text
experts.gate_up_proj [256, 1024, 2048]
experts.down_proj    [256, 2048,  512]
```

One BF16 expert page therefore contains:

$$
2(512)(2048) + (2048)(512) = 3{,}145{,}728
$$

parameters, or exactly 6 MiB. One token's routed set is 48 MiB per layer. The
full routed-expert reservoir contains 10,240 such pages across 40 layers, while
the routers, shared experts, embeddings, attention/DeltaNet path, norms, and
output head can form the resident scaffold.

This layout gives us semantic pages learned by the original model rather than
invented post-hoc blocks. It also exposes the central research problem directly:
predict, prefetch, retain, and evict a small expert working graph without ever
placing the 67–72 GB BF16 checkpoint in GPU or host RAM as one object.

## Hardware fit

The remote host provides:

- RTX 4090 with 24 GB VRAM;
- 54 GiB system RAM;
- about 328 GB free on the home volume; and
- a Samsung 970 EVO Plus NVMe backing the checkpoint cache.

The full BF16 checkpoint is about 71.9 GB, so ordinary eager loading would
exceed available RAM and VRAM. Read-only safetensor mmap plus expert slicing is
therefore a requirement, not merely a simulated optimization.

## Staged implementation

1. Read config and shard index only.
2. Download the two shards containing layer 0 and validate one-expert slicing.
3. Build a bounded CUDA expert cache and compare its routed output with the
   reference expert loop.
4. Instantiate the architecture on the meta device, replace packed expert
   modules with provider-backed shells, and load only non-expert tensors.
5. Run exact layer-by-layer routing with cold expert faults.
6. Measure expert overlap across prompts/tokens, then train an early predictor
   to prefetch later-layer expert bundles.

## First layer-0 result

The mmap source materialized one expert as the expected BF16 shapes and exactly
6 MiB of tensor data. A cache run over eight experts reported:

- 48 MiB CUDA working set;
- 38.9 ms total source-load time from warm filesystem cache;
- 2.8 ms for outstanding transfer plus routed execution;
- maximum absolute error `2.44e-4`; and
- mean absolute error `3.50e-6` against the reference BF16 expert loop.

The small difference is within BF16 accumulation rounding. The next parity test
will compare logits through the actual model layer and record top-token
agreement in addition to absolute error.

