# B1.5 results: full capability reference versus demand paging

## Result

Qwen3.6-35B-A3B produced the same complete greedy output through the
full-resident and demand-paged implementations on all six initial capability
probes. The comparison ran both paths sequentially on the same Radeon 8060S,
ROCm 7.2.4, PyTorch 2.10 runtime, and BF16 checkpoint.

The paged path reduced measured peak allocation from 69.49 GB to 7.04 GB:

| Path | Peak bytes | Fraction of reference |
|------|-----------:|----------------------:|
| Full text model | 69,490,578,432 | 100% |
| Paged scaffold plus 320 experts | 7,044,852,224 | 10.14% |

This is an **89.86% peak-memory reduction with exact observed greedy-output
parity** on the probe suite.

## Capability probes

The reference generated each answer first. The paged model then underwent two
checks:

1. teacher-forced replay of every generated token with full-distribution
   comparison; and
2. independent autoregressive generation through the real KV/recurrent-state
   and cache-churn path.

| Probe | Full reference | Paged autoregressive | Exact sequence |
|-------|----------------|----------------------|:--------------:|
| Instruction following | `SAPPHIRE` | `SAPPHIRE` | yes |
| Factual recall | `Paris` | `Paris` | yes |
| Arithmetic | `8` | `8` | yes |
| Translation | `Buenos días` | `Buenos días` | yes |
| Science | `Photosynthesis` | `Photosynthesis` | yes |
| Code completion | `y = x ** 2` | `y = x ** 2` | yes |

Both paths passed 6/6 expected-answer checks. All 36 teacher-forced positions
selected the token actually emitted by the reference generator, and all six
independent paged sequences exactly matched the reference sequences.

## Distribution parity

Across 36 generated-token positions:

- mean teacher KL was `0.0008068` per token;
- maximum per-token KL was `0.01757`;
- mean top-5 overlap was high on every probe; and
- all paged teacher-forced top-1 tokens matched the actual reference-generated
  tokens.

Raw reference-logit top-1 agreement was 35/36 because the translation trace's
first raw top-1 differed from the token chosen by the model's generation
pipeline. This is not a behavioral divergence: teacher-forced paged logits
selected the emitted token, and both independent generators produced the exact
same `Buenos días` sequence. Generation processors can make raw-logit top-1 a
different quantity from the final greedy token.

The nonzero KL and absolute-logit differences likely arise from different
expert accumulation/kernel order between the packed reference experts and the
Python paged expert loop. They did not alter any observed generated token.

## Memory accounting

The full control loaded 693 language tensors totaling 69,321,221,376 bytes.
The paged loader retained 613 scaffold tensors totaling 4,896,711,936 bytes and
bounded exact routed experts to 2,013,265,920 bytes (320 six-MiB pages).

| Measurement | Full | Paged |
|-------------|-----:|------:|
| Loaded language/scaffold bytes | 69.32 GB | 4.90 GB |
| Expert cache capacity | included above | 2.01 GB |
| Peak allocation | 69.49 GB | 7.04 GB |
| Load time | 16.54 s | 1.35 s |

This compares against the full **text** model, not the 71.90 GB multimodal
checkpoint file total. Vision and MTP tensors were excluded from both paths.

## The cost: page traffic and churn

Exact capability preservation does not imply an efficient runtime yet. Across
teacher-forced and autoregressive checks, the paged cache recorded:

| Metric | Value |
|--------|------:|
| Expert requests | 48,384 |
| Hits | 2,557 |
| Misses | 45,827 |
| Evictions | 45,507 |
| Bytes transferred | 288,318,554,112 |
| Source-load time | 25.40 s |
| Total paged evaluation time | 46.34 s |

True paged generations took 3.37–4.71 seconds each. Their hit rates remained
low because the cache holds only one token-scale working set and exact routing
changes substantially across prompt prefill and decoding.

This cleanly separates the two research claims:

1. **Memory/capability:** supported by this run. A ~7 GB resident working model
   preserved all observed outputs of the ~69.5 GB full text model.
2. **Efficient demand retrieval:** not solved. Reactive page faults cause very
   high transfer amplification and remain the dominant optimization target.

## Environment and reproducibility

- Hardware: AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`)
- Unified memory: 122 GiB physical, 100 GiB TTM/GTT limit
- Runtime: official AMD ROCm 7.2.4 / PyTorch 2.10 container
- Checkpoint revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Probe configuration: six probes, 12 maximum new tokens, 320 expert pages
- Raw result: `outputs/b1-capability-strix-full.json`

Run with:

```bash
bobsphog-b1-capability \
  --checkpoint /path/to/Qwen3.6-35B-A3B \
  --probe-limit 6 \
  --max-new-tokens 12 \
  --autoregressive-probes 6
```

## Next experiment

Proceed to B2 with capability parity as the invariant. Measure expert-set union
and overlap across longer sequences and larger cache budgets, then add
prompt-level prefetch and relationship-aware bundles. Every optimization should
rerun this parity suite so transfer reductions are never mistaken for preserved
capability.
