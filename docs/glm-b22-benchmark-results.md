# GLM-5.2 B2.2 benchmark: coherent accuracy, exact but storage-bound inference

## Outcome

The complete 1.506 TB `zai-org/GLM-5.2` checkpoint can answer a public
multiple-choice benchmark through the exact file-backed implementation on the
128 GiB Strix Halo. A deterministic 16-subject zero-shot MMLU cross-section
scored **14/16 (87.5%)**. The 95% Wilson interval is **64.0%–96.5%**, so this is
evidence of coherent benchmark capability rather than a leaderboard-quality
estimate.

The B2.2 prompt-conditioned cache also reduced decode faults and improved token
rate on an exact same-KV control:

- a 320-page reactive cache decoded at 0.0305 token/s;
- a 300-page prefill-frequency bundle decoded at 0.0332 token/s;
- faults fell 11.7% and decode improved **1.088x**; and
- every independently computed predicted-branch top-1 token matched the
  reactive branch.

The bundle's 19.9-second synchronous prefetch did not amortize over four decode
forwards. End-to-end latency was 1.9% worse. Exact GLM execution is possible,
but fragmented expert reads remain the dominant bottleneck.

## Public benchmark protocol

MMLU was selected because it is a public four-choice benchmark spanning 57
academic and professional tasks. The source is the
[official ICLR 2021 repository](https://github.com/hendrycks/test), and the
benchmark design is described in the
[MMLU paper](https://arxiv.org/abs/2009.03300). The official data archive used
for this run had SHA-256:

```text
bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b  data.tar
```

This run deliberately uses a bounded systems protocol:

1. Seed `20260715` selects 16 subjects across the full subject list, then one
   test question per selected subject.
2. GLM's chat template is applied with thinking disabled.
3. The prompt requires exactly one answer letter.
4. Accuracy is the argmax over the next-token logits for A, B, C, and D.
5. No development examples are included.

This is **not** the official full five-shot MMLU score. The selected-example
digest is:

```text
5592ead929086d2b27ebad647fd5dbdce499b7a0741e468b71d275b8933b48cc
```

## Accuracy result

| Metric | Result |
|--------|-------:|
| Subjects / questions | 16 / 16 |
| Correct | 14 |
| Accuracy | **87.5%** |
| 95% Wilson interval | 64.0%–96.5% |
| Random four-choice baseline | 25.0% |
| Real prompt tokens | 1,956 |
| Padded positions | 4,720 |
| Exact forward time | 1,316.84 s |
| Aggregate real prompt throughput | 1.485 token/s |
| Aggregate padded throughput | 3.584 token/s |

The two incorrect answers were the selected human-sexuality and
security-studies questions. The other 14 subjects were correct, including
business ethics, clinical knowledge, college biology, computer security,
European history, government and politics, international law, machine
learning, medical genetics, professional medicine, and world religions.

## Accuracy memory and I/O

The accuracy cache held 256 exact 72-MiB experts, or 18 GiB, so any saturated
layer could execute. The batch touched 18,998 of the possible 19,200
layer-specific expert pages—**98.95% of the entire routed reservoir**—which
explains why a longer prompt-conditioned bundle cannot make prefill cheap.

| Metric | Result |
|--------|-------:|
| Peak PyTorch allocation | 59,991,288,320 bytes (55.87 GiB) |
| Expert requests / misses / hits | 18,998 / 18,998 / 0 |
| Expert traffic | 1,434,300,973,056 bytes (1.304 TiB) |
| Source-load time | 1,243.25 s |
| Effective source rate | 1.154 GB/s |
| Cache evictions | 18,742 |

An attempted one-batch run over all 57 selected subjects contained 7,139 real
tokens and 21,375 padded positions. ROCm cleanly rejected a 24-MiB allocation
before a full expert layer could fit. The host did not freeze or reboot. The
validated 16-subject batch used 4,720 padded positions and retained roughly
5 GB of GTT headroom during the steady-state sweep.

## B2.2 decode control

The speed prompt was:

```text
Explain in one sentence why the sky appears blue.
```

Its formatted chat prompt contained 17 tokens. Prefill took 350.84 seconds,
moved 385.19 GB across 5,102 compulsory page faults, and ran at 0.0485 prompt
token/s. The selected five-token continuation was:

```text
The sky appears blue because
```

Both decode branches shared deep-copied exact prefill KV state. The reactive
branch established the greedy token path. The B2.2 branch was forced along the
same input tokens while independently recomputing and checking every top-1
prediction.

| Metric | Reactive LRU | B2.2 prefill bundle |
|--------|-------------:|---------------------:|
| Total cache pages | 320 | 320 |
| Pinned pages | 0 | 300 |
| Peak allocation | 61.70 GB | 61.66 GB |
| Decode forwards | 4 | 4 |
| Decode hits | 0 | 280 |
| Decode misses | 2,400 | 2,120 |
| Decode traffic | 181.19 GB | 160.05 GB |
| Decode time | 131.21 s | 120.57 s |
| Decode rate | 0.0305 token/s | **0.0332 token/s** |
| Top-1 agreement | reference path | **100%** |

The prompt bundle uses the deployable B2.2 prefill-reuse baseline: each of 75
sparse layers receives four pinned experts ranked by this prompt's prefill
route frequency. This is weaker than the three-neighbor Qwen predictor because
no diverse GLM route corpus or full-resident GLM reference exists.

The 300-page bundle moved 22.65 GB and took 19.88 seconds to prefetch. Total
latency was therefore 482.04 seconds for reactive LRU versus 491.29 seconds for
the predicted bundle. At the measured average decode savings, prefetch breaks
even around the eighth decode forward; longer outputs should retain the 8.8%
steady decode advantage.

## What B2.2 changed relative to B1.5

The original GLM B1.5 probe used 16 pages (1.125 GiB). This run safely raised
the short-prompt cache to 320 pages (22.5 GiB), a 20x increase, while keeping
the 34.65-GiB scaffold resident. Measured GTT use stayed below the 61.42-GiB
aperture.

Cache size alone did not create decode hits: one token requests 600 pages and
the layer order evicts a 320-page global LRU before the next token reaches the
same layers. B2.2's cross-layer pinned bundle is what produced the 280 exact
hits. The comparison to B1.5's 42.49-second one-token probe is not a controlled
speedup measurement because the prompt and Linux file-cache state differ.

The cache loader was also changed to stream fixed-size expert pages one at a
time, overlapping the preceding device copy and reaping completed pinned
staging between reads. During the 1.434-TB accuracy sweep, ordinary process RSS
remained below roughly 2 GiB instead of materializing an 18-GiB saturated layer
in host staging.

## Decision

| Question | Result | Decision |
|----------|--------|----------|
| Can exact GLM-5.2 answer a public benchmark here? | 14/16 MMLU sample | Pass |
| Is the result a full MMLU estimate? | Wide interval; zero-shot subset | No |
| Does a larger exact cache fit? | 320 pages / 22.5 GiB for short prompt | Pass |
| Does B2.2 reduce decode faults? | 11.7% | Pass |
| Does B2.2 improve steady decode? | 1.088x | Pass |
| Does four-token end-to-end latency improve? | 1.9% worse | Fail |
| Is output parity preserved? | 100% top-1 on forced path | Pass |
| Is exact interactive latency practical? | 350.8-s TTFT, 0.033 token/s | Fail |

The next speed gate is storage layout rather than more reactive cache. GLM
needs contiguous expert superpages, background reads, and a route corpus for a
confidence-gated predictor. A longer generation should then test whether the
measured eighth-forward break-even survives beyond this short control.

## Reproduction

```bash
# Accuracy cross-section validated on this Strix Halo:
python -m bobsphog.glm_b22_benchmark \
  --checkpoint /srv/models/GLM-5.2 \
  --mmlu-root /workspace/data/mmlu \
  --mode accuracy \
  --accuracy-cache-pages 256 \
  --subject-limit 16 \
  --accuracy-batch-size 16 \
  --seed 20260715

# Same-KV reactive versus B2.2 decode control:
python -m bobsphog.glm_b22_benchmark \
  --checkpoint /srv/models/GLM-5.2 \
  --mode speed \
  --speed-cache-pages 320 \
  --pinned-pages 300 \
  --decode-forwards 4
```

## Artifacts

- `outputs/glm-b22-mmlu16-strix.json`: complete per-question logits, answers,
  timing, memory, and cache telemetry.
- `outputs/glm-b22-speed-strix.json`: prefill and same-KV decode comparison.
