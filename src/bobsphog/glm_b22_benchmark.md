# `glm_b22_benchmark.py`

## Purpose

Measures exact file-backed GLM-5.2 on a public multiple-choice benchmark and a
controlled decode comparison that adapts the B2.2 prompt-bundle policy to the
model's much larger expert pages.

## Components

### MMLU sample loader and scorer

- **Does**: Selects a deterministic subject-stratified sample from the official
  MMLU test CSVs, formats zero-shot direct-choice chat prompts, and scores the
  single-token A/B/C/D logits.
- **Bounded suites**: When a subject limit is required for memory or time, the
  subjects are seeded-sampled across the complete sorted subject list rather
  than taking an alphabetically biased prefix.
- **Rationale**: The validated Strix default samples one item from 16 seeded
  subjects in one batched prefill. Larger devices can raise the subject limit
  toward all 57 tasks. Every result reports a Wilson interval and is never
  labeled as the full five-shot MMLU score.

### `GlmRoutePinnedCache`

- **Does**: Retains detached router index tensors for an explicitly delimited
  prefill and aggregates per-layer expert frequency after the synchronized
  forward.
- **Rationale**: This supplies the B2.2 prefill-reuse baseline without adding a
  host synchronization at every sparse layer.

### Decode control

- **Does**: Runs one reactive large-cache branch and one equal-layer
  prefill-frequency bundle on deep-copied prefill KV state. The predicted branch
  follows the baseline input path and independently checks every top-1 token.
- **Memory plan**: The accuracy batch uses 256 72-MiB pages. The short speed
  prompt uses 320 pages (22.5 GiB), of which 300 are pinned, leaving 20 pages
  for an eight-expert atomic layer request.
- **Validated batch bound**: Sixteen sampled subjects produced 4,720 padded
  positions and fit. A single 57-subject/21,375-position batch exhausted the
  61.42-GiB ROCm aperture, so 16 is the safe default on this host.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Official MMLU data | Six-column `*_test.csv` files and A-D labels | Different schema |
| GLM tokenizer | A, B, C, and D each encode as one token | Multi-token choice labels |
| B2.2 bundle | Every sparse layer receives a near-equal quota | Global selection that starves layers |
| Exact comparison | Both decode branches share the same prefill state and input path | Independent sampling |
| Memory safety | Cache plus scaffold stays below measured Strix Halo GTT | Raising defaults without re-measuring headroom |

## Interpretation

The accuracy phase is a reproducible small-sample systems benchmark, not a
leaderboard submission. The speed phase tests the deployable B2.2 prefill-reuse
baseline; it is weaker than the three-neighbor predictor trained for Qwen
because no GLM route corpus or full-resident GLM reference exists.
