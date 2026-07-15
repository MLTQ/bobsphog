# `b22_collect.py`

## Purpose

Builds the B2.2 supervised routing corpus by running many prompts through the
full-resident Qwen text model and recording exact prefill and autoregressive
decode expert sets.

## Components

### `B22CollectConfig`

- **Does**: Selects the checkpoint, stratified prompt corpus, device, fixed
  output length, and optional pilot limit.

### `load_prompt_corpus`

- **Does**: Validates unique IDs, domain labels, train/validation/test splits,
  and non-empty prompts.

### `run_b22_collect`

- **Does**: Loads one resident Qwen model, reuses it across every prompt,
  performs fixed-length greedy decoding with real recurrent state, and emits
  layer-ordered route traces and selected text.
- **Interacts with**: `ExpertRouteRecorder` and `load_reference_qwen`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| B2.2 predictor | Every trace contains one prefill group and `N-1` decode groups per layer | Variable or omitted decode steps |
| Split evaluation | Corpus split labels are preserved verbatim | Reassigning records during collection |
| Route parity | Greedy tokens use the resident exact model | Sampling or approximate experts |

## Notes

The collector intentionally ignores EOS until the fixed output length is
reached so every example supplies the same number of route transitions. It is a
research corpus generator, not a user-facing generation API.

