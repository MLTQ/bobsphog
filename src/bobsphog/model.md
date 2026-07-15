# `model.py`

## Purpose

Provides a minimal causal transformer that can run the same inputs under full,
base-only, static-budget, or dropout page plans while keeping attention resident.

## Components

### `ToyConfig`

- **Does**: Defines model size and low-rank page granularity.
- **Interacts with**: All model constructors.
- **Rationale**: `factorized_page_count` bypasses dense SVD construction for
  scaled systems benchmarks while preserving the same execution contracts.

### `CausalSelfAttention`

- **Does**: Implements resident multi-head causal attention.
- **Rationale**: The first prototype isolates FFN paging instead of paging every
  subsystem at once.

### `PagedMLP`

- **Does**: Applies paged expansion and projection matrices with a stable layer
  naming scheme.
- **Interacts with**: `PagedLinear` in `decomposition.py` and `PagePlan` in
  `paging.py`.

### `TransformerBlock`

- **Does**: Combines pre-normalized resident attention with a paged FFN.

### `ToyTransformer`

- **Does**: Produces causal LM logits/loss and exposes page layout and parameter
  accounting.
- **Interacts with**: Smoke experiments, training loops, and A3 retrieval.

### `ToyTransformer.hidden_states`

- **Does**: Returns final normalized sequence states under any page plan.
- **Interacts with**: A3 counterfactual label collection and learned selection.
- **Rationale**: The retriever needs a resident computation query without
  changing the language-model output contract.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `smoke.py` | Stable layer IDs from `page_counts()` | Renaming block/MLP IDs |
| Page policies | `plan=None` means all pages | Default forward semantics |
| Metrics | `resident_parameter_bytes` excludes unselected logical pages | Byte-accounting changes |
| Training code | `ModelOutput.loss` is next-token-compatible cross entropy over provided targets | Output or loss shape changes |
| A3 retriever | `hidden_states` uses exactly the same transformer path as `forward` | Separate or differently normalized query path |

## Notes

- This is logical paging: all parameters are still allocated in the PyTorch
  module unless a physical page provider is attached.
- The token embedding and language head share weights.
- Progressive SVD reconstruction is guaranteed at the matrix level, not strict
  monotonic output quality through nonlinear transformer layers.
