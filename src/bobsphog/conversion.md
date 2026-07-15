# `conversion.py`

## Purpose

Creates the A2 paged student from a trained dense teacher without changing the
full-budget function beyond floating-point SVD reconstruction error.

## Components

### `convert_dense_to_paged`

- **Does**: Copies shared tensors and replaces every dense FFN matrix with an
  ordered resident base plus residual pages.
- **Interacts with**: `DenseToyTransformer`, `ToyTransformer`, and `PagedLinear`.
- **Rationale**: Training the teacher before factorization separates learned
  page-budget effects from a factorized-teacher initialization confound.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `a2.py` | Returned student matches teacher at full residency | Approximate conversion |
| Conversion tests | Shared attention and embedding weights are copied | Model topology changes |
| Page baselines | SVD ordering defines prefix priority | Different factorization method |
