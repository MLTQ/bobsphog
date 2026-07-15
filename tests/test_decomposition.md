# `test_decomposition.py`

## Purpose

Protects the mathematical contract of the SVD page decomposition.

## Components

### Reconstruction tests

- **Does**: Verifies full pages reproduce the dense layer and ordered prefixes
  monotonically reduce matrix reconstruction error.
- **Interacts with**: `PagedLinear` in `src/bobsphog/decomposition.py`.

### Validation test

- **Does**: Ensures ambiguous duplicate page selections are rejected.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Decomposition implementation | Numerical equality at full rank | Approximate-only full representation |
| Prefix baselines | SVD-value page ordering | Learned or reordered page IDs |
