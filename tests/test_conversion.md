# `test_conversion.py`

## Purpose

Ensures the full-residency paged student begins as a functional copy of the
dense teacher.

## Components

### Conversion parity test

- **Does**: Compares end-to-end logits after dense-to-SVD-page conversion.
- **Interacts with**: `DenseToyTransformer` and `convert_dense_to_paged`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A2 baseline | Conversion introduces only numerical SVD error | Approximate or lossy full path |
