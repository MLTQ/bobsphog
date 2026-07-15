# `dense_model.py`

## Purpose

Provides a genuinely dense A2 teacher with the same embeddings, attention,
normalization, block topology, and output head as the paged student.

## Components

### `DenseMLP`

- **Does**: Implements the teacher's ordinary dense FFN.

### `DenseTransformerBlock`

- **Does**: Combines shared resident-attention architecture with the dense FFN.
- **Interacts with**: `CausalSelfAttention` and `ToyConfig` in `model.py`.

### `DenseToyTransformer`

- **Does**: Trains and evaluates the full-capacity teacher.
- **Interacts with**: `convert_dense_to_paged` in `conversion.py` and A2 training.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `conversion.py` | Block and parameter topology mirrors `ToyTransformer` | Layer naming or topology changes |
| `a2.py` | Forward returns `ModelOutput.logits` | Output contract changes |
| Conversion test | Dense output equals the student's full-page output | Unsupported teacher operations |
