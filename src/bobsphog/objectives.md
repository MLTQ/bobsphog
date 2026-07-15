# `objectives.py`

## Purpose

Defines answer-masked task, accuracy, and teacher-distillation calculations used
at every page budget.

## Components

### `masked_cross_entropy`

- **Does**: Computes mean causal classification loss only at selected targets.
- **Interacts with**: `SyntheticBatch.answer_mask` from `synthetic.py`.

### `masked_cross_entropy_per_example`

- **Does**: Returns one mean answer loss per example for counterfactual utility
  labels.
- **Interacts with**: A3 label collection in `utility_data.py`.

### `masked_kl_divergence`

- **Does**: Computes temperature-scaled teacher-to-student KL at selected
  targets.
- **Interacts with**: Partial and full student paths in `a2.py`.

### `masked_accuracy`

- **Does**: Reports top-token answer accuracy.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `a2.py` | Losses are means over `True` mask entries | Reduction changes |
| Tests | Teacher-to-identical-student KL is numerically zero | KL direction or scaling changes |
| Dataset code | Boolean masks align with `(batch, sequence)` | Shape semantics changes |
| A3 utility labels | Per-example losses preserve batch order | Reduction or ordering changes |
