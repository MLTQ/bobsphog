# `test_model.py`

## Purpose

Protects end-to-end toy transformer execution and logical parameter accounting
across page budgets.

## Components

### Execution tests

- **Does**: Exercise full, base-only, and explicit-full paths with causal LM
  logits and loss.
- **Interacts with**: `ToyTransformer`, `PagePlan`, and `PagingTrace`.

### Accounting test

- **Does**: Verifies resident bytes grow with selected pages and equal total
  model parameters for the full plan.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Toy experiments | Base and full paths share one architecture | Separate-model execution design |
| Metrics | Full logical residency equals stored parameter bytes | External or shared page storage changes |
