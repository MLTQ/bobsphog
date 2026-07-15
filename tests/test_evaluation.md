# `test_evaluation.py`

## Purpose

Checks that random-budget evaluation spans base-to-full residency and that
single-page ablation covers the complete logical page layout.

## Components

### Evaluation integration test

- **Does**: Runs two mask densities and all page ablations on a minimal model.
- **Interacts with**: `evaluate_random_budget_curve`,
  `page_ablation_utilities`, and `ToyTransformer`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A2 reporting | Dropout 1.0 is base-only and 0.0 is full residency | Dropout semantics |
| Future retriever labels | Every page receives one ablation utility | Page-key or ablation changes |
