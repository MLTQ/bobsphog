# `evaluation.py`

## Purpose

Measures A2 answer quality across logical page budgets and estimates each page's
domain-specific causal utility by full-model ablation.

## Components

### `TaskMetrics` / `evaluate_domain`

- **Does**: Reports masked loss and accuracy on deterministic domain batches.
- **Interacts with**: Dense and paged toy models plus `TwoDomainArithmetic`.

### `evaluate_budget_curve`

- **Does**: Evaluates both domains for every uniform SVD-prefix budget and adds
  logical residency.

### `evaluate_random_budget_curve`

- **Does**: Estimates expected domain quality and logical residency under fresh
  independent masks at each dropout rate.
- **Rationale**: Learned pages can leave their original SVD priority order, so
  uniform prefixes alone are not a fair multi-budget measurement.

### `page_ablation_utilities`

- **Does**: Measures loss increase after removing exactly one page from the
  full model.
- **Rationale**: Direct ablation is the causal reference for later retrievers.

### `summarize_specialization`

- **Does**: Compares each domain's top utility quartile and complete utility
  vectors.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `a2.py` | Budget rows contain per-domain loss, accuracy, and residency | Result schema changes |
| Future retriever | Positive utility means omission increased loss | Utility sign changes |
| Comparisons | Every budget uses deterministic evaluation samples | Seed behavior changes |

## Notes

Uniform prefixes are a static SVD baseline, not a task-aware selection policy.
Single-page ablations miss higher-order page interactions.
