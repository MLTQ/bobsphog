# `test_retriever.py`

## Purpose

Provides a small end-to-end check for A3 counterfactual collection, estimator
training, and label-free greedy selection.

## Components

### Retriever integration test

- **Does**: Collects direct utilities, trains two optimizer steps, and selects
  two unique pages.
- **Interacts with**: Catalog, utility data, estimator, and toy model.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A3 CLI | Collection-to-selection pipeline executes on CPU | Data or model signature changes |
