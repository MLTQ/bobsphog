# `test_b1_capability.py`

## Purpose

Validates the pure numerical comparison used by the full-versus-paged
capability experiment.

## Components

### Exact-parity test

- **Does**: Confirms identical logits yield zero KL, complete top-k overlap,
  and exact greedy-token agreement.

### Changed-decision test

- **Does**: Confirms one altered top-1 decision is localized and lowers the
  reported agreement fraction.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Capability report | Fractions are token-weighted and mismatches are indexed | Aggregate-only comparison |
| Mac CI | Metrics run with tiny CPU tensors | Eager pretrained loading |
