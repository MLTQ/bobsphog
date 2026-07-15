# `test_compositional_relationship.py`

## Purpose

Checks A4 compositional target semantics and sparse relationship-graph
construction on a minimal model.

## Components

### Task and graph integration test

- **Does**: Recomputes composition answers, builds singleton/pair utilities, and
  verifies fixed-budget selections and nonzero retained interactions.
- **Interacts with**: `CompositionalArithmetic`, `PageCatalog`, and relationship
  graph builder.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A4 training | Shifted answer mask matches five-token clauses | Task layout changes |
| Bundle selector | Retained graph edges represent strongest signed synergy | Edge filtering changes |
