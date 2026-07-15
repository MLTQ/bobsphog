# `test_catalog_oracle.py`

## Purpose

Protects global page indexing and direct-loss greedy selection semantics.

## Components

### Catalog test

- **Does**: Verifies global IDs produce exactly the intended number of layer
  selections, masks, and names.

### Oracle test

- **Does**: Confirms the greedy oracle returns unique pages at its fixed budget
  and does not increase calibration loss on the fixture.
- **Interacts with**: `PageCatalog`, `ToyTransformer`, and `greedy_oracle_selection`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A3 experiment | Global and layer-local page counts agree | Catalog mapping changes |
| Oracle baseline | Greedy additions minimize measured loss | Selection rule changes |
