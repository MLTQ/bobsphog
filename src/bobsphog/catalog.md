# `catalog.py`

## Purpose

Defines one stable global index across every model page and converts global
selections into the per-layer `PagePlan` representation used for execution.

## Components

### `PageRef`

- **Does**: Names one page and records its global ID, layer-local ID, and bytes.

### `PageCatalog`

- **Does**: Validates IDs, builds plans and resident masks, reports bytes/names,
  and defines the static SVD-prefix baseline.
- **Interacts with**: `ToyTransformer.paged_layers` and all A3 selectors.
- **Rationale**: Global IDs make candidates comparable across layers while the
  execution model retains layer-local IDs.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Oracle and retriever | IDs are contiguous and stable for one model layout | Catalog ordering changes |
| Static baseline | Pages are ordered by local SVD rank, then layer order | Sort priority changes |
| Equal-budget evaluation | `parameter_bytes` reflects factor tensors | Storage representation changes |

## Notes

The current experiment budgets page count. All A3 pages have equal byte size;
future heterogeneous pages must use a byte-constrained selector.
