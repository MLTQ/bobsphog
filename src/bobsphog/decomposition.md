# `decomposition.py`

## Purpose

Implements the first executable page representation: an ordered low-rank base
plus independently loadable low-rank residuals derived from a dense matrix.

## Components

### `LowRankPage`

- **Does**: Stores $UV^\top$ as two trainable factors and applies it without
  materializing the dense matrix.
- **Interacts with**: `PagedLinear` in this module.
- **Rationale**: Additive low-rank factors are small, executable, and compatible
  with ordinary PyTorch operations.

### `PagedLinear`

- **Does**: Runs a resident base, selected residual pages, and an optional bias.
- **Interacts with**: `PageEvent` and `PagingTrace` in `paging.py`.
- **Rationale**: SVD orders components by reconstruction value, giving the toy
  prototype a deterministic progressive baseline before learned routing.

### `PagedLinear.from_linear`

- **Does**: Converts `torch.nn.Linear` into balanced SVD factors using
  $U\sqrt{S}$ and $\sqrt{S}V^\top$.
- **Interacts with**: `ToyTransformer` construction in `model.py`.

### `PagedLinear.random_factorized`

- **Does**: Creates resident and optional factors directly without an expensive
  dense matrix or SVD.
- **Interacts with**: The scaled A5 physical-paging benchmark.

### `PageProvider` / `PagedLinear.set_page_provider`

- **Does**: Lets an external inference runtime supply exact page computation.
- **Interacts with**: `PhysicalPageCache` in `physical_cache.py`.
- **Rationale**: Source page parameters can remain on CPU while cached copies
  execute on the accelerator.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `model.py` | `active_pages=None` executes every page | Changing full-path semantics |
| `model.py` | Empty page IDs execute only the resident base | Changing empty-selection semantics |
| `paging.py` traces | Byte counts reflect stored factor tensors | Quantization or storage-layout changes |
| Tests | Full page set reconstructs the source matrix to numerical tolerance | Decomposition algorithm changes |
| Physical pager | Provider receives stable `layer_id`, local page ID, and accelerator inputs | Provider signature changes |

## Notes

- Bias is resident and counted with the base.
- Page ordering follows descending singular values.
- Without a provider, selection remains ordinary in-module execution. A bound
  provider is inference-only and owns physical residency.
