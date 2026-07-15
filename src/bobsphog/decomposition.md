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

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `model.py` | `active_pages=None` executes every page | Changing full-path semantics |
| `model.py` | Empty page IDs execute only the resident base | Changing empty-selection semantics |
| `paging.py` traces | Byte counts reflect stored factor tensors | Quantization or storage-layout changes |
| Tests | Full page set reconstructs the source matrix to numerical tolerance | Decomposition algorithm changes |

## Notes

- Bias is resident and counted with the base.
- Page ordering follows descending singular values.
- Physical device paging is not implemented yet; page selection currently
  simulates residency while all parameters remain allocated by PyTorch.
