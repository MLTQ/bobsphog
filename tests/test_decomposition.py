import torch
from torch import nn
from torch.nn import functional as F

from bobsphog.decomposition import PagedLinear


def test_full_pages_reconstruct_dense_linear() -> None:
    torch.manual_seed(1)
    dense = nn.Linear(7, 5, dtype=torch.float64)
    paged = PagedLinear.from_linear(dense, base_rank=2, page_rank=2)
    inputs = torch.randn(3, 4, 7, dtype=torch.float64)

    assert paged.page_count == 2
    torch.testing.assert_close(paged.effective_weight(), dense.weight, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(paged(inputs), F.linear(inputs, dense.weight, dense.bias), rtol=1e-12, atol=1e-12)


def test_svd_prefix_reconstruction_error_is_non_increasing() -> None:
    torch.manual_seed(2)
    dense = nn.Linear(11, 9, bias=False, dtype=torch.float64)
    paged = PagedLinear.from_linear(dense, base_rank=1, page_rank=2)

    errors = []
    for page_count in range(paged.page_count + 1):
        approximation = paged.effective_weight(range(page_count))
        errors.append(torch.linalg.matrix_norm(dense.weight - approximation).item())

    assert all(next_error <= error + 1e-12 for error, next_error in zip(errors, errors[1:]))
    assert errors[-1] < 1e-12


def test_invalid_page_ids_are_rejected() -> None:
    dense = nn.Linear(4, 4)
    paged = PagedLinear.from_linear(dense, base_rank=1, page_rank=1)

    try:
        paged.effective_weight([0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate page IDs should fail")


def test_direct_factorized_initialization_has_requested_layout() -> None:
    paged = PagedLinear.random_factorized(
        7,
        11,
        base_rank=3,
        page_rank=2,
        page_count=4,
    )
    inputs = torch.randn(5, 7)

    assert paged.base.rank == 3
    assert tuple(page.rank for page in paged.pages) == (2, 2, 2, 2)
    assert paged(inputs).shape == (5, 11)
