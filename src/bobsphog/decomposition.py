"""Low-rank executable pages and dense-to-paged decomposition."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bobsphog.paging import PageEvent, PagingTrace


class PageProvider(Protocol):
    """Apply an exact page supplied by an external residency runtime."""

    def apply(self, layer_id: str, page_id: int, inputs: Tensor) -> Tensor: ...


class LowRankPage(nn.Module):
    """An independently executable low-rank matrix factorization."""

    def __init__(self, left: Tensor, right: Tensor) -> None:
        super().__init__()
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("page factors must both be matrices")
        if left.shape[1] != right.shape[0]:
            raise ValueError("page factor ranks do not match")
        self.left = nn.Parameter(left.detach().clone())
        self.right = nn.Parameter(right.detach().clone())

    @property
    def rank(self) -> int:
        return self.right.shape[0]

    @property
    def parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(F.linear(inputs, self.right), self.left)

    def materialize(self) -> Tensor:
        return self.left @ self.right


class PagedLinear(nn.Module):
    """A linear layer represented by a resident base and optional residual pages."""

    def __init__(
        self,
        base: LowRankPage,
        pages: Iterable[LowRankPage],
        bias: Tensor | None,
    ) -> None:
        super().__init__()
        self.base = base
        self.pages = nn.ModuleList(pages)
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self._page_provider: PageProvider | None = None

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        base_rank: int,
        page_rank: int,
    ) -> PagedLinear:
        """Factor a dense layer using ordered SVD components."""

        weight = linear.weight.detach()
        max_rank = min(weight.shape)
        if not 0 <= base_rank <= max_rank:
            raise ValueError(f"base_rank must be between 0 and {max_rank}")
        if page_rank <= 0:
            raise ValueError("page_rank must be positive")

        u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)

        def make_page(start: int, stop: int) -> LowRankPage:
            root = singular_values[start:stop].sqrt()
            left = u[:, start:stop] * root.unsqueeze(0)
            right = root.unsqueeze(1) * vh[start:stop, :]
            return LowRankPage(left, right)

        base = make_page(0, base_rank)
        pages = [
            make_page(start, min(start + page_rank, max_rank))
            for start in range(base_rank, max_rank, page_rank)
        ]
        return cls(base, pages, linear.bias)

    @classmethod
    def random_factorized(
        cls,
        in_features: int,
        out_features: int,
        *,
        base_rank: int,
        page_rank: int,
        page_count: int,
        bias: bool = True,
    ) -> PagedLinear:
        """Initialize factors directly without constructing or decomposing a dense matrix."""

        if min(in_features, out_features, base_rank, page_rank, page_count) <= 0:
            raise ValueError("feature sizes, ranks, and page_count must be positive")
        component_count = page_count + 1

        def make_page(rank: int) -> LowRankPage:
            right = torch.randn(rank, in_features) / sqrt(in_features)
            left = torch.randn(out_features, rank) / sqrt(rank * component_count)
            return LowRankPage(left, right)

        resident = make_page(base_rank)
        pages = [make_page(page_rank) for _ in range(page_count)]
        bias_tensor = torch.zeros(out_features) if bias else None
        return cls(resident, pages, bias_tensor)

    @property
    def in_features(self) -> int:
        return self.base.right.shape[1]

    @property
    def out_features(self) -> int:
        return self.base.left.shape[0]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def base_parameter_bytes(self) -> int:
        bias_bytes = 0 if self.bias is None else self.bias.numel() * self.bias.element_size()
        return self.base.parameter_bytes + bias_bytes

    @property
    def page_parameter_bytes(self) -> tuple[int, ...]:
        return tuple(page.parameter_bytes for page in self.pages)

    def normalize_page_ids(self, active_pages: Iterable[int] | None) -> tuple[int, ...]:
        if active_pages is None:
            return tuple(range(self.page_count))
        page_ids = tuple(active_pages)
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("active page IDs must be unique")
        if any(page_id < 0 or page_id >= self.page_count for page_id in page_ids):
            raise IndexError("active page ID is out of range")
        return page_ids

    def set_page_provider(self, provider: PageProvider | None) -> None:
        """Bind an inference-time provider without registering it as a child module."""

        self._page_provider = provider

    def forward(
        self,
        inputs: Tensor,
        *,
        active_pages: Iterable[int] | None = None,
        layer_id: str = "",
        trace: PagingTrace | None = None,
    ) -> Tensor:
        page_ids = self.normalize_page_ids(active_pages)
        output = self.base(inputs)
        for page_id in page_ids:
            if self._page_provider is None:
                contribution = self.pages[page_id](inputs)
            else:
                if not layer_id:
                    raise ValueError("layer_id is required when a page provider is bound")
                contribution = self._page_provider.apply(layer_id, page_id, inputs)
            output = output + contribution
        if self.bias is not None:
            output = output + self.bias

        if trace is not None:
            trace.record(
                PageEvent(
                    layer_id=layer_id,
                    selected_pages=page_ids,
                    base_bytes=self.base_parameter_bytes,
                    page_bytes=tuple(self.page_parameter_bytes[index] for index in page_ids),
                )
            )
        return output

    def effective_weight(self, active_pages: Iterable[int] | None = None) -> Tensor:
        page_ids = self.normalize_page_ids(active_pages)
        weight = self.base.materialize()
        for page_id in page_ids:
            weight = weight + self.pages[page_id].materialize()
        return weight
