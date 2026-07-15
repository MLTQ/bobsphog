"""Public package surface for the bobsphog toy prototype."""

from bobsphog.conversion import convert_dense_to_paged
from bobsphog.decomposition import PagedLinear
from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.paging import PagePlan, PagingTrace
from bobsphog.synthetic import TwoDomainArithmetic

__all__ = [
    "DenseToyTransformer",
    "PagePlan",
    "PagedLinear",
    "PagingTrace",
    "ToyConfig",
    "ToyTransformer",
    "TwoDomainArithmetic",
    "convert_dense_to_paged",
]
