"""Exact conversion from the dense toy teacher to a paged student."""

from __future__ import annotations

from dataclasses import replace

import torch

from bobsphog.decomposition import PagedLinear
from bobsphog.dense_model import DenseToyTransformer
from bobsphog.model import ToyTransformer


@torch.no_grad()
def convert_dense_to_paged(
    teacher: DenseToyTransformer,
    *,
    base_rank: int,
    page_rank: int,
) -> ToyTransformer:
    """Copy shared weights and SVD-factor every dense teacher FFN."""

    config = replace(teacher.config, base_rank=base_rank, page_rank=page_rank)
    reference_parameter = next(teacher.parameters())
    student = ToyTransformer(config).to(
        device=reference_parameter.device,
        dtype=reference_parameter.dtype,
    )
    student.token_embedding.weight.copy_(teacher.token_embedding.weight)
    student.position_embedding.weight.copy_(teacher.position_embedding.weight)
    student.final_norm.load_state_dict(teacher.final_norm.state_dict())

    for dense_block, paged_block in zip(teacher.blocks, student.blocks, strict=True):
        paged_block.attention_norm.load_state_dict(dense_block.attention_norm.state_dict())
        paged_block.attention.load_state_dict(dense_block.attention.state_dict())
        paged_block.mlp_norm.load_state_dict(dense_block.mlp_norm.state_dict())
        paged_block.mlp.expansion = PagedLinear.from_linear(
            dense_block.mlp.expansion,
            base_rank=base_rank,
            page_rank=page_rank,
        )
        paged_block.mlp.projection = PagedLinear.from_linear(
            dense_block.mlp.projection,
            base_rank=base_rank,
            page_rank=page_rank,
        )

    return student.train(teacher.training)
