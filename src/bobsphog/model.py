"""A tiny causal transformer whose FFN residual capacity is pageable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bobsphog.decomposition import PagedLinear
from bobsphog.paging import PagePlan, PagingTrace


@dataclass(frozen=True)
class ToyConfig:
    vocab_size: int = 128
    context_length: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.0
    base_rank: int = 8
    page_rank: int = 8
    factorized_page_count: int | None = None

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.context_length <= 0 or self.n_layers <= 0:
            raise ValueError("context_length and n_layers must be positive")
        if self.base_rank < 0 or self.page_rank <= 0:
            raise ValueError("base_rank must be non-negative and page_rank positive")
        if self.factorized_page_count is not None and self.factorized_page_count <= 0:
            raise ValueError("factorized_page_count must be positive when provided")


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None


class CausalSelfAttention(nn.Module):
    """Resident causal self-attention for the toy model."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, channels = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, sequence_length, self.n_heads, self.head_size).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.residual_dropout(self.projection(output))


class PagedMLP(nn.Module):
    """Transformer FFN with pageable expansion and projection matrices."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()

        def make_linear(in_features: int, out_features: int) -> PagedLinear:
            if config.factorized_page_count is None:
                return PagedLinear.from_linear(
                    nn.Linear(in_features, out_features),
                    base_rank=config.base_rank,
                    page_rank=config.page_rank,
                )
            return PagedLinear.random_factorized(
                in_features,
                out_features,
                base_rank=config.base_rank,
                page_rank=config.page_rank,
                page_count=config.factorized_page_count,
            )

        self.expansion = make_linear(config.d_model, config.d_ff)
        self.projection = make_linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: Tensor,
        *,
        layer_prefix: str,
        plan: PagePlan,
        trace: PagingTrace | None,
    ) -> Tensor:
        expansion_id = f"{layer_prefix}.expansion"
        projection_id = f"{layer_prefix}.projection"
        hidden = self.expansion(
            inputs,
            active_pages=plan.selected(expansion_id, self.expansion.page_count),
            layer_id=expansion_id,
            trace=trace,
        )
        hidden = F.gelu(hidden)
        hidden = self.projection(
            hidden,
            active_pages=plan.selected(projection_id, self.projection.page_count),
            layer_id=projection_id,
            trace=trace,
        )
        return self.dropout(hidden)


class TransformerBlock(nn.Module):
    """A pre-normalized resident-attention, paged-FFN transformer block."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = PagedMLP(config)

    def forward(
        self,
        inputs: Tensor,
        *,
        layer_prefix: str,
        plan: PagePlan,
        trace: PagingTrace | None,
    ) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.mlp(
            self.mlp_norm(hidden),
            layer_prefix=f"{layer_prefix}.mlp",
            plan=plan,
            trace=trace,
        )


class ToyTransformer(nn.Module):
    """A nanoGPT-like model instrumented for logical FFN page selection."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def hidden_states(
        self,
        input_ids: Tensor,
        *,
        plan: PagePlan | None = None,
        trace: PagingTrace | None = None,
    ) -> Tensor:
        """Return final normalized states under a logical page plan."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("input sequence exceeds configured context length")

        active_plan = plan or PagePlan.full()
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                layer_prefix=f"blocks.{index}",
                plan=active_plan,
                trace=trace,
            )
        return self.final_norm(hidden)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        plan: PagePlan | None = None,
        trace: PagingTrace | None = None,
    ) -> ModelOutput:
        hidden = self.hidden_states(input_ids, plan=plan, trace=trace)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return ModelOutput(logits=logits, loss=loss)

    def paged_layers(self) -> Mapping[str, PagedLinear]:
        layers: dict[str, PagedLinear] = {}
        for block_index, block in enumerate(self.blocks):
            prefix = f"blocks.{block_index}.mlp"
            layers[f"{prefix}.expansion"] = block.mlp.expansion
            layers[f"{prefix}.projection"] = block.mlp.projection
        return layers

    def page_counts(self) -> Mapping[str, int]:
        return {layer_id: layer.page_count for layer_id, layer in self.paged_layers().items()}

    def total_parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def resident_parameter_bytes(self, plan: PagePlan) -> int:
        layers = self.paged_layers()
        all_page_bytes = sum(sum(layer.page_parameter_bytes) for layer in layers.values())
        fixed_bytes = self.total_parameter_bytes() - all_page_bytes
        selected_bytes = sum(
            sum(layer.page_parameter_bytes[index] for index in plan.selected(layer_id, layer.page_count))
            for layer_id, layer in layers.items()
        )
        return fixed_bytes + selected_bytes
