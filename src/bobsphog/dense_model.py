"""Dense teacher counterpart to the pageable toy transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bobsphog.model import CausalSelfAttention, ModelOutput, ToyConfig


class DenseMLP(nn.Module):
    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.expansion = nn.Linear(config.d_model, config.d_ff)
        self.projection = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dropout(self.projection(F.gelu(self.expansion(inputs))))


class DenseTransformerBlock(nn.Module):
    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = DenseMLP(config)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.mlp(self.mlp_norm(hidden))


class DenseToyTransformer(nn.Module):
    """A dense nanoGPT-like teacher with architecture parity to the student."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DenseTransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("input sequence exceeds configured context length")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return ModelOutput(logits=logits, loss=loss)
