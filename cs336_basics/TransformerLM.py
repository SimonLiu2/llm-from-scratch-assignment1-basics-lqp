from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, einsum

from .TransformerBlock import TransformerBlock
from .Embedding import Embedding
from .RMSNorm import RMSNorm
from .Linear import Linear
from .Softmax import Softmax


class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size: Int,
                 context_size: Int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device: str = 'cpu'
                 ):
        super().__init__()
        self.device = device
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleDict()
        for layer in range(num_layers):
            block = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                d_in=d_model,
                is_rope=True,
                theta=rope_theta,
                max_seq_len=context_size,
            )
            self.layers.add_module(f"{layer}", block)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, " batch sequence_length"],
                token_pos: Int[Tensor, " batch sequence_length"] = None):
        x = self.token_embeddings(x)
        for layer in self.layers.values():
            x = layer(x, token_pos=torch.arange(x.shape[1], device=self.device))
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


        
