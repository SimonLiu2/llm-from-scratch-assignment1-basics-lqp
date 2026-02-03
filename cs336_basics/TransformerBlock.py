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

from .RoPE import RoPE
from .RMSNorm import RMSNorm

from .Attention import attention
from .Linear import Linear
from .RoPE import RoPE
from .MultiHeadAttention import MultiHeadAttention
from .SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: Int, 
                 num_heads: Int, 
                 d_ff: Int,
                 d_in: Int,
                 is_rope: bool = False,
                 theta: float = 100000.0,
                 max_seq_len: int = 2048,
        ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, d_in, is_rope, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_ff, d_model)
    
    def forward(self, x: Float[Tensor, " batch sequence_length d_in"],
                token_pos: Int[Tensor, " batch sequence_length"] = None):
        # Multi-head attention
        add_x = self.ln1(x)
        add_x = self.attn(add_x, token_pos)
        x = x + add_x
        # Feed-forward network
        add_x = self.ln2(x)
        add_x = self.ffn(add_x)
        x = x + add_x
        return x