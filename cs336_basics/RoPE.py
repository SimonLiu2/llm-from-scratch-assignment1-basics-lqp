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

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.factory_kwargs = {'device': device}
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = d_k
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, **self.factory_kwargs) / d_k))
        pos_seq = torch.arange(max_seq_len, **self.factory_kwargs)
        freq = torch.outer(pos_seq, freq)  # (max_seq_len, d_k/2)
        self.register_buffer('sin_cache', freq.sin(), persistent=False)
        self.register_buffer('cos_cache', freq.cos(), persistent=False)

    def forward(self, x: Float[Tensor, "... d_k"], token_pos: Int[Tensor, "..."]):
        cos_pos = self.cos_cache[token_pos]  # (..., d_k/2)
        sin_pos = self.sin_cache[token_pos]
        x1, x2 = x[..., ::2], x[..., 1::2]

        x_rotated = x1 * cos_pos - x2 * sin_pos
        y_rotated = x1 * sin_pos + x2 * cos_pos
        out = torch.empty_like(x)
        out[..., ::2] = x_rotated
        out[..., 1::2] = y_rotated
        return out

        



    
    
