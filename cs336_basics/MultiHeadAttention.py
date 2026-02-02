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

from cs336_basics.Attention import attention
from cs336_basics.Linear import Linear
from .RoPE import RoPE

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model: Int, 
                 num_heads: Int, 
                 d_in: Int, 
                 is_rope: bool = False,
                 theta: float = 100000.0,
                 token_positions = None,
                 max_seq_len: int = 2048
        ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # dimension of each head
        self.d_in = d_in
        self.is_rope = is_rope
        self.q_proj = Linear(d_in, self.d_model)
        self.k_proj = Linear(d_in, self.d_model)
        self.v_proj = Linear(d_in, self.d_model)
        self.output_proj = Linear(self.d_model, self.d_model)
        if is_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len=max_seq_len)
            self.token_positions = token_positions
        
    
    def forward(self, x: Float[Tensor, " ... sequence_length d_in"]):
        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # casual mask
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device), diagonal=0).bool()
        # Split into multiple heads
        Q = rearrange(Q, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', 
                      num_heads=self.num_heads)
        K = rearrange(K, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', 
                      num_heads=self.num_heads)
        V = rearrange(V, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', 
                      num_heads=self.num_heads)
        # Apply RoPE if specified
        if self.is_rope:
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)
        # Apply attention on all the projected vectors in batch
        attn_output = attention(Q, K, V, mask=mask)
        # Concatenate heads
        attn_output = rearrange(attn_output, 
                                '... num_heads seq_len d_k -> ... seq_len (num_heads d_k)')
        # Final linear layer
        output = self.output_proj(attn_output)
        return output