from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.factory_kwargs = factory_kwargs
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((dim,), **factory_kwargs))
    
    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x / (x.shape[-1] ** 0.5)
        x_normalized = x / torch.sqrt((rms_x**2 + self.eps))
        output = x_normalized * self.weight
        return output.to(in_dtype)
    