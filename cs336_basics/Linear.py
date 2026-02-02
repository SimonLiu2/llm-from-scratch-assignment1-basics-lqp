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


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.factory_kwargs = factory_kwargs
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        nn.init.trunc_normal_(self.weight)
    
    def forward(self, input: Tensor) -> Tensor:
        #using torch.nn.init.trunc_normal_ to initialize weights
        output = einsum(input, self.weight, '... d_in, d_out d_in -> ... d_out')
        return output