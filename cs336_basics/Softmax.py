from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Softmax(nn.Module):
    def __init__(self, dim: Int = -1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.factory_kwargs = factory_kwargs
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x_exp = torch.exp(x - torch.max(x, dim=self.dim, keepdim=True).values)
        x_sum = torch.sum(x_exp, dim=self.dim, keepdim=True)
        return x_exp / x_sum