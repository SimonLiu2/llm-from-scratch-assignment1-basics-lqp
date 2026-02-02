from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.Linear import Linear

class SiLU(nn.Module):
    def __init__(self, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.factory_kwargs = factory_kwargs
    
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_ff: Int, d_model: Int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.factory_kwargs = factory_kwargs
        self.w1 = Linear(d_model, d_ff, 
                          device=self.factory_kwargs['device'], 
                          dtype=self.factory_kwargs['dtype'])
        self.w2 = Linear(d_ff, d_model,
                            device=self.factory_kwargs['device'], 
                            dtype=self.factory_kwargs['dtype'])
        self.w3 = Linear(d_model, d_ff,
                            device=self.factory_kwargs['device'], 
                            dtype=self.factory_kwargs['dtype'])
        self.activation = SiLU(device=self.factory_kwargs['device'], 
                              dtype=self.factory_kwargs['dtype'])
    
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w3(x)
        activated = self.activation(x1)
        gated = activated * x2
        output = self.w2(gated)
        return output