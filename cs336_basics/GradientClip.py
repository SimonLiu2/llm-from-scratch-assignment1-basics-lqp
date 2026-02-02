from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> None:
    """Clips gradient norm of an iterable of parameters.
    Args:
        params : Iterable of model parameters.
        max_norm (float): Maximum norm value.
    """
    grad = [p.grad.data for p in params if p.grad is not None]
    l2_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grad))
    if l2_norm > max_norm:
        clip_coef = max_norm / (l2_norm + 1e-6)
        for g in grad:
            g.mul_(clip_coef)

    