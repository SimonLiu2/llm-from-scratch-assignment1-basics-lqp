from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Get learning rate using cosine schedule with warmup.
    Args:
        it (int): Current iteration.
        max_learning_rate (float): Maximum learning rate.
        min_learning_rate (float): Minimum learning rate.
        warmup_iters (int): Number of warmup iterations.
        cosine_cycle_iters (int): Number of iterations in one cosine cycle.
    Returns:
        float: Learning rate at current iteration.
    """
    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        cosine_it = it - warmup_iters
        lr = min_learning_rate + 0.5 *  (
            1 + math.cos(math.pi * cosine_it / (cosine_cycle_iters - warmup_iters)
        ))* (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
    return lr