from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

class AdamW(torch.optim.Optimizer):
    def __init__(self, 
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01, 
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, lambda_=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Any = None) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lambda_ = group['lambda_']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                m = state['m']
                v = state['v']
                t = state['t']
                t += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad.pow(2)
                lr_t = lr * ((1 - beta2 ** t)**0.5 / (1 - beta1 ** t))
                p.data = p.data - lr_t * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * lambda_ * p.data
                state['m'] = m
                state['v'] = v
                state['t'] = t
        return loss




    
