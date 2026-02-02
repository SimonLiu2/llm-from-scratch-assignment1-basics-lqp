from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

def CrossEntropyLoss(
    logits: Float[Tensor, " batch_size num_classes"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    def log_softmax(logits):
        max_logits = torch.max(logits, dim=1, keepdim=True).values
        stabilized_logits = logits - max_logits
        exp_logits = torch.exp(stabilized_logits)
        sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
        log_probs = stabilized_logits - torch.log(sum_exp_logits)
        return log_probs
    log_probs = log_softmax(logits)
    batch_size = logits.shape[0]
    loss = -torch.sum(log_probs[torch.arange(batch_size), targets]) / batch_size
    return loss