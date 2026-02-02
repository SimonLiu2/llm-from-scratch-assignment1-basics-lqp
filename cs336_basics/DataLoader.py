from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Randomly sample starting indices for the input sequences
    start_indices = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=batch_size
    )

    # Initialize tensors to hold the input sequences and labels
    input_sequences = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    labels = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)

    # Populate the input sequences and labels
    for i, start_idx in enumerate(start_indices):
        input_sequences[i] = torch.tensor(dataset[start_idx:start_idx + context_length], dtype=torch.long)
        labels[i] = torch.tensor(dataset[start_idx + 1:start_idx + context_length + 1], dtype=torch.long)

    return input_sequences, labels