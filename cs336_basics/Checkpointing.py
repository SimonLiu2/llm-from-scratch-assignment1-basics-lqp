from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

def save_checkpoint(model, optimizer, iteration, out):
    """Saves a checkpoint of the model and optimizer state.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer whose state to save.
        iteration: The current training iteration.
        out: A file-like object or a string path where the checkpoint will be saved.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    if isinstance(out, str):
        torch.save(checkpoint, out)
    else:
        torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    """Loads a checkpoint into the model and optimizer.

    Args:
        src: A file-like object or a string path from where the checkpoint will be loaded.
        model: The PyTorch model to load the state into.
        optimizer: The optimizer to load the state into.

    Returns:
        The iteration number stored in the checkpoint.
    """
    if isinstance(src, str):
        checkpoint = torch.load(src)
    else:
        checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration
