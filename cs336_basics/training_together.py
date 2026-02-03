from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch.optim as optim
from .TransformerLM import TransformerLM
from .DataLoader import get_batch
import numpy as np

# recommend d_ff = 8/3 * d_model
model_config = {
    'vocab_size': 10000, 
    'context_size': 256,
    'd_model': 512,
    'num_layers': 4,
    'num_heads': 32,
    'd_ff': 1344,
    'rope_theta': 10000,
    'token_positions': None,
}

optimizer_config = {
    "lr": 1e-2,
    "weight_decay": 0.01,
}

dataloader_config = {
    'batch_size': 32,
    'context_length': model_config['context_size'],
    'device': 'cuda',  # Assuming using GPU
}



def train_and_evaluate(train_data: npt.NDArray, val_data: npt.NDArray, 
                       is_save: bool = False) -> None:
    loss = nn.CrossEntropyLoss()
    model = TransformerLM(**model_config)
    optimizer = optim.AdamW(model.parameters(), **optimizer_config)
    num_epochs = 40000
    model.to('cuda')  # Assuming using GPU
    
    total_train_loss = 0.0
    total_val_loss = 0.0
    print_epoch = 1000
    for epoch in range(num_epochs):
        model.train()
        inputs, targets = get_batch(train_data,
                  batch_size=dataloader_config['batch_size'],
                  context_length=dataloader_config['context_length'], 
                  device=dataloader_config['device']) # Assuming using GPU 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = loss(outputs.view(-1, model_config['vocab_size']), 
                          targets.view(-1))
        loss_value.backward()
        optimizer.step()
        total_train_loss += loss_value.item()
        avg_train_loss = total_train_loss / print_epoch

        # Validation
        model.eval()
        with torch.no_grad():
            val_inputs, val_targets = get_batch(val_data,
                      batch_size=dataloader_config['batch_size'],
                      context_length=dataloader_config['context_length'], 
                      device=dataloader_config['device'])
            val_outputs = model(val_inputs)
            val_loss_value = loss(val_outputs.view(-1, model_config['vocab_size']), 
                                  val_targets.view(-1))
            total_val_loss += val_loss_value.item()
            avg_val_loss = total_val_loss / print_epoch
        if (epoch + 1) % print_epoch == 0:
            percent_complete = (epoch + 1) / num_epochs * 100
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"{percent_complete:.2f}% complete, "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
            #clear total losses
            total_train_loss = 0.0
            total_val_loss = 0.0
    if is_save:
        torch.save(model.state_dict(), '/home/std10/extend/transformer_lm_model.pth')

if __name__=='__main__':
    train_path = '/home/std10/extend/generated_data/tokenized_data_train.npy'
    val_path = '/home/std10/extend/generated_data/tokenized_data_valid.npy'
    # Assuming train_data and val_data are numpy arrays of token IDs
    train_data = np.load(train_path, allow_pickle=True, mmap_mode='r')
    val_data = np.load(val_path, allow_pickle=True, mmap_mode='r')
    train_and_evaluate(train_data, val_data, is_save=True)


