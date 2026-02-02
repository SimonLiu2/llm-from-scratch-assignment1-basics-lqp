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

from .train_bpe import train_bpe
from .tokenizer import Tokenizer
import json

def train_tokenizer(input_path: str, vocab_size: int, 
                    special_tokens=["<|endoftext|>"]) -> Tokenizer:
    # Dummy function for tokenizer training
    vocab_size = vocab_size
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens=special_tokens)
    return Tokenizer(vocab, merges, special_tokens=special_tokens)

def tokenize_data(tokenizer: Tokenizer, 
                  input_path: str) -> npt.NDArray:
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    return np.array(tokens, dtype=np.int32)

if __name__ == "__main__":
    input_path = '/home/std10/extend/TinyStoriesV2-GPT4-train.txt'
    valid_path = '/home/std10/extend/TinyStoriesV2-GPT4-valid.txt'
    save_dir =  '/home/std10/extend/generated_data'
    vocab_size = 10000
    print('Training tokenizer...')
    tokenizer = train_tokenizer(input_path, vocab_size)
    print('Tokenizing data...')
    tokenized_data = tokenize_data(tokenizer, valid_path)
    # Save tokenized data
    np.save(os.path.join(save_dir, 'tokenized_data_valid.npy'), tokenized_data)
    # Save tokenizer
    with open(os.path.join(save_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
        # Save vocab and merges as txt files
        vocab  = tokenizer.vocab
        for i, token in enumerate(vocab):
            f.write(f"{token}\n")
    with open(os.path.join(save_dir, 'merges.txt'), 'w', encoding='utf-8') as f:
        merges = tokenizer.merges
        for merge in merges:
            f.write(f"{merge[0]} {merge[1]}\n")
    
    


    
    
