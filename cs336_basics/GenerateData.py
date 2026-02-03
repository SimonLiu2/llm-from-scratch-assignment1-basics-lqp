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

from .train_bpe import train_bpe, fast_train_bpe
from .tokenizer import Tokenizer
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def _tokenize_part(payload: tuple[Tokenizer, str]) -> list[int]:
    tokenizer, part_text = payload
    return tokenizer.encode(part_text)

@timer
def train_tokenizer(input_path: str, vocab_size: int, 
                    special_tokens=["<|endoftext|>"]) -> Tokenizer:
    # Dummy function for tokenizer training
    vocab_size = vocab_size
    vocab, merges = fast_train_bpe(input_path, vocab_size, special_tokens=special_tokens)
    return Tokenizer(vocab, merges, special_tokens=special_tokens)

@timer
def tokenize_data(tokenizer: Tokenizer, 
                  input_path: str) -> npt.NDArray:
    import concurrent.futures

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    length = len(text)
    part_size = length // 32
    tokens = []

    nums_workers = min(32, os.cpu_count() or 1)
    parts = [text[i*part_size:(i+1)*part_size] if i < 31 else text[i*part_size:] for i in range(32)]
    args = [(tokenizer, part) for part in parts]

    with concurrent.futures.ProcessPoolExecutor(max_workers=nums_workers) as executor:
        results = list(executor.map(_tokenize_part, args))
    for part_tokens in results:
        tokens.extend(part_tokens)
    return np.array(tokens, dtype=np.int32)

if __name__ == "__main__":
    input_path = '/home/std10/extend/owt_train.txt'
    valid_path = '/home/std10/extend/owt_valid.txt'
    save_dir =  '/home/std10/extend/generated_data'
    vocab_size = 32000
    print('Training tokenizer...')
    tokenizer = train_tokenizer(input_path, vocab_size)
    print('Tokenizing data...')
    tokenized_data = tokenize_data(tokenizer, input_path)
    # Save tokenized data
    np.save(os.path.join(save_dir, 'owt_train.npy'), tokenized_data)
    tokenized_data = tokenize_data(tokenizer, valid_path)
    # Save tokenized data
    np.save(os.path.join(save_dir, 'owt_valid.npy'), tokenized_data)
    # Save tokenizer
    with open(os.path.join(save_dir, 'owt_vocab.txt'), 'w', encoding='utf-8') as f:
        # Save vocab and merges as txt files
        vocab  = tokenizer.vocab
        for i, token in enumerate(vocab):
            f.write(f"{token}\n")
    with open(os.path.join(save_dir, 'owt_merges.txt'), 'w', encoding='utf-8') as f:
        merges = tokenizer.merges
        for merge in merges:
            f.write(f"{merge[0]} {merge[1]}\n")
    
    


    
    
