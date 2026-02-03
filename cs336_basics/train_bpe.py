from __future__ import annotations

import os
import regex as re
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from .pretokenization_example import find_chunk_boundaries
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.bpe_fast_heapq import *

def read_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]  # remove empty strings

def word2bytes(word):
    "Convert word string to tuple of bytes"
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)

def count_word(text):
    "Split text into word bytes using GPT2 pattern and count word bytes frequency."
    word_cnt = defaultdict(int)
    for m in PAT.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes)>=2:
            word_cnt[word_bytes]+=1
    return word_cnt

def merge_dicts(dicts):
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged

def count_pair(word_cnt):
    pair_cnt = defaultdict(int)
    for word_bytes,cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1],word_bytes[1:]):
            pair_cnt[pair]+=cnt
    return pair_cnt

def get_max_pair(pair_cnt):
    max_pair, _ = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))  # lexicographic tie-breaker
    return max_pair


def get_basic_vocab(special_tokens):
    vocab={token:bytes([token]) for token in range(256)}

    for i,token in enumerate(special_tokens):
        token_id = 256+i
        vocab[token_id] = token.encode("utf-8")
    return vocab


def apply_merge(word_bytes,merge):
    merged = merge[0]+merge[1]
    i = 0
    new_word_bytes = []
    while i < len(word_bytes):
        # Check for match
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i+1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)

def update_cnt(word_cnt,pair_cnt, merge_pair):

    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt) # copy with defaultdict

    for word_bytes,cnt in word_cnt.items():

        #----------for word cnt ---------------

        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # Keep the original count if the merge not appear in the key
        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes]+=cnt
            continue

        # Use updated key if merge appear
        new_word = apply_merge(word_bytes,merge_pair)
        new_word_cnt[new_word]+=cnt

        #--------for pair cnt ----------------

        # Decrease all old pair counts
        for pair in old_pairs:
            new_pair_cnt[pair]-=cnt
            if new_pair_cnt[pair] ==0:
                del new_pair_cnt[pair]

        # Count new pairs in the new word
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt

    return new_word_cnt,new_pair_cnt


def _pre_tokenize_chunk_worker(args):
    special_tokens, PAT, chunk = args
    pre_tokens = {}
    # split the chunk on special tokens
    parts = re.split("|".join(special_tokens), chunk)
    for part in parts:
        tokens = re.finditer(PAT, part)
        for match in tokens:
            token_bytes = match.group(0).encode('utf-8')
            token_tuple = (token_bytes,)
            pre_tokens[token_tuple] = pre_tokens.get(token_tuple, 0) + 1
    return pre_tokens

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 1. Pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens_dict: dict[tuple[bytes], int] = {}
    # Parallel processing of chunks
    with open(input_path, "rb") as f:
        num_process = 64  # You can adjust this number
        boundaries = find_chunk_boundaries(f, num_process, b'<|endoftext|>')
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            chunk_args.append((special_tokens, PAT, chunk))
        with ProcessPoolExecutor(max_workers=num_process) as executor:
            results = list(tqdm(executor.map(_pre_tokenize_chunk_worker, chunk_args), total=len(chunk_args), desc="Pre-tokenizing"))
        # Merge results
        for d in tqdm(results, desc="Merging pre-tokenized results"):
            for k, v in d.items():
                key = k[0]
                pre_tokens_dict[key] = pre_tokens_dict.get(key, 0) + v
    print("Pre-tokenization completed. Number of unique pre-tokens:", 
          len(pre_tokens_dict))
    vocab = {}
    merges = []
    #bulid initial vocab with special tokens and ascii characters
    for i in range(256):
        vocab[i] = bytes([i])
    for i, special_token in enumerate(special_tokens):
        vocab[i+256] = special_token.encode('utf-8')
    
    current_size = 256 + len(special_tokens)
    current_tokens = {}  #store current tokens to be merged
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    loop_number = 0
    while(current_size < vocab_size):
        if loop_number == 0:
            for key in pre_tokens_dict.keys():
                if key not in current_tokens:
                    current_tokens[key] = list(key)
                tokens = current_tokens[key]
                for i in range(len(tokens)-1):
                    pair  = (tokens[i], tokens[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + pre_tokens_dict[key]
        else:
            for key in modified_words:
                old_tokens = modified_words[key]
                new_tokens = current_tokens[key]
                # remove counts for old tokens
                for i in range(len(old_tokens)-1):
                    pair  = (old_tokens[i], old_tokens[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) - pre_tokens_dict[key]
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                # add counts for new tokens
                for i in range(len(new_tokens)-1):
                    pair  = (new_tokens[i], new_tokens[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + pre_tokens_dict[key] 
        # find the most frequent pair
        # if there are multiple pairs with same frequency, choose one with greatest lex order
        def pair_sorter(item):
            bytes1 = vocab[item[0][0]]
            bytes2 = vocab[item[0][1]]
            return (item[1], bytes1, bytes2)
        # return first pair
        best_pair = max(pair_counts.items(), key=pair_sorter)[0]

        # add new token to vocab
        vocab[current_size] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # update current_tokens
        modified_words = {}
        for key in pre_tokens_dict.keys():
            tokens = current_tokens[key]
            i = 0
            modified_flag = False
            new_tokens = []
            n = len(tokens)
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(current_size)
                    i += 2
                    modified_flag = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if modified_flag:
                # store the modified words before updating
                modified_words[key] = current_tokens[key]
            current_tokens[key] = new_tokens
        current_size += 1
        loop_number += 1
        if current_size % 10 == 0:
            print("Current vocab size:", current_size, end="\r")
    # ajust value order, move special tokens to the front
    new_vocab = {}
    for i in range(vocab_size):
        if i < 256 + len(special_tokens):
            if i < len(special_tokens):
                new_vocab[i] = vocab[i+256]
            else:
                new_vocab[i] = vocab[i - len(special_tokens)]
        else:
            new_vocab[i] = vocab[i]
    vocab = new_vocab
        
    return vocab, merges

def fast_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
):
    train = BPE_Trainer()
    vocab, merges = train.train(
        input_path,
        vocab_size,
        special_tokens,
    )
    return vocab, merges

            





            
