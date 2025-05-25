from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import os

val = np.memmap("E:/PyCharm 2024.3.5/projects/data/tinystories/train_30M.bin", dtype=np.uint16, mode='r')

SPECIAL_TOKENS = ["[PAD]", "[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]

enc = tiktoken.get_encoding('gpt2')
enc = tiktoken.Encoding(
    name=enc.name,
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens={**enc._special_tokens, **{token: len(enc._mergeable_ranks) + i for i, token in enumerate(SPECIAL_TOKENS)}}
)

print(enc.decode(val[0:300]))
print(enc.decode(val[300:600]))
print(enc.decode(val[600:900]))