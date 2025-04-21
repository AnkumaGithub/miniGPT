from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import os

val = np.memmap("data/tinystories/val_256.bin", dtype=np.uint16, mode='r')

encoder = tiktoken.get_encoding('gpt2')
block = val[0:128]

wpe=nn.Embedding(128, 4)
B, T = block.size()
pos = torch.arange(0, T, device="cuda").unsqueeze(0)
pos = torch.clamp(pos, 0, 128 - 1)
pos_emb = wpe(pos)
print(pos_emb)
print(val[0:128])
print(encoder.decode(val[0:128]))