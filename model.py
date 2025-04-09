import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.2
    drop_path_rate: float = 0.1
    bias: bool = False  # Можно включить если нужно


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, offset=0):
        seq_len = x.size(1)
        t = torch.arange(offset, offset + seq_len, dtype=self.inv_freq.dtype, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, seq_len, 1, self.dim)
        sin = emb.sin().view(1, seq_len, 1, self.dim)
        return (x * cos) + (self.rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)),
                                 persistent=False)

    def forward(self, x, past_key_values=None, use_cache=False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        offset = past_key_values[0].size(2) if past_key_values else 0
        q = self.rope(q, offset)
        k = self.rope(k, offset)

        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=2)
            v = torch.cat([past_key_values[1], v], dim=2)
        new_key_values = (k, v) if use_cache else None

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_key_values


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return checkpoint(lambda x: self.dropout(self.c_proj(self.gelu(self.c_fc(x)))), x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.drop_path = nn.Dropout(config.drop_path_rate) if config.drop_path_rate > 0 else nn.Identity()

    def forward(self, x, past_key_values=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln_1(x), past_key_values, use_cache)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return (x, new_kv) if use_cache else x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_key_values=None, use_cache=False):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_key_values = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values else None
            x, kv = block(x, past_kv, use_cache)
            new_key_values.append(kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return (logits, new_key_values) if use_cache else logits

    def configure_optimizers(self, weight_decay=0.1, learning_rate=3e-4):
        decay = set()
        no_decay = {'bias', 'LayerNorm.weight'}

        params = [
            {'params': [p for n, p in self.named_parameters() if not p.requires_grad], 'lr': 0},  # Замороженные
            {'params': [p for n, p in self.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.named_parameters() if
                        p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay}
        ]
        return torch.optim.AdamW(params, lr=learning_rate)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        past_key_values = None

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, past_key_values = self(idx_cond, past_key_values, use_cache=True)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx