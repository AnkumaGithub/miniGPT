import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # На уровне gpt 2
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.2
    drop_path_rate: float = 0.1
    bias: bool = False
    lora_rank: int = 12


class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        # Freeze original weights
        self.linear.weight.requires_grad_(False)
        if bias:
            self.linear.bias.requires_grad_(False)

        # LoRA initialization
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x))


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
        sin = freqs.sin()
        cos = freqs.cos()

        # Повторяем для парных элементов
        sin = sin.repeat(1, 2)
        cos = cos.repeat(1, 2)

        sin = sin.view(1, seq_len, 1, self.dim)
        cos = cos.view(1, seq_len, 1, self.dim)

        x_rot = (x * cos) + (self.rotate_half(x) * sin)
        return x_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        self.c_attn = LoRALinear(config.n_embd, 3 * config.n_embd, rank=config.lora_rank)
        self.c_proj = LoRALinear(config.n_embd, config.n_embd, rank=config.lora_rank)
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)),
                                 persistent=False)  # Исправлено

    def forward(self, x, past_key_values=None, use_cache=False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE with offset for past tokens
        offset = past_key_values[0].size(2) if past_key_values else 0
        q = self.rope(q, offset)
        k = self.rope(k, offset)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_key_values = (k, v) if use_cache else None

        if self.flash and past_key_values is None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if self.flash:
                mask = None
            else:
                if past_key_values is None:
                    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                else:
                    L = k.size(2)
                    mask = torch.ones(T, L, dtype=torch.bool, device=x.device)
                    mask = torch.tril(mask, diagonal=offset)
                mask = mask.view(1, 1, T, L)
                att = att.masked_fill(~mask, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_key_values

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        assert not torch.isnan(input).any(), "Обнаружены NaN в активациях"
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Работа с многомерными тензорами
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        x = x / keep_prob  # Масштабирование для сохранения матожидания
        return x * mask

class SwiGLU(nn.Module):
    def forward(self, x):
        assert not torch.isnan(x).any(), "Обнаружены NaN в активациях"
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)  # Корректная реализация SwiGLU

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int((4 * config.n_embd) * 2 / 3)
        hidden_dim = hidden_dim + (8 - hidden_dim % 8)  # Round to multiple of 8

        self.c_fc = LoRALinear(config.n_embd, 2 * hidden_dim, rank=config.lora_rank)
        self.swiglu = SwiGLU()
        self.c_proj = LoRALinear(hidden_dim, config.n_embd, rank=config.lora_rank)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)  # Весь тензор передается в SwiGLU
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Добавляем DropPath
        self.drop_path1 = DropPath(config.drop_path_rate)
        self.drop_path2 = DropPath(config.drop_path_rate)

    def forward(self, x, past_key_values=None, use_cache=False):
        # Применяем DropPath к выходу self-attention
        attn_out, new_key_values = self.attn(self.ln_1(x), past_key_values, use_cache)
        x = x + self.drop_path1(attn_out)
        # Применяем DropPath к выходу MLP
        mlp_out = self.mlp(self.ln_2(x))
        x = x + self.drop_path2(mlp_out)

        return x, new_key_values


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if 'lora_' in pn:
                if 'lora_A' in pn:
                    nn.init.normal_(p, std=0.02)
                elif 'lora_B' in pn:
                    nn.init.zeros_(p)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if not hasattr(module, 'lora_A'):  # Инициализация только оригинальных весов
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, past_key_values=None, use_cache=False):
        B, T = idx.size()
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        new_key_values = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values else None
            x, kv = block(x, past_kv, use_cache)
            new_key_values.append(kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return (logits, new_key_values) if use_cache else logits

    def configure_optimizers(self, weight_decay=0.1, learning_rate=3e-4):
        # Разделение параметров с весом и без
        decay_params = []
        no_decay_params = []
        for pn, p in self.named_parameters():
            if p.requires_grad:  # Только обучаемые параметры
                if 'lora_' in pn or 'bias' in pn:
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        past_key_values = None

        for _ in range(max_new_tokens):
            if idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
                past_key_values = [(
                    kv[0][:, :, -self.config.block_size:, :],
                    kv[1][:, :, -self.config.block_size:, :]
                ) for kv in past_key_values] if past_key_values else None
            else:
                idx_cond = idx

            logits, past_key_values = self(idx_cond, past_key_values)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx