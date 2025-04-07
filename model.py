import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        batch_size, n_heads, seq_len, head_dim = q.shape
        sin = self.sin[:seq_len]
        cos = self.cos[:seq_len]

        #Повторяем элементы для совмещения размерности
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        #Добавляем размерности для broadcast [1, 1, seq_len, 1, head_dim]
        sin = sin.view(1, 1, seq_len, head_dim)
        cos = cos.view(1, 1, seq_len, head_dim)

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


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
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_key_values=None):
        B, T, C = x.size()

        #Проецируем и разделяем Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        #Изменяем форму для применения RoPE
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        #Применяем ротационные позиционные эмбеддинги
        q, k = self.rope(q, k)

        #Кэширование ключей и значений
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        new_key_values = (k, v) if not self.training else None

        #Вычисляем внимание
        if self.flash and past_key_values is None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            #Ручная реализация с поддержкой кэширования
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            #Создаем маску
            if past_key_values is None:
                mask = self.bias[:, :, :T, :T]
            else:
                mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device)
                mask = torch.tril(mask, diagonal=k.size(2) - T)
                mask = mask.view(1, 1, T, k.size(2))

            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        #Собираем выход
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return (y, new_key_values) if new_key_values is not None else y
