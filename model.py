import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 128
    batch_size: int = 80
    lr: float = 1e-4
    dropout: float = 0.15
    drop_path_rate: float = 0.1
    bias: bool = False  # Можно включить если нужно
    mode: str = 'little_f'
    stride: int = 128
    weight_decay: float = 0.05


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, head_dim, max_seq_len=4096):
        super().__init__()
        assert head_dim % 2 == 0 # dim должен быть чётным
        self.head_dim = head_dim
        # [dim // 2]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)) # Для sin и cos
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1) # Если head_dim нечётный то поломается, можно x1 = x[..., : x.shape[-1] // 2] x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, offset=0):
        seq_len = x.size(-2) # Длина текущего фрагмента последовательности
        # Глобальные позиции токенов [1, seq_len, 1]
        t = torch.arange(offset, offset + seq_len, dtype=self.inv_freq.dtype, device=x.device).unsqueeze(0).unsqueeze(-1)
        freqs = t * self.inv_freq # [1, seq_len, D//2]
        emb = torch.cat((freqs, freqs), dim=-1) #[1, seq_len, D]
        cos = emb.cos().unsqueeze(1) # [1, 1, seq_len, D]
        sin = emb.sin().unsqueeze(1)
        return (x * cos) + (self.rotate_half(x) * sin) #[B, n_head, seq_len, D]


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        # преобразовываем в q k v
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)),
                                 persistent=False)



    def forward(self, x, past_key_values=None, use_cache=False):
        B, T, C = x.size() # Batch, seq_len, emb_dim
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # Разделяем для голов
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # q k v - [Batch, n_head, seq_len, head_dim] тк emb_dim = n_head * head_dim
        if past_key_values is not None: # Складываем кэш
            k_prev, v_prev = past_key_values
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        offset = k.size(2) - T # Если есть kv то k-size() - T скажет с какой позиции обрабатывать
        with torch.amp.autocast(device_type='cuda', enabled=False):   # Для стабильности ротаций
            q = self.rope(q, offset) #[query, offset]
            k = self.rope(k, offset) #[keys, offset]
        new_key_values = (k, v) if use_cache else None

        if self.flash:
            dropout_p = self.attn_dropout.p if self.training else 0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            # Ручная реализация внимания с маской
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            total_len = T + (past_key_values[0].size(2) if past_key_values else 0)
            mask = torch.tril(torch.ones(T, total_len, device=x.device))
            mask = mask[:, -k.size(2):]  # Обрезаем до актуальной длины ключей
            mask = mask.unsqueeze(0).unsqueeze(0)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_key_values

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.activ = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def _forward_impl(self, x):
        return self.dropout(self.c_proj(self.activ(self.c_fc(x))))

    def forward(self, x):
        return checkpoint(self._forward_impl, x, use_reentrant=False)

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        if drop_prob < 0 or drop_prob >= 1:
            raise ValueError('drop_prob should be in [0, 1)')
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = (torch.rand(*mask_shape, device=x.device) > self.drop_prob).to(dtype=x.dtype)
            return x * mask / (1 - self.drop_prob)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0 else nn.Identity()

    def forward(self, x, past_key_values=None, use_cache=False):
        # Attention block
        residual = x
        x = self.ln_1(x)
        attn_out, new_kv = self.attn(x, past_key_values, use_cache)
        x = residual + self.drop_path(attn_out)

        # MLP block
        residual = x
        x = self.ln_2(x)
        mlp_out = self.mlp(x)
        x = residual + self.drop_path(mlp_out)

        return (x, new_kv) if use_cache else x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Дополнительная инициализация для стабильности
        #self._additional_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _additional_init(self): # У модели были огромные градиенты

        for block in self.transformer.h:
            nn.init.xavier_normal_(block.mlp.c_proj.weight, gain=0.02)

        # Инициализация позиционных эмбеддингов
        #nn.init.normal_(self.transformer.wpe.weight, std=0.02)

    def forward(self, idx, past_key_values=None, use_cache=False):
        B, T = idx.size()

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        new_key_values = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values else None
            if use_cache:
                x, kv = block(x, past_kv, use_cache=True)
                new_key_values.append(kv)
            else:
                x = block(x, past_kv, use_cache)

        #x = F.layer_norm(x, (self.config.n_embd,))  #Дополнительная нормализация
        logits = self.lm_head(x)
        return (logits, new_key_values) if use_cache else logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and "bias" not in n]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or "bias" in n]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_p=None, repetition_penalty=None, stop_token=None, echo=None) -> torch.Tensor:
        self.eval()
        original_len = idx.size(1)
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                input_ids = idx
            else:
                input_ids = idx[:, -1:]

            # Проверка превышения максимальной длины
            current_length = input_ids.size(1) + (past_key_values[0][0].size(2) if past_key_values else 0)
            if current_length >= self.config.block_size:
                keep_len = self.config.block_size - 1
                input_ids = input_ids[:, -keep_len:]
                past_key_values = [(k[..., -keep_len:, :], v[..., -keep_len:, :]) for (k, v) in past_key_values]

            # Прямой проход
            logits, new_key_values = self(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = new_key_values


            # Сэмплинг следующего токена
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                unique_tokens, counts = torch.unique(idx, return_counts=True)
                for token, count in zip(unique_tokens, counts):
                    if count > 1:
                        logits[:, token] /= repetition_penalty ** (count - 1)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

            # Остановка по токену
            if stop_token is not None and idx_next.item() == stop_token:
                break

        # Сборка полной последовательности
        full_sequence = idx if echo else idx[:, original_len:]

        # Обрезка до исходной максимальной длины + новых токенов
        return full_sequence[:, :original_len + max_new_tokens]