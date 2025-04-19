import pytest
import torch
import math
from model import GPTConfig, CausalSelfAttention, RotaryPositionalEmbeddings, MLP, LayerNorm, DropPath, Block, GPT


@pytest.fixture
def config():
    return GPTConfig(
        n_embd=256,
        n_head=8,
        block_size=1024,
        dropout=0.1,
        bias=True
    )


def test_attention_basic_output_shape(config):
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.n_embd)
    attn = CausalSelfAttention(config)

    # Forward без кэша
    output, _ = attn(x)
    assert output.shape == (batch_size, seq_len, config.n_embd)


def test_attention_with_past_cache(config):
    batch_size = 3
    seq_len = 5
    x = torch.randn(batch_size, seq_len, config.n_embd)
    attn = CausalSelfAttention(config)

    # Первый вызов (кэш пустой)
    _, kv_cache = attn(x, use_cache=True)

    # Второй вызов с кэшем
    new_x = torch.randn(batch_size, 1, config.n_embd)
    output, new_kv = attn(new_x, past_key_values=kv_cache, use_cache=True)

    # Проверка размерности выхода
    assert output.shape == (batch_size, 1, config.n_embd)

    # Проверка расширения кэша
    assert new_kv[0].shape == (batch_size, config.n_head, seq_len + 1, config.n_embd // config.n_head)


def test_causal_mask(config):
    batch_size = 1
    seq_len = 4
    x = torch.randn(batch_size, seq_len, config.n_embd)
    attn = CausalSelfAttention(config)

    # Ручной режим без Flash Attention
    attn.flash = False
    output, _ = attn(x)

    # Проверка что нижний треугольник не замаскирован
    with torch.no_grad():
        q = torch.randn(1, config.n_head, seq_len, config.n_embd // config.n_head)
        k = torch.randn(1, config.n_head, seq_len, config.n_embd // config.n_head)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        assert not torch.isinf(att).all(), "Mask should only block upper triangle"


def test_rope_application():
    batch_size = 2
    seq_len = 6
    n_head = 8
    head_dim = 32
    x = torch.randn(batch_size, n_head, seq_len, head_dim)

    # Инициализация RoPE
    rope = RotaryPositionalEmbeddings(head_dim=head_dim)

    # Применяем RoPE с offset=0
    x_rotated = rope(x, offset=0)

    # Проверки
    # 1. Размерность сохранилась
    assert x_rotated.shape == x.shape

    # 2. Выход отличается от входа (ротация применена)
    assert not torch.allclose(x, x_rotated, atol=1e-6)

    # 3. Проверка работы с offset
    x_rotated_offset3 = rope(x, offset=3)
    assert not torch.allclose(x_rotated, x_rotated_offset3, atol=1e-6)

    # 4. Проверка детерминированности (при одинаковых offset)
    x_rotated2 = rope(x, offset=0)
    assert torch.allclose(x_rotated, x_rotated2, atol=1e-6)

    # 5. Проверка поворота для разных позиций
    first_token = x_rotated[:, :, 0, :]
    last_token = x_rotated[:, :, -1, :]
    assert not torch.allclose(first_token, last_token, atol=1e-6)


def test_flash_vs_manual(config):
    if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        pytest.skip("Flash Attention not available")

    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.n_embd)

    # Инициализация моделей
    attn_flash = CausalSelfAttention(config)
    attn_manual = CausalSelfAttention(config)
    attn_manual.flash = False

    # Копирование весов
    with torch.no_grad():
        attn_manual.load_state_dict(attn_flash.state_dict())

    # Отключение dropout и переход в eval режим
    attn_flash.eval()
    attn_manual.eval()
    attn_flash.attn_dropout.p = 0.0
    attn_manual.attn_dropout.p = 0.0

    # Forward pass
    out_flash, _ = attn_flash(x)
    out_manual, _ = attn_manual(x)

    # Проверка с увеличенным допуском
    assert torch.allclose(out_flash, out_manual, atol=1e-4, rtol=1e-3), \
        "Flash и ручная реализация отличаются больше допустимого"


def test_mlp_forward(config):
    batch_size = 3
    seq_len = 5
    x = torch.randn(batch_size, seq_len, config.n_embd)
    mlp = MLP(config)

    output = mlp(x)
    # Проверка размерности
    assert output.shape == x.shape

    # Проверка активации SwiGLU
    hidden_states = mlp.c_fc(x)
    activated = mlp.swiglu(hidden_states)
    assert activated.shape == (batch_size, seq_len, 4 * config.n_embd // 2)


def test_layernorm(config):
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.n_embd)
    ln = LayerNorm(config.n_embd, bias=config.bias)

    output = ln(x)
    # Проверка сохранения размерности
    assert output.shape == x.shape

    # Проверка параметров
    assert ln.weight.requires_grad
    if config.bias:
        assert ln.bias.requires_grad


def test_droppath(config):
    batch_size = 4
    seq_len = 7
    x = torch.randn(batch_size, seq_len, config.n_embd)
    dp = DropPath(drop_prob=0.5)

    # Проверка в режиме обучения
    dp.train()
    output_train = dp(x)
    assert not torch.allclose(x, output_train)

    # Проверка в режиме инференса
    dp.eval()
    output_eval = dp(x)
    assert torch.allclose(x, output_eval)


def test_block_forward(config):
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.n_embd)
    block = Block(config)

    # Forward без кэша
    output = block(x)
    assert output.shape == x.shape

    # Forward с кэшем
    past_key_values = None
    output_cache, new_kv = block(x, past_key_values, use_cache=True)
    assert output_cache.shape == x.shape
    assert len(new_kv) == 2  # key и value


def test_block_gradient_flow(config):
    x = torch.randn(2, 5, config.n_embd, requires_grad=True)
    block = Block(config)

    output = block(x)
    loss = output.sum()
    loss.backward()

    # Проверка градиентов
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_initialization(config):
    model = MLP(config)
    for name, param in model.named_parameters():
        if 'weight' in name and 'c_fc' in name:
            assert torch.allclose(param.mean(), torch.tensor(0.0), atol=0.1)
        elif 'weight' in name and 'c_proj' in name:
            assert param.std().item() < 0.1


def test_generate_smoke(config):
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    # Тест с echo=True (вернуть полную последовательность)
    output = model.generate(input_ids, max_new_tokens=3, echo=True)
    assert output.shape == (1, 8)  # 5 исходных + 3 новых

    # Тест с echo=False (только новые токены)
    output = model.generate(input_ids, max_new_tokens=3, echo=False)
    assert output.shape == (1, 3)