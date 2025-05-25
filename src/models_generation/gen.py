import torch
import tiktoken
from src.model.model import GPT, GPTConfig

import matplotlib.pyplot as plt


#plt.rcParams['font.family'] = 'Noto Sans CJK JP'

def plot_attention(attention_maps, tokens, layer=0, head=0, step=-1, max_tokens=30):
    if not attention_maps:
        print("Нет данных внимания.")
        return

    step = len(attention_maps) - 1 if step == -1 else step
    current_step = attention_maps[step]

    # Обрезаем токены и матрицу
    tokens = tokens[:max_tokens]
    attn = current_step[layer][0, head].numpy()[:max_tokens, :max_tokens]

    plt.figure(figsize=(12, 10))
    plt.imshow(attn, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
    plt.yticks(range(len(tokens)), tokens, fontsize=8)
    plt.title(f"Attention (Layer {layer}, Head {head}, Step {step})")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def analyze_attention(attention_maps, tokens, steps_to_check=[0, 10, -1], top_k=3, threshold=0.2):
    if not attention_maps:
        print("Нет данных внимания.")
        return

    for step in steps_to_check:
        # Корректируем индекс шага
        adj_step = len(attention_maps) + step if step < 0 else step
        current_step = attention_maps[adj_step]
        print(f"\n=== Шаг {step} (Токен: '{tokens[step]}') ===")

        for layer in range(len(current_step)):
            for head in range(current_step[layer].shape[1]):
                attn = current_step[layer][0, head].numpy()
                strong_connections = []

                for i in range(attn.shape[0]):
                    for j in range(attn.shape[1]):
                        if attn[i, j] > threshold and i != j:
                            strong_connections.append((i, j, attn[i, j]))

                strong_connections.sort(key=lambda x: x[2], reverse=True)

                print(f"\nСлой {layer}, Голова {head}:")
                for idx, (i, j, score) in enumerate(strong_connections[:top_k]):
                    src_token = tokens[i] if i < len(tokens) else "[PAD]"
                    tgt_token = tokens[j] if j < len(tokens) else "[PAD]"
                    print(f"  {src_token} → {tgt_token} ({score:.2f})")

config = GPTConfig(
    vocab_size=50263,
    block_size=300,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.05,
    drop_path_rate=0.05,
    batch_size=12,
    lr=3e-4,
    bias=False,
    mode='ts_2',
    stride=300,
    weight_decay=0.1,
    pad_token_id=None
)

#CHECKPOINT_PATH = f"latest_checkpoint.pth"  # путь до чекпоинта
CHECKPOINT_PATH = f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth"  # путь до чекпоинта
ENCODING = "gpt2"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


model = GPT(config).to(DEVICE)

with torch.serialization.safe_globals([GPTConfig]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model.load_state_dict(checkpoint["model"])
model.eval()

SPECIAL_TOKENS = ["[PAD]", "[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]

enc = tiktoken.get_encoding(ENCODING)
enc = tiktoken.Encoding(
    name=enc.name,
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens={**enc._special_tokens, **{token: len(enc._mergeable_ranks) + i for i, token in enumerate(SPECIAL_TOKENS)}}
)

def generate_text(prompt, max_new_tokens=200, temperature=0.3, top_p=0.85, repetition_penalty=8.0, stop_token="[EOS]", return_attention=True):
    input_ids = torch.tensor([enc.encode(prompt, allowed_special=set(SPECIAL_TOKENS))], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output, attention_maps = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            echo=True,
            stop_token=enc.encode(stop_token, allowed_special=set(SPECIAL_TOKENS))[0] if stop_token else None,
            bad_words_ids=[[enc.encode("[USER]", allowed_special=set(SPECIAL_TOKENS))[0]]]
        )

    tokens = [enc.decode([t]) for t in output[0].tolist()]

    if return_attention:
        return tokens, attention_maps
    else:
        return tokens

def main():
    while True:
        prompt = input("\nпромт\n> ")
        if prompt.lower() in ["exit", "quit"]:
            break
        return_attention = True
        if return_attention == True:
            output, attention_maps = generate_text(prompt)
        else:
            output = generate_text(prompt, return_attention)
        print("\nСгенерированный текст:\n")
        print("".join(output))

if __name__ == "__main__":
    main()
    # Автоматический анализ внимания
    #analyze_attention(
    #    attention_maps,
    #    output,
    #    steps_to_check=[0, 10, -1]
    #)
