import torch
import tiktoken
from model import GPT, GPTConfig

config = GPTConfig(
    vocab_size=50257,
    block_size=511,
    n_layer=6,
    n_head=8,
    n_embd=368,
    dropout=0.15,
    drop_path_rate=0.15,
    batch_size=16,
    lr=3e-4,
    bias=False,
    mode='train_512_1',
    stride=256
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

enc = tiktoken.get_encoding(ENCODING)

def generate_text(prompt, max_new_tokens=100, temperature=0.6, top_k=20):
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            echo=True
        )[0].tolist()

    return enc.decode(output_ids)

if __name__ == "__main__":
    while True:
        prompt = input("\nпромт\n> ")
        if prompt.lower() in ["exit", "quit"]:
            break

        output = generate_text(prompt, max_new_tokens=40)
        print("\nСгенерированный текст:\n")
        print(output)
