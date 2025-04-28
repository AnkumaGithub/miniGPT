import torch
import tiktoken
from model import GPT, GPTConfig

config = GPTConfig(
    vocab_size=50262,
    block_size=512,
    n_layer=16,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    drop_path_rate=0.1,
    batch_size=12,
    lr=3e-4,
    bias=False,
    mode='ts',
    stride=368,
    weight_decay=0.1,
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

SPECIAL_TOKENS = ["[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]

enc = tiktoken.get_encoding(ENCODING)
enc = tiktoken.Encoding(
    name=enc.name,
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens={**enc._special_tokens, **{token: len(enc._mergeable_ranks) + i for i, token in enumerate(SPECIAL_TOKENS)}}
)

def generate_text(prompt, max_new_tokens=400, temperature=1.2, top_p=0.98, repetition_penalty=1.5, stop_token="[EOS]"):
    input_ids = torch.tensor([enc.encode(prompt, allowed_special=set(SPECIAL_TOKENS))], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            echo=True,
            stop_token=enc.encode(stop_token, allowed_special=set(SPECIAL_TOKENS))[0] if stop_token else None,
            bad_words_ids=[[enc.encode("[Q]", allowed_special=set(SPECIAL_TOKENS))[0]]]
        )[0].tolist()

    return enc.decode(output_ids)

def ask(question):
    prompt = f"[Q] {question} [A]"
    return generate_text(prompt, temperature=0.4, stop_token="[EOS]")

#print(ask("Tell me something about Earth"))

if __name__ == "__main__":
    while True:
        prompt = input("\nпромт\n> ")
        if prompt.lower() in ["exit", "quit"]:
            break

        output = generate_text(prompt)
        print("\nСгенерированный текст:\n")
        print(output)