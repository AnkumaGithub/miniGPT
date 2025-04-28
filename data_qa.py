import logging
from datasets import load_dataset
import numpy as np
import os
import tiktoken

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

ENCODING = "gpt2"
OUTPUT_DIR = "data/qa"
SPECIAL_TOKENS = ["[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]
enc = tiktoken.get_encoding(ENCODING)
enc = tiktoken.Encoding(
    name=enc.name,
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens={**enc._special_tokens, **{t: len(enc._mergeable_ranks) + i for i, t in enumerate(SPECIAL_TOKENS)}}
)
for token in SPECIAL_TOKENS:
    token_id = enc.encode_single_token(token)
    logging.info(f"Токен '{token}' имеет ID: {token_id}")

def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for token in SPECIAL_TOKENS:
        token_id = enc.encode_single_token(token)
        logging.info(f"Токен '{token}' имеет ID: {token_id}")
    tokens = []

    dataset = load_dataset("rubenroy/GammaCorpus-Fact-QA-450k")["train"]

    for example in dataset:
        text = f"[Q] {example['question']} [A] {example['answer']} [EOS]"
        tokens.extend(enc.encode(text, allowed_special=set(SPECIAL_TOKENS)))

    # Сохранение
    arr = np.memmap(os.path.join(OUTPUT_DIR, "train.bin"), dtype=np.uint16, mode="w+", shape=(len(tokens),))
    arr[:] = tokens
    print(f"Сохранено {len(tokens)} токенов")


if __name__ == "__main__":
    prepare_data()
    arr = np.memmap("data/qa/train.bin", dtype=np.uint16, mode='r')
    print(arr[:900])
    print(enc.decode(arr[:900]))