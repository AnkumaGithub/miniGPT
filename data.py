import os
import json
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer


def prepare_data():
    # Конфигурация
    data_dir = "./data/raw"
    output_dir = "./data/processed"
    block_size = 1024

    # Инициализация токенизатора
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Сбор всех текстов
    all_tokens = []

    for filename in tqdm(os.listdir(data_dir), desc="Processing files"):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line)["text"]
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    all_tokens.extend(tokens)
                except:
                    continue

    # Разделение на train/val и сохранение
    split_idx = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    os.makedirs(output_dir, exist_ok=True)

    # Сохранение в бинарном формате
    for name, data in [("train", train_tokens), ("val", val_tokens)]:
        arr = np.memmap(
            os.path.join(output_dir, f"{name}.bin"),
            dtype=np.uint16,
            mode="w+",
            shape=(len(data)))
        arr[:] = np.array(data)
        arr.flush()

        if __name__ == "__main__":
            prepare_data()