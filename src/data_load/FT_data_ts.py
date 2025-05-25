from datasets import load_dataset, DatasetDict
from transformers import GPT2TokenizerFast
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

BLOCK_SIZE = 256
SPECIAL_TOKENS = ["[EOS]", "[PAD]"]
OUTPUT_DIR = "/data/tinystories_HF"


def prepare_data():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("Загрузка TinyStories...")
    try:
        dataset = load_dataset("roneneldan/TinyStories")
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {str(e)}")
        return

    def process_text(examples):
        processed = {"input_ids": [], "attention_mask": [], "labels": []}

        for text in examples["text"]:
            cleaned_text = text.replace("\n", " ").strip()

            tokens = tokenizer(
                cleaned_text,
                truncation=True,
                max_length=BLOCK_SIZE - 1,
                add_special_tokens=False,
            )["input_ids"]

            tokens = tokens[:BLOCK_SIZE - 1]
            tokens = tokens + [tokenizer.eos_token_id]
            pad_len = BLOCK_SIZE - len(tokens)

            input_ids = tokens + [tokenizer.pad_token_id] * pad_len
            attention_mask = [1] * (len(tokens)) + [0] * pad_len

            processed["input_ids"].append(input_ids)
            processed["attention_mask"].append(attention_mask)
            processed["labels"].append(input_ids.copy())

        return processed

    logging.info("Обработка данных...")
    dataset = dataset.map(
        process_text,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        num_proc=4
    )

    def check_length(example):
        return len(example["input_ids"]) == BLOCK_SIZE

    dataset = dataset.filter(check_length, num_proc=4)

    logging.info("Сохранение данных...")
    dataset.save_to_disk(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logging.info(f"Данные сохранены в {OUTPUT_DIR}")
    logging.info(f"Пример токена: [EOS] = {tokenizer.eos_token_id}, [PAD] = {tokenizer.pad_token_id}")


if __name__ == "__main__":
    prepare_data()