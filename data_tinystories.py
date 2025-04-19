from datasets import load_dataset
import numpy as np
import os
import tiktoken
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Конфигурация
BLOCK_SIZE = 256
ENCODING = "gpt2"
OUTPUT_DIR = "data/tinystories"


def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Инициализация токенизатора
    enc = tiktoken.get_encoding(ENCODING)
    dtype = np.uint32 if enc.n_vocab >= 65536 else np.uint16
    logging.info(f"Using tokenizer: {ENCODING}, dtype: {dtype}")

    # Загрузка данных
    logging.info("Loading TinyStories...")
    try:
        dataset_train = load_dataset("roneneldan/TinyStories", split="train")
        dataset_val = load_dataset("roneneldan/TinyStories", split="validation")
    except Exception as e:
        logging.error(f"Dataset loading failed: {str(e)}")
        return

    # Функция обработки текста
    def process_text(example):
        text = example['text'].replace('\n', ' ').strip()
        ids = enc.encode_ordinary(text)
        chunks = []
        for i in range(0, len(ids), BLOCK_SIZE):
            chunk = ids[i:i + BLOCK_SIZE]
            if len(chunk) >= 64:  # Минимальная длина чанка
                chunks.append(chunk)
        return {'ids': chunks, 'len': len(ids)}

    # Обработка данных
    def process_split(dataset, split_name):
        tokenized = dataset.map(
            process_text,
            remove_columns=['text'],
            desc=f"Tokenizing {split_name}",
            num_proc=4
        )
        tokenized = tokenized.filter(lambda x: x['len'] > 0)

        # Сохранение в бинарный формат
        output_path = os.path.join(OUTPUT_DIR, f'{split_name}.bin')
        total_tokens = sum(len(chunk) for example in tokenized for chunk in example['ids'])

        arr = np.memmap(output_path, dtype=dtype, mode='w+', shape=(total_tokens,))
        idx = 0
        for example in tokenized:
            for chunk in example['ids']:
                arr[idx:idx + len(chunk)] = chunk
                idx += len(chunk)
        arr.flush()
        logging.info(f"Saved {idx} tokens to {output_path}")

    # Обработка train и validation
    process_split(dataset_train, "train_256")
    process_split(dataset_val, "val_256")


if __name__ == "__main__":
    prepare_data()