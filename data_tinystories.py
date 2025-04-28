from datasets import load_dataset
import numpy as np
import os
import tiktoken
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Конфигурация
BLOCK_SIZE = 300
ENCODING = "gpt2"
OUTPUT_DIR = "data/tinystories"
SPECIAL_TOKENS = ["[PAD]", "[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]  # Добавлен PAD

enc = tiktoken.get_encoding(ENCODING)
special_tokens_dict = {token: len(enc._mergeable_ranks) + idx for idx, token in enumerate(SPECIAL_TOKENS)}
enc = tiktoken.Encoding(
    name=enc.name,
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens={**enc._special_tokens, **special_tokens_dict}
)


def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Инициализация токенизатора с кастомными токенами
    eos_id = enc.encode_single_token("[EOS]")
    pad_id = enc.encode_single_token("[PAD]")
    logging.info(f"Токены: [EOS]={eos_id}, [PAD]={pad_id}")

    dtype = np.uint32 if enc.n_vocab >= 65536 else np.uint16
    logging.info(f"Using tokenizer: {ENCODING}, dtype: {dtype}")

    # Загрузка данных
    logging.info("Loading TinyStories...")
    try:
        dataset_train = load_dataset("roneneldan/TinyStories", split="train")
        dataset_val = load_dataset("roneneldan/TinyStories", split="validation")
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {str(e)}")
        return

    def process_text(example):
        text = example['text'].replace('\n', ' ').strip()
        ids = enc.encode_ordinary(text)

        # Обрезаем до 299 токенов
        truncated = ids[:299]
        pad_needed = 299 - len(truncated)

        # Формируем блок: текст + паддинг + EOS
        block = (
                truncated +
                [pad_id] * pad_needed +
                [eos_id]
        )

        # Контроль длины
        if len(block) != BLOCK_SIZE:
            logging.warning(f"Некорректная длина блока: {len(block)}")
            return {'ids': [], 'len': 0}

        return {'ids': [block], 'len': len(ids)}

    def process_split(dataset, split_name):
        tokenized = dataset.map(
            process_text,
            remove_columns=['text'],
            desc=f"Обработка {split_name}",
            num_proc=4
        ).filter(lambda x: x['len'] > 0)

        # Сохранение данных
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.bin")
        total_blocks = sum(len(ex['ids']) for ex in tokenized)
        arr = np.memmap(output_path, dtype=dtype, mode='w+', shape=(total_blocks * BLOCK_SIZE,))

        idx = 0
        for ex in tokenized:
            for chunk in ex['ids']:
                arr[idx:idx + BLOCK_SIZE] = chunk
                idx += BLOCK_SIZE
        arr.flush()
        logging.info(f"Сохранено {total_blocks} блоков в {output_path}")

    # Обработка данных
    process_split(dataset_train, "train")
    process_split(dataset_val, "val")


if __name__ == "__main__":
    #prepare_data()
    # Пример проверки
    arr = np.memmap("data/tinystories/train.bin", dtype=np.uint16, mode='r')
    print("Первый блок:", enc.decode(arr[BLOCK_SIZE*3000:BLOCK_SIZE*3001]))
    print("Последний токен:", arr[BLOCK_SIZE - 1])