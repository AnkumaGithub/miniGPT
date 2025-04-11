import os
import numpy as np
import psutil
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_data():
    # Конфигурация
    OUTPUT_DIR = "./data/openwebtext"  # Выходная директория
    NUM_PROC = 8  # Количество процессов обработки
    NUM_PROC_LOAD = 4  # Процессов для загрузки
    VAL_RATIO = 0.005  # Доля валидации
    SEED = 42  # Сид для воспроизводимости
    MAX_TOKENS = 1_000_000_000  # Максимум токенов (1B)
    MAX_SEQ_LEN = 1024  # Макс длина последовательности
    ENCODING = "gpt2"  # Название токенизатора
    MIN_TEXT_LENGTH = 64

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Инициализация токенизатора
    enc = tiktoken.get_encoding(ENCODING)
    if enc.n_vocab >= 65536:
        dtype = np.uint32
    else:
        dtype = np.uint16
    logging.info(f"Токенизатор: {ENCODING}, dtype: {dtype}")

    # Загрузка датасета
    logging.info("Загрузка OpenWebText...")
    try:
        dataset = load_dataset("Skylion007/openwebtext", num_proc=NUM_PROC_LOAD)
    except Exception as e:
        logging.error(f"Ошибка загрузки датасета: {str(e)}")
        return

    if 'train' not in dataset:
        logging.error("Датасет не содержит раздела 'train'")
        return

    # Создание train/val разделов
    split_dataset = dataset["train"].train_test_split(
        test_size=VAL_RATIO,
        seed=SEED,
        shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    # Функция обработки текста
    def process_text(example):
        try:
            text = example['text'].replace('\n', ' ').strip()
            if len(text) < MIN_TEXT_LENGTH:  # Добавить проверку минимальной длины
                return {'ids': [], 'len': 0}

            ids = enc.encode_ordinary(text)
            chunks = []
            for i in range(0, len(ids), MAX_SEQ_LEN):
                chunk = ids[i:i + MAX_SEQ_LEN]
                if len(chunk) >= MIN_TEXT_LENGTH:  # Иначе короткие чанки попадут в данные
                    chunks.append(chunk)
            return {'ids': chunks, 'len': len(ids)}
        except Exception as e:
            logging.error(f"Ошибка обработки: {str(e)}")
            return {'ids': [], 'len': 0}

    # Токенизация
    logging.info("Начало токенизации...")
    tokenized = split_dataset.map(
        process_text,
        remove_columns=['text'],
        desc="Токенизация данных",
        num_proc=NUM_PROC,
    )

    # Фильтрация пустых примеров
    tokenized = tokenized.filter(lambda x: x['len'] > 0)
    if len(tokenized['train']) == 0 or len(tokenized['val']) == 0:
        logging.error("После фильтрации осталось 0 примеров!")
        return

    # Сохранение в бинарный формат
    for split in ['train', 'val']:
        output_path = os.path.join(OUTPUT_DIR, f'{split}.bin')
        logging.info(f"Обработка {split} раздела -> {output_path}")

        all_ids = []
        current_tokens = 0

        for example in tqdm(tokenized[split], desc=f"Обработка {split}"):
            if current_tokens >= MAX_TOKENS:
                break
            for chunk in example['ids']:
                if current_tokens + len(chunk) > MAX_TOKENS:
                    chunk = chunk[:MAX_TOKENS - current_tokens]
                all_ids.extend(chunk)
                current_tokens += len(chunk)
                if current_tokens >= MAX_TOKENS:
                    break

        required_space = len(all_ids) * dtype().itemsize
        free_space = psutil.disk_usage(OUTPUT_DIR).free
        if required_space > free_space:
            logging.error("Недостаточно места на диске")
            return
        # Сохранение в memmap
        arr = np.memmap(output_path, dtype=dtype, mode='w+', shape=(len(all_ids),))
        arr[:] = np.array(all_ids, dtype=dtype)
        arr.flush()
        logging.info(f"Сохранено {len(all_ids)} токенов")
    # Финализация
    logging.info("Обработка данных завершена!")

if __name__ == "__main__":
    prepare_data()