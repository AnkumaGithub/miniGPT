import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_data():
    """Основная функция обработки данных с жестко заданными параметрами"""

    # Конфигурация
    OUTPUT_DIR = "./data/openwebtext"  # Выходная директория
    NUM_PROC = 8  # Количество процессов обработки
    NUM_PROC_LOAD = 4  # Процессов для загрузки
    VAL_RATIO = 0.005  # Доля валидации
    SEED = 42  # Сид для воспроизводимости
    MAX_TOKENS = 1_000_000_000  # Максимум токенов (1B)
    MAX_SEQ_LEN = 1024  # Макс длина последовательности
    ENCODING = "gpt2"  # Название токенизатора

    # Инициализация токенизатора
    enc = tiktoken.get_encoding(ENCODING)
    logging.info(f"Загружен токенизатор: {ENCODING} (vocab size: {enc.n_vocab})")

    # Загрузка датасета
    logging.info("Загрузка OpenWebText...")
    dataset = load_dataset("openwebtext", num_proc=NUM_PROC_LOAD)

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
            text = example['text'].replace('\n', ' ')  # Чистка текста
            ids = enc.encode_ordinary(text)  # Без спецтокенов
            return {'ids': ids, 'len': len(ids)}
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

    # Сохранение в бинарный формат
    for split in ['train', 'val']:
        logging.info(f"Обработка {split} раздела...")
        output_path = os.path.join(OUTPUT_DIR, f'{split}.bin')
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Расчет общего размера
        total_len = sum(min(l, MAX_SEQ_LEN) for l in tokenized[split]['len'])
        total_len = min(total_len, MAX_TOKENS)

        # Создание memmap файла
        arr = np.memmap(
            output_path,
            dtype=np.uint16,
            mode='w+',
            shape=(total_len,))

        # Постепенная запись
        current_idx = 0
        for example in tqdm(tokenized[split], desc=f"Запись {split}.bin"):
            ids = example['ids'][:MAX_SEQ_LEN]
            available_space = total_len - current_idx
            if available_space <= 0:
                break
            ids = ids[:available_space]
            arr[current_idx:current_idx + len(ids)] = ids
            current_idx += len(ids)

    arr.flush()
    logging.info(f"Сохранено {current_idx} токенов в {output_path}")


# Финализация
logging.info("Обработка данных завершена!")

if __name__ == "__main__":
    prepare_data()