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
    VAL_RATIO = 0.05  # Доля валидации
    SEED = 42  # Сид для воспроизводимости
    MAX_TOKENS = 2_000_000_000  # Максимум токенов (1B)
    MAX_SEQ_LEN = 1024  # Макс длина последовательности
    ENCODING = "gpt2"  # Название токенизатора
    MIN_SEQ_LENGTH = 64
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
                if len(chunk) >= MIN_SEQ_LENGTH:  # Иначе короткие чанки попадут в данные
                    chunks.append(chunk)
            return {'ids': chunks, 'len': len(ids)}
        except Exception as e:
            sample = example['text'][:50] + '...' if example['text'] else 'Empty'
            logging.error(f"Ошибка в тексте: '{sample}' -> {str(e)}")
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
    # Сохранение с прямой записью в memmap
    for split in ['train', 'val']:
        output_path = os.path.join(OUTPUT_DIR, f'{split}.bin')
        logging.info(f"Обработка {split} -> {output_path}")

        # Создаем memmap файл сразу
        arr = np.memmap(output_path, dtype=dtype, mode='w+', shape=(MAX_TOKENS,)) # Может занять много места, 1B это ~ 2gb
        current_idx = 0

        pbar = tqdm(total=MAX_TOKENS, desc="Запись токенов")
        for example in tqdm(tokenized[split], desc=f"Обработка {split}"):
            if current_idx >= MAX_TOKENS:
                break
            for chunk in example['ids']:
                chunk_len = len(chunk)
                if current_idx + chunk_len > MAX_TOKENS:
                    chunk_len = MAX_TOKENS - current_idx
                    chunk = chunk[:chunk_len]

                arr[current_idx:current_idx + chunk_len] = chunk
                current_idx += chunk_len
                if current_idx >= MAX_TOKENS:
                    break
                pbar.update(chunk_len)

        # Обрезаем файл до реального размера
        pbar.close()
        arr = arr[:current_idx]
        arr.flush()
        del arr  # Закрытие memmap

        # Валидация
        test_arr = np.memmap(output_path, dtype=dtype, mode='r')
        assert len(test_arr) == current_idx, "Несоответствие размера файла!"
        del test_arr
        logging.info(f"Сохранено {current_idx} токенов")
        logging.info(f"Использовано RAM: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
    # Финализация
    logging.info("Обработка данных завершена!")

if __name__ == "__main__":
    prepare_data()