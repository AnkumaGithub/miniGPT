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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Инициализация токенизатора
    enc = tiktoken.get_encoding(ENCODING)
    logging.info(f"Загружен токенизатор: {ENCODING} (vocab size: {enc.n_vocab})")

    # Загрузка датасета
    logging.info("Загрузка OpenWebText...")
    try:
        dataset = load_dataset("openwebtext", num_proc=NUM_PROC_LOAD)
    except Exception as e:
        logging.error(f"Ошибка загрузки датасета: {str(e)}")
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
            if not text:  # Проверка на пустой текст
                return {'ids': [], 'len': 0}

            ids = enc.encode_ordinary(text)
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
    if len(tokenized['train']) == 0 or len(tokenized['val']) == 0:
        logging.error("После фильтрации осталось 0 примеров!")
        return

    # Сохранение в бинарный формат
    for split in ['train', 'val']:
        output_path = os.path.join(OUTPUT_DIR, f'{split}.bin')
        logging.info(f"Обработка {split} раздела -> {output_path}")

        # Расчет общего размера
        lengths = [min(l, MAX_SEQ_LEN) for l in tokenized[split]['len']]
        total_len = min(sum(lengths), MAX_TOKENS)

        if total_len == 0:
            logging.warning(f"{split} раздел пуст!")
            continue

        # Создание memmap файла
        arr = np.memmap(
            output_path,
            dtype=np.uint16,
            mode='w+',
            shape=(total_len,))

        # Постепенная запись
        current_idx = 0
        for example in tqdm(tokenized[split], desc=f"Запись {split}"):
            if current_idx >= total_len:
                break

            ids = np.array(example['ids'][:MAX_SEQ_LEN], dtype=np.uint16)
            write_len = min(len(ids), total_len - current_idx)

            if write_len > 0:
                arr[current_idx:current_idx + write_len] = ids[:write_len]
                current_idx += write_len

        arr.flush()  # Форсированная запись на диск
        logging.info(f"Сохранено {current_idx} токенов в {output_path}")
    # Финализация
    logging.info("Обработка данных завершена!")

if __name__ == "__main__":
    prepare_data()