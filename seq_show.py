# decode_train_bin.py
import numpy as np
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import os


def decode_train_bin(
        bin_path: str = "E:/PyCharm 2024.3.5/projects/data/openwebtext/train_stride_256.bin",
        num_samples: int = 20,  # Сколько примеров показать
        max_seq_len: int = 512
):
    # Инициализация токенизатора как в data.py
    enc = tiktoken.get_encoding("gpt2")

    # Загрузка бинарного файла
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')

    print(f"Всего токенов: {len(data):,}")
    print(f"Пример декодирования первых {max_seq_len} токенов:\n")

    # Декодирование блоков
    for i in range(num_samples):
        start = i * max_seq_len
        end = start + max_seq_len

        # Берем блок токенов
        chunk = data[start:end].tolist()

        # Декодируем с обработкой специальных токенов
        text = enc.decode(chunk)

        # Выводим результат
        print(f"--- Пример {i + 1} (токены {start}-{end}) ---")
        print(text.replace('\n', ' ') + "...")


if __name__ == "__main__":
    decode_train_bin()