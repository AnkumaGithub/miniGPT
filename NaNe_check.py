import numpy as np
import os
from tqdm import tqdm


def check_bin_file(file_path, vocab_size=50257):
    """
    Проверяет .bin файл на:
    - Наличие NaN/Inf значений
    - Корректность токенов (0 <= token < vocab_size)
    """
    try:
        # Загрузка файла как memmap для экономии памяти
        data = np.memmap(file_path, dtype=np.uint16, mode='r')

        # Проверка на NaN/Inf (для uint16 это невозможно, но для других типов полезно)
        if data.dtype != np.uint16:
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
        else:
            has_nan = has_inf = False

        # Проверка выхода токенов за пределы словаря
        invalid_tokens = np.any(data >= vocab_size)

        # Результаты
        print(f"Проверка файла: {os.path.basename(file_path)}")
        print(f"  Размер: {len(data)} токенов")
        print(f"  NaN: {'Обнаружены!' if has_nan else 'Нет'}")
        print(f"  Inf: {'Обнаружены!' if has_inf else 'Нет'}")
        print(f"  Некорректные токены: {'Есть!' if invalid_tokens else 'Нет'}")
        print("-" * 50)

    except Exception as e:
        print(f"Ошибка проверки {file_path}: {str(e)}")


if __name__ == "__main__":
    # Список файлов для проверки
    bin_files = [
        "E:/PyCharm 2024.3.5/projects/data/openwebtext/train_stride_512.bin",
        "E:/PyCharm 2024.3.5/projects/data/openwebtext/val_stride_512.bin"
    ]

    # Проверка каждого файла
    for file in bin_files:
        check_bin_file(file)