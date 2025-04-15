import numpy as np
import os


def check_blocks(file_path, block_size=512):
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    num_blocks = total_tokens // block_size
    print(f"Всего токенов: {total_tokens}")
    print(f"Ожидаемое число блоков по 512: {num_blocks}")

    # Проверка, что все блоки имеют длину 1024
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = data[start:end]
        if len(block) != block_size:
            print(f"Блок {i} имеет длину {len(block)} токенов!")

    # Проверка остатка
    remainder = total_tokens % block_size
    if remainder != 0:
        print(f"Последний блок имеет длину {remainder} токенов.")


if __name__ == "__main__":
    check_blocks("E:/PyCharm 2024.3.5/projects/data/openwebtext/trainbackup.bin")