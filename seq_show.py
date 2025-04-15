
import numpy as np
import tiktoken
import torch
from tiktoken.load import load_tiktoken_bpe
import os
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, split, block_size, stride=256):
        self.data_path = f'E:/PyCharm 2024.3.5/projects/data/openwebtext/{split}.bin'
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found")

        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.stride = stride
        self.total_samples = (len(self.data) - 1 - block_size) // stride + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.block_size + 1
        # Проверка, что end не превышает длину данных
        if end > len(self.data):
            # Возвращаем последний доступный блок
            start = max(0, len(self.data) - self.block_size - 1)
            end = len(self.data)
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64)).long()
        return chunk[:-1], chunk[1:]

def decode_train_bin(
        bin_path: str = "E:/PyCharm 2024.3.5/projects/data/openwebtext/trainbackup.bin",
        num_samples: int = 5,  # Сколько примеров показать
        max_seq_len: int = 1024
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
        end = start + max_seq_len - 1

        # Берем блок токенов
        chunk = data[start:end].tolist()
        chunkend = data[end-10:end + 1].tolist()

        # Декодируем с обработкой специальных токенов
        text = enc.decode(chunk)
        textend = enc.decode(chunkend)

        # Выводим результат
        print(f"--- Пример {i + 1} (токены {start}-{end}) ---")
        print(text.replace('\n', ' ') + "...")


if __name__ == "__main__":
    #decode_train_bin()
    enc = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(split='trainbackup', block_size=511, stride=512)
    for i in range(3):
        x, y = dataset[i]
        print(f"Блок {i}:")
        print("Вход:", enc.decode(x.numpy()[-50:]))  # Последние 50 токенов входа
        print("Цель:", enc.decode(y.numpy()[:50]))  # Первые 50 токенов цели