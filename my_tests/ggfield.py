import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.model.model import GPTConfig
import logging
import os


class GPTDataset(Dataset):
    def __init__(self, split, block_size, stride=256):
        self.data_path = f'E:/PyCharm 2024.3.5/projects/data/openwebtext/{split}.bin'
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found")

        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.stride = stride
        self.total_samples = (len(self.data) - block_size) // stride + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        start = idx * self.stride  # Учитываем шаг между примерами
        end = start + self.block_size + 1
        if end > len(self.data):
            raise IndexError(f"Index {idx} out of data range (start={start}, end={end})")

        chunk = torch.from_numpy(self.data[start:end].astype(np.int32)).long()
        return chunk[:-1], chunk[1:]
def train():
    config = GPTConfig(
        vocab_size=50257,
        block_size = 1024,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1,
        drop_path_rate=0.1,
        batch_size = 12,
        lr = 2e-4,
        bias=False
    )

    num_workers = min(2, os.cpu_count() // 4)
    print("DataLoader-train-start")
    train_loader = DataLoader(
        GPTDataset('train', config.block_size, stride=256),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True
    )
    print("DataLoader-train-end")
    print("DataLoader-val-start")
    val_loader = DataLoader(
        GPTDataset('val', config.block_size, stride=256),
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True
    )
    print("DataLoader-val-end")
    print(len(train_loader))
    print(len(val_loader))
    global_step = 0
    global_step1 = 0
    global_step2 = 0
    global_step3 = 0
    global_step4 = 0
    global_step5 = 0
    for x, y in tqdm(train_loader):
        print(x.shape, y.shape, " ", global_step)
        global_step += 1

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")