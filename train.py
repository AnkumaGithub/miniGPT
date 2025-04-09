import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from model import GPT, GPTConfig


class GPTDataset(Dataset):
    def __init__(self, split, block_size):
        self.data = np.memmap(f'./data/processed/{split}.bin',
                              dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return chunk[:-1], chunk[1:]


def train():
    # Конфигурация для RTX 3060
    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1,
        drop_path_rate=0.1
    )

    # Инициализация модели с оптимизациями
    model = GPT(config).cuda()

    # PyTorch 2.0+
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model, mode="max-autotune")
        print("Модель компилируется с orch.compile()")
    else:
        print("Обновись до torch 2.0")

    torch.backends.cuda.matmul.allow_tf32 = True  # Для тензорных ядер
    torch.backends.cudnn.allow_tf32 = True

    train_loader = DataLoader(
        GPTDataset('train', config.block_size),
        batch_size=12,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        GPTDataset('val', config.block_size),
        batch_size=12,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # Оптимизатор и скейлер
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=3e-4,
                                  weight_decay=0.1,
                                  fused=True)  # Включен fused Adam

    for epoch in range(20):
        model.train()
        total_loss = 0

        for X, Y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            X, Y = X.to('cuda', non_blocking=True), Y.to('cuda', non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(X)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                    ignore_index=0
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            torch.cuda.empty_cache()

        # Валидация с кешированием
        model.eval()
        val_loss = 0
        with torch.no_grad(), autocast():
            for X, Y in val_loader:
                X, Y = X.to('cuda'), Y.to('cuda')
                logits = model(X)
                val_loss += torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                    ignore_index=0
                ).item()
        print(f"Epoch {epoch + 1} | "
              f"Train Loss: {total_loss / len(train_loader):.3f} | "
              f"Val Loss: {val_loss / len(val_loader):.3f}")

        # Асинхронное сохранение
        torch.save(model.state_dict(), f"gpt_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train()