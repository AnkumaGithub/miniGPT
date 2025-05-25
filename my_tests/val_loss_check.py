import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from src.model.model import GPT, GPTConfig
import logging
import os
from dotenv import load_dotenv

os.environ["TMPDIR"] = "E:/temp_pytorch"
os.environ["TEMP"] = "E:/temp_pytorch"

load_dotenv(dotenv_path='../.env') # Подгружаем секретные данные

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
        end = start + self.block_size
        # Проверка, что end не превышает длину данных
        if end > len(self.data):
            # Возвращаем последний доступный блок
            start = max(0, len(self.data) - self.block_size)
            end = len(self.data)
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64)).long()
        return chunk[:-1], chunk[1:]


def train():
    # Конфигурация для RTX 3060
    config = GPTConfig(
        vocab_size=50257,
        block_size = 512,
        n_layer=10,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        drop_path_rate=0.1,
        batch_size = 12,
        lr = 3e-4,
        bias=False,
        mode='webtext_new_test',
        stride = 512,
        weight_decay = 0.1,
    )

    try:
        # Инициализация модели с оптимизациями
        model = GPT(config).cuda()

        # Проверка наличия CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # PyTorch 2.0+
        #if torch.__version__ >= "2.0.0":
            #model = torch.compile(model, mode="default")
            #print("Модель компилируется с torch.compile()")
        #else:
           # print("Обновись до torch 2.0")

        torch.backends.cuda.matmul.allow_tf32 = True  # Для тензорных ядер
        torch.backends.cudnn.allow_tf32 = True

        # DataLoader с проверкой данных
        num_workers = min(4, os.cpu_count() // 4)
        print("DataLoader-val-start")
        val_loader = DataLoader(
            GPTDataset('val_stride_128_4h_5m', config.block_size, stride=config.stride),
            batch_size=config.batch_size,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True
        )
        print("DataLoader-val-end")
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {str(e)}")
        return

    # Оптимизатор и скейлер
    scaler = torch.amp.GradScaler(device='cuda')
    fused_available = hasattr(torch.optim, 'fused_adam')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.lr,
                                  weight_decay=0.05,
                                  fused=fused_available,
                                  betas=(0.9, 0.95))

    checkpoint_name = f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth"

    warmup_steps = 3000  # Абсолютное число шагов прогрева
    planned_total_epochs = 14
    start_epoch = 1
    global_step = 0
    planned_total_epochs = 1
    if os.path.exists(checkpoint_name):
        with torch.serialization.safe_globals([GPTConfig]):
            checkpoint = torch.load(checkpoint_name)
        remaining_epochs = planned_total_epochs - checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        logging.info(f"Загружен чекпоинт эпохи {checkpoint['epoch']} для режима {config.mode}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(val_loader),
        T_mult=1,
        eta_min=3e-5
    )

    model.eval()
    val_loss = 0
    total_tokens = 0
    with torch.inference_mode(), autocast(device_type='cuda', dtype=torch.float16):
        for X, Y in tqdm(val_loader, desc="Validation"):
            X, Y = X.cuda(non_blocking=True), Y.cuda(non_blocking=True)
            logits = model(X)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                Y.view(-1),
                reduction='sum',
            )
            val_loss += loss.item()
            total_tokens += Y.numel()
    val_loss /= total_tokens
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    current_lr = scheduler.get_last_lr()[0]
    logging.info(
        f"Val Loss: {val_loss:.3f} | "
        f"Val Perplexity: {perplexity:.2f} | "
        f"LR: {current_lr:.2e}"
    )

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")