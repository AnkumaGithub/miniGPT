import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from model import GPT, GPTConfig
import logging
import os
from comet_ml import Experiment
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env') # Подгружаем секретные данные

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class GPTDataset(Dataset):
    def __init__(self, split, block_size):
        self.data_path = f'./data/processed/{split}.bin'
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found")

        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.total_samples = len(self.data) - block_size

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        start = idx
        end = start + self.block_size + 1
        if end > len(self.data):
            raise IndexError("Index out of data range")

        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return chunk[:-1], chunk[1:]


def log_gradients(model, experiment, step):
    grad_data = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_data[f"grad/{name}_mean"] = grad.abs().mean().item()
            grad_data[f"grad/{name}_max"] = grad.abs().max().item()
    experiment.log_metrics(grad_data, step=step)


def train():
    # Инициализация Comet ML
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="minigpt",
        workspace="ankumagithub",
        auto_param_logging=False,
        disabled=False
    )
    # Конфигурация для RTX 3060
    config = GPTConfig(
        block_size=1024,
        vocab_size=50258,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1,
        drop_path_rate=0.1
    )

    # Логирование гиперпараметров
    experiment.log_parameters({
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "lr": 3e-4,
        "batch_size": 8
    })

    try:
        # Инициализация модели с оптимизациями
        model = GPT(config).cuda()

        # Проверка наличия CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # PyTorch 2.0+
        if torch.__version__ >= "2.0.0":
            model = torch.compile(model, mode="max-autotune")
            print("Модель компилируется с torch.compile()")
        else:
            print("Обновись до torch 2.0")

        torch.backends.cuda.matmul.allow_tf32 = True  # Для тензорных ядер
        torch.backends.cudnn.allow_tf32 = True

        # DataLoader с проверкой данных
        try:
            num_workers = min(8, os.cpu_count() // 2)
            train_loader = DataLoader(
                GPTDataset('train', config.block_size),
                batch_size=8,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            val_loader = DataLoader(
                GPTDataset('val', config.block_size),
                batch_size=8,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True
            )
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {str(e)}")
            return

        # Оптимизатор и скейлер
        scaler = GradScaler()
        fused_available = hasattr(torch.optim, 'fused_adam')
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=3e-4,
                                      weight_decay=0.1,
                                      fused=fused_available)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * 20, # Полный цикл на 20 эпох
            eta_min=3e-5
        )

        # Чекпоинтинг
        start_epoch = 0
        if os.path.exists("latest_checkpoint.pth"):
            checkpoint = torch.load("latest_checkpoint.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Загружен чекпоинт эпохи {checkpoint['epoch']}")

        global_step = 0

        for epoch in range(start_epoch, 20):
            try:
                model.train()
                total_loss = 0

                train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
                for X, Y in train_iter:
                    X, Y = X.to('cuda', non_blocking=True), Y.to('cuda', non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    with autocast():
                        logits = model(X)
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            Y.view(-1),
                        )

                    scaler.scale(loss).backward()

                    if global_step % 100 == 0:
                        log_gradients(model, experiment, global_step)
                        grad_norms = [torch.norm(p.grad.detach(), 2).item()
                                      for p in model.parameters()
                                      if p.grad is not None]

                        if grad_norms:  # Защита от пустого списка
                            avg_norm = np.mean(grad_norms)
                            max_norm = np.max(grad_norms)

                            experiment.log_metrics({
                                "grad/avg_norm": avg_norm,
                                "grad/max_norm": max_norm
                            }, step=global_step)
                            if max_norm > 1e5:
                                logging.error(f"Exploding gradients detected! Max norm: {max_norm:.2f}")
                        else:
                            logging.warning("No gradients found!")
                    train_iter.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

                    # Gradient Clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    total_loss += loss.item()
                    global_step += 1

                # Валидация
                model.eval()
                val_loss = 0
                with torch.inference_mode(), autocast():
                    for X, Y in tqdm(val_loader, desc="Validation"):
                        X, Y = X.cuda(non_blocking=True), Y.cuda(non_blocking=True)
                        logits = model(X)
                        val_loss += torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            Y.view(-1),
                        ).item()
                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                current_lr = scheduler.get_last_lr()[0]
                logging.info(
                    f"Epoch {epoch + 1} | "
                    f"Train Loss: {avg_train_loss:.3f} | "
                    f"Val Loss: {avg_val_loss:.3f} | "
                    f"LR: {current_lr:.2e}"
                )

                experiment.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                    "gpu_memory_peak": torch.cuda.max_memory_allocated() / 1e9,
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9
                }, step=global_step
                )

                if epoch % 5 == 0:
                    for name, param in model.named_parameters():
                        experiment.log_histogram_3d(
                            values=param.data.cpu().numpy().flatten(),
                            name=f"weights/{name}",
                            step=global_step
                        )

                # чекпоинт
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": config
                }
                torch.save(checkpoint, f"epoch_{epoch:02d}.pth")
                torch.save(checkpoint, "latest_checkpoint.pth")
                experiment.log_model(
                    f"epoch-{epoch:02d}",
                    f"epoch_{epoch:02d}.pth"
                )
            except Exception as e:
                logging.error(f"Ошибка в эпохе {epoch + 1}: {str(e)}")
                experiment.log_text(f"Exception: {str(e)}")
                experiment.log_metrics({"error": 1})
                break
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")
        experiment.log_text(f"Exception: {str(e)}")
        experiment.log_metrics({"error": 1})
        raise
    finally:
        experiment.end()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")