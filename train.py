from comet_ml import Experiment
import comet_ml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from model import GPT, GPTConfig
import logging
import os
import psutil
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env') # Подгружаем секретные данные

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
        end = start + self.block_size + 1
        # Проверка, что end не превышает длину данных
        if end > len(self.data):
            # Возвращаем последний доступный блок
            start = max(0, len(self.data) - self.block_size - 1)
            end = len(self.data)
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64)).long()
        return chunk[:-1], chunk[1:]


def log_memory_usage(experiment, step):
    # RAM
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1e9  # В гигабайтах

    # GPU
    gpu_usage = {}
    if torch.cuda.is_available():
        gpu_usage = {
            "gpu_mem_allocated": torch.cuda.memory_allocated() / 1e9,
            "gpu_mem_reserved": torch.cuda.memory_reserved() / 1e9,
            "gpu_mem_max_allocated": torch.cuda.max_memory_allocated() / 1e9
        }

    # Логирование в Comet
    metrics = {"ram_usage": ram_usage}
    if torch.cuda.is_available():
        metrics.update(gpu_usage)
    experiment.log_metrics(metrics, step=step)

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
        vocab_size=50257,
        block_size = 1024,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1,
        drop_path_rate=0.1,
        batch_size = 12,
        lr = 2e-4,
        bias=False,
        mode='test'
    )

    # Логирование гиперпараметров
    experiment.log_parameters({
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "drop_path_rate": config.drop_path_rate,
        "lr": config.lr,
        "batch_size": config.batch_size
    })

    try:
        # Инициализация модели с оптимизациями
        model = GPT(config).cuda()

        # Проверка наличия CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # PyTorch 2.0+
        #if torch.__version__ >= "2.0.0":
            #model = torch.compile(model, mode="default") # Triton не работает у меня(
            #print("Модель компилируется с torch.compile()")
        #else:
            #print("Обновись до torch 2.0")

        torch.backends.cuda.matmul.allow_tf32 = True  # Для тензорных ядер
        torch.backends.cudnn.allow_tf32 = True

        # DataLoader с проверкой данных
        try:
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
            test_loader = DataLoader(
                GPTDataset('test', config.block_size, stride=256),
                batch_size=config.batch_size,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True
            )
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {str(e)}")
            return

        # Оптимизатор и скейлер
        scaler = torch.amp.GradScaler(device='cuda')
        fused_available = hasattr(torch.optim, 'fused_adam')
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.lr,
                                      weight_decay=0.03,
                                      fused=fused_available)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_loader),
            T_mult=1,
            eta_min=3e-5
        )

        # Чекпоинтинг
        start_epoch = 0
        checkpoint_name = f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth"
        if os.path.exists(checkpoint_name):
            with torch.serialization.safe_globals([GPTConfig]):
                checkpoint = torch.load(checkpoint_name)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Загружен чекпоинт эпохи {checkpoint['epoch']} для режима {config.mode}")

        global_step = 0
        print("epochs-start")
        for epoch in range(start_epoch, 4):
            torch.cuda.reset_peak_memory_stats()
            iter_step = 0
            try:
                model.train()
                total_loss = 0

                train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
                for X, Y in train_iter:
                    X, Y = X.to('cuda', non_blocking=True, dtype=torch.long), Y.to('cuda', non_blocking=True, dtype=torch.long)

                    optimizer.zero_grad(set_to_none=True)

                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(X)
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            Y.view(-1),
                        )
                    experiment.log_metric("batch_loss", loss.item(), step=global_step)

                    scaler.scale(loss).backward()

                    if iter_step % 100 == 0:
                        log_gradients(model, experiment, global_step)
                        log_memory_usage(experiment, global_step)

                    train_iter.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
                    iter_step += 1

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
                with torch.inference_mode(), autocast(device_type='cuda', dtype=torch.float16):
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

                for name, param in model.named_parameters():
                    experiment.log_histogram_3d(
                        values=param.data.cpu().numpy().flatten(),
                        name=f"weights/{name}",
                        step=epoch + 1
                    )

                # чекпоинт
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "config": config
                }
                torch.save(checkpoint, f"E:\PyCharm 2024.3.5\projects\saves\_{config.mode}_epoch_{epoch + 1:02d}.pth")
                torch.save(checkpoint, f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth")
                #experiment.log_model( # Лучше не сохранять в облако, много весят
                    #f"E:\PyCharm 2024.3.5\projects\saves\_{config.mode}_epoch_{epoch + 1:02d}.pth",
                    #f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth"
                #)
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