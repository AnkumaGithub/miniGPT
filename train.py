import math

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
import torch.nn.functional as F
from dotenv import load_dotenv

os.environ["TMPDIR"] = "E:/temp_pytorch"
os.environ["TEMP"] = "E:/temp_pytorch"

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
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
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
            grad_mean = grad.abs().mean().item()
            grad_max = grad.abs().max().item()
            grad_data[f"grad/{name}_mean"] = grad_mean
            grad_data[f"grad/{name}_max"] = grad_max
            # Вывод в консоль для проверки
            if "lm_head" in name:
                logging.info(f"[DEBUG] {name} | max_grad={grad_max:.4f}")
    experiment.log_metrics(grad_data, step=step)

def estimate_loss(model, val_loader, ctx):
    """Возвращает loss и perplexity"""
    model.eval()
    losses = []
    for X, Y in val_loader:
        X, Y = X.cuda(non_blocking=True), Y.cuda(non_blocking=True)
        with ctx:
            logits = model(X)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                Y.view(-1))
        losses.append(loss.item())
    model.train()
    avg_loss = torch.tensor(losses).mean().item()
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def get_lr(it, learning_rate, warmup_iters, min_lr, lr_decay_iters):
    # 1) Линейный прогрев
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # 2) Косинусное затухание
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train():
    # Конфигурация для RTX 3060
    config = GPTConfig(
        vocab_size=50257,
        block_size = 511,
        n_layer=6,
        n_head=8,
        n_embd=368,
        dropout=0.1,
        drop_path_rate=0.1,
        batch_size = 16,
        lr = 3e-4,
        bias=False,
        mode='train_512_2',
        stride = 512
    )

    # Инициализация Comet ML
    experiment = None
    new_exp = 0
    checkpoint_name = f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth"

    # Создаём новый эксперимент
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="minigpt",
        workspace="ankumagithub",
        auto_param_logging=False
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
        "batch_size": config.batch_size,
        "weight_decay": 0.02,
        "stride": config.stride,
    })

    try:
        # Проверка наличия CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        model = GPT(config).cuda()
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)

        # PyTorch 2.0+
        #if torch.__version__ >= "2.0.0":
            #model = torch.compile(model, mode="default")
            #print("Модель компилируется с torch.compile()")
        #else:
           # print("Обновись до torch 2.0")

        torch.backends.cuda.matmul.allow_tf32 = True  # Для тензорных ядер
        torch.backends.cudnn.allow_tf32 = True


        # DataLoader с проверкой данных
        try:
            num_workers = min(4, os.cpu_count() // 4)
            print("DataLoader-train-start")
            train_loader = DataLoader(
                GPTDataset('train_stride_128', config.block_size, stride=config.stride),
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True
            )
            print("DataLoader-train-end")
            print("DataLoader-val-start")
            val_loader = DataLoader(
                GPTDataset('train_stride_32', config.block_size, stride=config.stride),
                batch_size=config.batch_size,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True
            )
            print("DataLoader-val-end")
            test_loader = DataLoader(
                GPTDataset('test', config.block_size, stride=config.stride),
                batch_size=config.batch_size,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True
            )
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {str(e)}")
            return

        warmup_iters = 500  # Совпадает с вашим текущим прогревом
        min_lr = 3e-5  # Из вашего шедулера
        lr_decay_iters = 5 * len(train_loader)  # Общее число итераций
        gradient_accumulation_steps = 4  # Пункт 1: Градиентная аккумуляция

        # Оптимизатор и скейлер
        scaler = torch.amp.GradScaler(device='cuda')
        fused_available = hasattr(torch.optim, 'fused_adam')
        best_val_loss = float('inf')
        optimizer = model.configure_optimizers(
            weight_decay=0.05,
            learning_rate=config.lr,
            betas=(0.9, 0.95),
            device_type='cuda'
        )

        warmup_steps = 500  # Абсолютное число шагов прогрева
        total_train_steps = 14 * len(train_loader)  # 3 текущие + 11 будущих эпох
        planned_total_epochs = 14
        if os.path.exists(checkpoint_name):
            with torch.serialization.safe_globals([GPTConfig]):
                checkpoint = torch.load(checkpoint_name)
            remaining_epochs = planned_total_epochs - checkpoint['epoch']
            total_train_steps = remaining_epochs * len(train_loader)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            logging.info(f"Загружен чекпоинт эпохи {checkpoint['epoch']} для режима {config.mode}")


        # Чекпоинтинг
        start_epoch = 1

        global_step = 0

        print("epochs-start")
        experiment.log_other("train_samples", len(train_loader))
        experiment.log_other("val_samples", len(val_loader))
        experiment.log_other("stride", config.stride)
        current_epochs = 2
        perplexity = None
        for epoch in range(start_epoch, start_epoch + current_epochs):
            torch.cuda.reset_peak_memory_stats()
            iter_step = 0
            try:
                model.train()
                optimizer.zero_grad()

                train_iter = tqdm(train_loader, desc=f"Epoch {epoch}")
                for micro_step, (X, Y) in enumerate(train_iter):
                    # Пункт 1: Градиентная аккумуляция
                    X, Y = X.to(device='cuda', non_blocking=True), Y.to(device='cuda', non_blocking=True)

                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(X)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            Y.view(-1))
                        loss = loss / gradient_accumulation_steps  # Масштабируем loss
                        train_perplexity = math.exp(loss.item())
                        if math.isnan(train_perplexity) or math.isinf(train_perplexity):
                            logging.error("Invalid perplexity value! Stopping training.")
                            break

                    scaler.scale(loss).backward()

                    if (micro_step + 1) % gradient_accumulation_steps == 0:
                        # Пункт 3: Обновление LR
                        lr = get_lr(global_step, config.lr, warmup_iters, min_lr, lr_decay_iters)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        global_step += 1

                        if global_step % 50 == 0:
                            logging.info(
                                f"Step {global_step} | "
                                f"Train Loss: {loss.item():.3f} | "
                                f"Train Perplexity: {train_perplexity:.2f}"
                            )

                        # Логирование и валидация
                        if global_step % 100 == 0:
                            val_loss, val_perplexity = estimate_loss(model, val_loader, ctx)
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                checkpoint = {
                                    "model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "config": config
                                }
                                torch.save(checkpoint, f"E:\PyCharm 2024.3.5\projects\saves\_best_val_checkpoint_{config.mode}.pth")
                            log_gradients(model, experiment, global_step)
                            log_memory_usage(experiment, global_step)
                            logging.info(
                                f"Step {global_step} | "
                                f"Val Loss: {val_loss:.3f} | "
                                f"Val Perplexity: {val_perplexity:.2f}"
                            )

                            experiment.log_metrics({
                                "val_loss": val_loss,
                                "val_perplexity": val_perplexity,
                            }, step=global_step)

                for name, param in model.named_parameters():
                    experiment.log_histogram_3d(
                        values=param.data.cpu().numpy().flatten(),
                        name=f"weights/{name}",
                        step=epoch
                    )

                    experiment.log_metrics({
                        f"weights/{name}_mean": param.data.mean().item(),
                        f"weights/{name}_std": param.data.std().item(),
                    }, step=epoch)

                # чекпоинт
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": config,
                    "global_step": global_step,
                    "experiment_id": experiment.get_key(),
                    "planned_total_epochs" : 14
                }
                torch.save(checkpoint, f"E:\PyCharm 2024.3.5\projects\saves\_{config.mode}_epoch_{epoch:02d}.pth")
                torch.save(checkpoint, f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth")
                #experiment.log_model( # Лучше не сохранять в облако, много весят
                    #f"E:\PyCharm 2024.3.5\projects\saves\_{config.mode}_epoch_{epoch + 1:02d}.pth",
                    #f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth"
                #)
            except Exception as e:
                logging.error(f"Ошибка в эпохе {epoch}: {str(e)}")
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