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
import math

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
        end = start + self.block_size
        # Проверка, что end не превышает длину данных
        if end > len(self.data):
            # Возвращаем последний доступный блок
            start = max(0, len(self.data) - self.block_size)
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
            grad_mean = grad.abs().mean().item()
            grad_max = grad.abs().max().item()
            grad_data[f"grad/{name}_mean"] = grad_mean
            grad_data[f"grad/{name}_max"] = grad_max
            # Вывод в консоль для проверки
            if "lm_head" in name:
                logging.info(f"[DEBUG] {name} | max_grad={grad_max:.4f}")
    experiment.log_metrics(grad_data, step=step)

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
        block_size = 256,
        n_layer=6,
        n_head=8,
        n_embd=512,
        dropout=0.05,
        drop_path_rate=0.05,
        batch_size = 40,
        lr = 1e-4,
        bias=False,
        mode='webtext_new',
        stride = 256,
        weight_decay = 0.05
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
        "weight_decay": config.weight_decay,
        "stride": config.stride,
    })

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
        try:
            num_workers = min(4, os.cpu_count() // 4)
            print("DataLoader-train-start")
            train_loader = DataLoader(
                GPTDataset('train_stride_256_4h_200m_1', config.block_size, stride=config.stride),
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=True
            )
            print("DataLoader-train-end")
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
        warmup_iters = 500
        min_lr = 3e-5
        current_epochs = 4
        new_it = 0
        lr_decay_iters = current_epochs * len(train_loader)  # Общее число итераций

        # Оптимизатор и скейлер
        scaler = torch.amp.GradScaler(device='cuda')
        fused_available = hasattr(torch.optim, 'fused_adam')
        optimizer = model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.lr,
            betas=(0.9, 0.95),
            device_type='cuda'
        )

        # Чекпоинтинг
        start_epoch = 1
        global_step = 0
        planned_total_epochs = 1
        if os.path.exists(checkpoint_name):
            with torch.serialization.safe_globals([GPTConfig]):
                checkpoint = torch.load(checkpoint_name)
            remaining_epochs = planned_total_epochs - checkpoint['epoch']
            total_train_steps = remaining_epochs * len(train_loader)
            model.load_state_dict(checkpoint['model'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            logging.info(f"Загружен чекпоинт эпохи {checkpoint['epoch']} для режима {config.mode}")

        print("epochs-start")
        experiment.log_other("train_samples", len(train_loader))
        experiment.log_other("val_samples", len(val_loader))
        experiment.log_other("stride", config.stride)
        perplexity = None
        for epoch in range(start_epoch, start_epoch + current_epochs):
            torch.cuda.reset_peak_memory_stats()
            iter_step = 0
            try:
                model.train()
                total_loss = 0

                train_iter = tqdm(train_loader, desc=f"Epoch {epoch}")
                for X, Y in train_iter:
                    X, Y = X.to('cuda', non_blocking=True, dtype=torch.long), Y.to('cuda', non_blocking=True, dtype=torch.long)

                    optimizer.zero_grad(set_to_none=True)

                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(X)
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            Y.view(-1),
                        )

                    lr = get_lr(global_step, config.lr, warmup_iters, min_lr, lr_decay_iters)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    train_iter.set_postfix(batch_loss=f"{loss.item():.4f}", lr=f"{lr:.8f}")
                    experiment.log_metric("batch_loss", loss.item(), step=global_step)

                    scaler.scale(loss).backward()
                    iter_step += 1

                    scaler.unscale_(optimizer)

                    if iter_step % 50 == 0:
                        log_gradients(model, experiment, global_step)
                        #log_memory_usage(experiment, global_step)

                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()
                    global_step += 1


                # Валидация
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
                avg_train_loss = total_loss / len(train_loader)
                current_lr = lr
                logging.info(
                    f"Epoch {epoch} | "
                    f"Train Loss: {avg_train_loss:.3f} | "
                    f"Val Loss: {val_loss:.3f} | "
                    f"Val Perplexity: {perplexity:.2f} | "
                    f"LR: {current_lr:.2e}"
                )

                experiment.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "val_perplexity": perplexity,
                    "epoch": epoch,
                    "gpu_memory_peak": torch.cuda.max_memory_allocated() / 1e9,
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9
                }, step=global_step)

                # чекпоинт
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": config,
                    "global_step": global_step,
                    "experiment_id": experiment.get_key()
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