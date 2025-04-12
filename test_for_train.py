import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from train import GPTDataset, train
from model import GPTConfig
import os
from pathlib import Path


@pytest.fixture
def mock_config():
    return GPTConfig(
        block_size=64,
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        drop_path_rate=0.1,
        bias=False
    )


@pytest.fixture
def mock_dataset(tmp_path, monkeypatch):
    # Создаем структуру директорий как в коде
    data_dir = tmp_path / "data" / "openwebtext"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Генерируем тестовые данные
    train_data = np.random.randint(0, 50257, size=1000, dtype=np.uint16)
    train_path = data_dir / "train.bin"
    train_data.tofile(train_path)

    val_path = data_dir / "val.bin"
    train_data.tofile(val_path)

    # Меняем текущую директорию на временную
    monkeypatch.chdir(tmp_path)

    return tmp_path


def test_dataset_creation(mock_dataset):
    dataset = GPTDataset('train', block_size=64)
    assert len(dataset) == 1000 - 64

    x, y = dataset[0]
    assert x.shape == (64,)
    assert y.shape == (64,)


def test_dataloader(mock_dataset):
    dataset = GPTDataset('train', block_size=64)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    batch = next(iter(loader))
    x, y = batch
    assert x.shape == (8, 64)
    assert y.shape == (8, 64)


@patch('train.Experiment')
@patch('torch.cuda.is_available', return_value=True)
def test_training_step(mock_cuda, mock_experiment, mock_config, mock_dataset):
    experiment_mock = Mock()
    mock_experiment.return_value = experiment_mock

    # Мокируем модель для работы на CPU
    with patch('model.GPT') as mock_gpt:
        mock_model = mock_gpt.return_value
        mock_model.parameters.return_value = [torch.randn(10)]

        try:
            train()
        except Exception as e:
            pytest.fail(f"Training failed: {str(e)}")

    assert experiment_mock.log_parameters.called

