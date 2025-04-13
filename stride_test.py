import unittest
import numpy as np
import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self, data, block_size, stride=256):
        self.data = data
        self.block_size = block_size
        self.stride = stride
        self.total_samples = (len(self.data) - 1 - block_size) // stride + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.block_size + 1
        if end > len(self.data):
            start = max(0, len(self.data) - self.block_size - 1)
            end = len(self.data)
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64)).long()
        return chunk[:-1], chunk[1:]

# Юнит-тест
class TestGPTDataset(unittest.TestCase):
    def setUp(self):
        self.fake_data = np.arange(2048, dtype=np.uint16)
        self.block_size = 1024
        self.stride = 256
        self.dataset = GPTDataset(self.fake_data, self.block_size, self.stride)

    def test_sample_shape(self):
        for i in range(len(self.dataset)):
            x, y = self.dataset[i]
            self.assertEqual(x.shape[0], self.block_size)
            self.assertEqual(y.shape[0], self.block_size)

    def test_last_block_alignment(self):
        last_idx = len(self.dataset) - 1
        x, y = self.dataset[last_idx]

        start = last_idx * self.stride
        end = start + self.block_size + 1
        expected_x = torch.arange(start, start + self.block_size)
        expected_y = torch.arange(start + 1, start + self.block_size + 1)

        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_too_short_data(self):
        short_data = np.arange(800, dtype=np.uint16)  # меньше, чем block_size + 1 (1025)
        block_size = 1024
        dataset = GPTDataset(short_data, block_size, stride=256)

        # В этом случае __len__ должен вернуть 0
        self.assertEqual(len(dataset), 0)

        x, y = dataset[0]

        expected_start = max(0, len(short_data) - block_size - 1)
        chunk = short_data[expected_start:]
        expected_x = torch.from_numpy(chunk[:-1].astype(np.int64))
        expected_y = torch.from_numpy(chunk[1:].astype(np.int64))

        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_total_samples(self):
        expected = (len(self.fake_data) - 1 - self.block_size) // self.stride + 1
        self.assertEqual(len(self.dataset), expected)

if __name__ == "__main__":
    unittest.main()
