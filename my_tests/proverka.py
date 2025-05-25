import matplotlib.pyplot as plt
import numpy as np

def plot_token_distribution(file_path, vocab_size=50257, top_k=50):
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    unique, counts = np.unique(data, return_counts=True)
    sorted_indices = np.argsort(-counts)
    plt.bar(range(top_k), counts[sorted_indices[:top_k]])
    plt.xlabel("Токен")
    plt.ylabel("Частота")
    plt.title(f"Топ-{top_k} токенов")
    plt.show()

plot_token_distribution("E:/PyCharm 2024.3.5/projects/data/openwebtext/train_stride_8.bin")