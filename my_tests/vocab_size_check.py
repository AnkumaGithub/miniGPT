import argparse
import numpy as np
from transformers import GPT2Tokenizer


def check_data_vocab(file_path, expected_vocab_size):
    # Загружаем данные с помощью memmap для экономии памяти
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    min_token = int(np.min(data))
    max_token = int(np.max(data))
    print(f"Найден диапазон токенов: min = {min_token}, max = {max_token}")

    if max_token >= expected_vocab_size:
        print(f"WARNING: Найден токен с id = {max_token}, что превышает ожидаемый vocab_size = {expected_vocab_size}.")
    else:
        print("Все токены находятся в пределах ожидаемого диапазона vocab_size.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка bin-файла данных на соответствие vocab_size.")
    parser.add_argument("E:/something/train_stride_512_5h_500m.bin", help="Путь к bin-файлу с данными.")
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Ожидаемый размер словаря. Если не указан, будет использован vocab_size GPT-2 токенизатора.")
    args = parser.parse_args()

    # Если vocab_size не указан, загружаем GPT-2 токенизатор
    if args.vocab_size is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        expected_vocab_size = tokenizer.vocab_size
        print(f"Используется vocab_size GPT-2 токенизатора: {expected_vocab_size}")
    else:
        expected_vocab_size = args.vocab_size

    check_data_vocab(args.file_path, expected_vocab_size)
