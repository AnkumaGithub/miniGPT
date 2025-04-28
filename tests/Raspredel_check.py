import numpy as np
import matplotlib.pyplot as plt


def load_tokens(filename, dtype=np.int32):
    """
    Загружает токены из бинарного файла.
    По умолчанию предполагается, что токены сохранены как np.int32.
    При необходимости замените dtype на нужный тип (например, np.uint16).
    """
    tokens = np.fromfile(filename, dtype=np.uint16)
    return tokens


def count_token_distribution(tokens):
    """
    Подсчитывает частоту встречаемости каждого токена.
    Если все токены – неотрицательные целые числа,
    можно использовать numpy-функцию np.bincount.
    """
    distribution = np.bincount(tokens)
    return distribution


def plot_distribution(distribution, top_n=20):
    """
    Строит график распределения (бар-чарт) для топ-N самых частых токенов.
    """
    # Получаем индексы токенов, которые встречаются
    token_ids = (distribution > 0).nonzero()[0]
    counts = distribution[token_ids]

    # Сортировка по убыванию
    sorted_indices = counts.argsort()[::-1]
    token_ids = token_ids[sorted_indices]
    counts = counts[sorted_indices]

    # Выбираем топ-N
    token_ids = token_ids[:top_n]
    counts = counts[:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar([str(t) for t in token_ids], counts, color='steelblue')
    plt.xlabel('Token ID')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Most Frequent Tokens')
    plt.show()


def main():
    filename = 'data/tinystories/train_256.bin'  # путь к вашему бинарному файлу с токенами
    tokens = load_tokens(filename, dtype=np.uint16)  # измените dtype при необходимости
    print(f"Загружено {len(tokens)} токенов из файла {filename}.")

    distribution = count_token_distribution(tokens)

    # Вывод статистики токенов в консоль (выведем только токены, которые встречаются)
    for token_id, count in enumerate(distribution):
        if count > 0:
            print(f"Token {token_id}: {count}")

    # Построение графика для топ-20 токенов
    plot_distribution(distribution, top_n=20)


if __name__ == '__main__':
    main()
    import tiktoken

    # Замените на свой список ID, полученный из графика
    top_token_ids = [13, 290, 11, 262, 284, 257, 220, 373, 340, 366, 1375, 607, 679, 1119, 531, 383, 339, 287, 673, 351]

    # Загружаем GPT-2 токенизатор
    enc = tiktoken.get_encoding("gpt2")

    # Расшифровываем токены
    decoded_tokens = [enc.decode([token_id]) for token_id in top_token_ids]

    # Выводим таблицу
    print("Token ID | Decoded Token")
    print("-" * 25)
    for token_id, token in zip(top_token_ids, decoded_tokens):
        repr_token = repr(token)  # Показывает специальные символы явно (например, '\n')
        print(f"{token_id:>8} | {repr_token}")

