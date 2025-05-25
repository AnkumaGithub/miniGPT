import numpy as np
import os

# Путь к файлу val.bin
val_path = "E:/PyCharm 2024.3.5/projects/data/tinystories/train.bin"

split = 'train_150M_4'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/tinystories", f'{split}.bin')

target_train_tokens_128 = 150_000_000 # для stride=128

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_128:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens_128,))

    #данные до целевого размера
    new_val[:] = val_data[target_train_tokens_128*3:target_train_tokens_128*4]
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens_128} токенов.")