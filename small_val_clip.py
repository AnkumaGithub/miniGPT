import numpy as np
import os
target_train_tokens = 199_999_488

# Путь к файлу val.bin
val_path = "E:/PyCharm 2024.3.5/projects/data/openwebtext/trainbackup.bin"

split = 'val_stride_128'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')

target_train_tokens_128 = 2_560_000 # для stride=128

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_128:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens_128,))

    #данные до целевого размера
    new_val[:] = val_data[(target_train_tokens * 4 + 100_000_000):(target_train_tokens_128 + target_train_tokens * 4 + 100_000_000)]
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens_128} токенов.")

split = 'val_stride_32'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')

target_val_tokens_32 = 614_400 # для stride=32

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_val_tokens_32:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_val_tokens_32,))

    #данные до целевого размера
    new_val[:] = val_data[(target_train_tokens * 4 + 100_000_000 + target_train_tokens_128):(target_val_tokens_32 + target_train_tokens_128 + target_train_tokens * 4 + 100_000_000)]
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_val_tokens_32} токенов.")
