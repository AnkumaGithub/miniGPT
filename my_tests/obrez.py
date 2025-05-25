import numpy as np
import os
target_train_tokens = 199_999_488

# Путь к файлу val.bin
val_path = "/data/openwebtext/trainbackup.bin"

split = 'train_stride_128'
output_path = os.path.join("/data/openwebtext", f'{split}.bin')

target_train_tokens_128 = 99_999_744 # для stride=128

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_128:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens_128,))

    #данные до целевого размера
    new_val[:] = val_data[target_train_tokens * 3:(target_train_tokens_128 + target_train_tokens * 3)]
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens_128} токенов.")

split = 'train_stride_512'
output_path = os.path.join("/data/openwebtext", f'{split}.bin')

target_train_tokens_512 = 399_998_976 # для stride=512

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_512:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens_512,))

    #данные до целевого размера
    new_val[:] = val_data[target_train_tokens:(target_train_tokens_512 + target_train_tokens)]
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens_512} токенов.")