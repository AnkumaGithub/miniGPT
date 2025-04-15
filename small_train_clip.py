import numpy as np
import os
target_train_tokens = 200_000_000

# Путь к файлу val.bin
val_path = "E:/PyCharm 2024.3.5/projects/data/openwebtext/trainbackup.bin"


split = 'train_stride_512_5h_500m'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')

target_train_tokens_512 = 599_998_464 # для stride=512 (чтобы делилось по 512 по 16)

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_512:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens_512,))

    #данные до целевого размера
    new_val[:] = val_data[:target_train_tokens_512] #0m - 600m
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens_512} токенов.")

split = 'train_stride_256_4h_200m'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')

target_train_tokens_256 = 199_999_488 # для stride=128 (чтобы делилось по 512 по 16)

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_256:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens_256,))

    #данные до целевого размера
    new_val[:] = val_data[target_train_tokens_512:(target_train_tokens_512 + target_train_tokens_256)] #600m - 800m
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens_256} токенов.")

split = 'train_stride_128_4h_100m'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')
target_train_tokens_128 = 99_999_744 # для stride=128 (чтобы делилось по 512 по 16)

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_256:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens_128,))

    #данные до целевого размера
    new_val[:] = val_data[(target_train_tokens_512 + target_train_tokens_256):(target_train_tokens_512 + target_train_tokens_256 + target_train_tokens_128)] #800m - 900m
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens_128} токенов.")

split = 'train_stride_64_4h_50m'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')
target_train_tokens_64 = 49_995_776 # для stride=64 (чтобы делилось по 512 по 16)

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_64:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens_64,))

    #данные до целевого размера
    new_val[:] = val_data[(target_train_tokens_512 + target_train_tokens_256 + target_train_tokens_128):(target_train_tokens_512 + target_train_tokens_256 + target_train_tokens_128 + target_train_tokens_64)] #900m - 950m
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens_64} токенов.")

last_tok = target_train_tokens_512 + target_train_tokens_256 + target_train_tokens_128 + target_train_tokens_64

# Добавим валидацию----------------------------------------------------------------

split = 'val_stride_512_5h_20m'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')

target_val_tokens_512 = 29_999_104 # для stride=512 (чтобы делилось по 512 по 16)

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_val_tokens_512:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_val_tokens_512,))

    #данные до целевого размера
    new_val[:] = val_data[last_tok:(target_val_tokens_512 + last_tok)] #950m - 980m
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_val_tokens_512} токенов.")

split = 'val_stride_256_4h_10m'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')

target_val_tokens_256 = 9_830_400 # для stride=128 (чтобы делилось по 512 по 16)

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_val_tokens_256:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_val_tokens_256,))

    #данные до целевого размера
    new_val[:] = val_data[(last_tok + target_val_tokens_512):(target_val_tokens_512 + target_val_tokens_256 + last_tok)] #980m - 990m
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_val_tokens_256} токенов.")

split = 'val_stride_128_4h_5m'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')
target_val_tokens_128 = 4_915_200 # для stride=128 (чтобы делилось по 512 по 16)

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens_256:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_val_tokens_128,))

    #данные до целевого размера
    new_val[:] = val_data[(last_tok + target_val_tokens_512 + target_val_tokens_256):(target_val_tokens_512 + target_val_tokens_256 + target_val_tokens_128 + last_tok)] #990m - 995m
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_val_tokens_128} токенов.")

split = 'val_stride_64_4h_3m'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split}.bin')
target_val_tokens_64 = 2_457_600 # для stride=64 (чтобы делилось по 512 по 16)

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_val_tokens_64:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_val_tokens_64,))

    #данные до целевого размера
    new_val[:] = val_data[(last_tok + target_val_tokens_512 + target_val_tokens_256 + target_val_tokens_128):(target_val_tokens_512 + target_val_tokens_256 + target_val_tokens_128 + target_val_tokens_64 + last_tok)] #995m - 998m
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_val_tokens_64} токенов.")