import numpy as np
import os

# Путь к файлу val.bin
val_path = "/data/wikitext/wiki_train_256.bin"
split1 = 'wiki_train_256_60M'
output_path = os.path.join("/data/wikitext", f'{split1}.bin')

target_train_tokens = 60_000_000

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens,))

    #данные до целевого размера
    new_val[:] = val_data[:target_train_tokens]
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens} токенов.")