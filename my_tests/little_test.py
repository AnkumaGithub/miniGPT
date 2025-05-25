import numpy as np
import os

# Путь к файлу val.bin
val_path = "/data/openwebtext/val.bin"
split = 'test'
output_path = os.path.join("/data/openwebtext", f'{split}.bin')

target_val_tokens = int(1_000_000_000 * 0.0005)  # 500_000 токенов

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_val_tokens:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_val_tokens,))

    #данные до целевого размера
    new_val[:] = val_data[:target_val_tokens]
    new_val.flush()
    del new_val

    print(f"test.bin до {target_val_tokens} токенов.")