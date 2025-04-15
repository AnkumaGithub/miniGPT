import numpy as np
import os

# Путь к файлу val.bin
val_path = "E:/PyCharm 2024.3.5/projects/data/openwebtext/train_stride_512_5h_500m.bin"
split1 = 'train_stride_256_4h_200m_1'
split2 = 'train_stride_256_4h_200m_2'
split3 = 'train_stride_256_4h_200m_3'
output_path = os.path.join("E:/PyCharm 2024.3.5/projects/data/openwebtext", f'{split3}.bin')

target_train_tokens_512 = 599_998_464
target_train_tokens = 199_999_488

#данные из val.bin
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
if len(val_data) <= target_train_tokens:
    print("Файл уже меньше или равен целевому размеру. Обрезка не требуется.")
else:
    #новый memmap-файл с целевым размером
    new_val = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(target_train_tokens,))

    #данные до целевого размера
    new_val[:] = val_data[target_train_tokens * 2:target_train_tokens + target_train_tokens + target_train_tokens]
    new_val.flush()
    del new_val

    print(f"train.bin обрезан до {target_train_tokens} токенов.")