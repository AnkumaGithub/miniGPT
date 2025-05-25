from datasets import load_dataset
import numpy as np
import os
import tiktoken
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

BLOCK_SIZE = 512
ENCODING = "gpt2"
OUTPUT_DIR = "../../data/gen_qa_dialog"
SPECIAL_TOKENS = ["[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]


def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Инициализация токенизатора
    enc = tiktoken.get_encoding(ENCODING)
    enc = tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={**enc._special_tokens, **{token: len(enc._mergeable_ranks) + i for i, token in enumerate(SPECIAL_TOKENS)}}
    )

    # Проверка токенов
    for token in SPECIAL_TOKENS:
        logging.info(f"Токен '{token}': {enc.encode_single_token(token)}")

    dtype = np.uint16
    dataset = load_dataset("anaonymous-aad/GenQA_dialog", streaming=True)

    # Функция преобразования диалога в пары "контекст-ответ"
    def process_dialog(example):
        dialog = example["messages"]
        pairs = []
        context = []
        for msg in dialog:
            if msg["role"] == "user":
                # Добавляем реплику пользователя в контекст
                context += f"[USER] {msg['content']} [BOT] "
            else:
                full_context = " ".join(context)
                # Формируем пример: контекст -> ответ ассистента
                target = f"{msg['content']} [EOS]"
                pairs.append({
                    "context": full_context.strip(),
                    "target": target
                })
                # Обновляем контекст для следующего шага
                context += f"{msg['content']} [EOS]"
        return {"pairs": pairs}

    # Обработка данных
    def process_split(split):
        logging.info(f"Обработка {split}...")
        split_dataset = dataset[split]

        # Обработка батчами по 1000 примеров
        batch_size = 1000
        all_pairs = []

        for batch in split_dataset.iter(batch_size=batch_size):
            processed_batch = [process_dialog(ex) for ex in batch]
            for item in processed_batch:
                all_pairs.extend(item["pairs"])


        # Токенизация
        def tokenize(pair):
            input_ids = enc.encode_ordinary(pair["context"])
            target_ids = enc.encode_ordinary(pair["target"])
            return {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "input_len": len(input_ids),
                "target_len": len(target_ids)
            }

        tokenized = [tokenize(pair) for pair in all_pairs]

        # Сохранение в .bin
        output_input_path = os.path.join(OUTPUT_DIR, f"{split}_input.bin")
        output_target_path = os.path.join(OUTPUT_DIR, f"{split}_target.bin")

        # Сохраняем inputs и targets отдельно
        total_input_tokens = sum(ex["input_len"] for ex in tokenized)
        total_target_tokens = sum(ex["target_len"] for ex in tokenized)

        arr_input = np.memmap(output_input_path, dtype=dtype, mode="w+", shape=(total_input_tokens,))
        arr_target = np.memmap(output_target_path, dtype=dtype, mode="w+", shape=(total_target_tokens,))

        idx_input = 0
        idx_target = 0
        for ex in tokenized:
            arr_input[idx_input: idx_input + ex["input_len"]] = ex["input_ids"]
            arr_target[idx_target: idx_target + ex["target_len"]] = ex["target_ids"]
            idx_input += ex["input_len"]
            idx_target += ex["target_len"]

        arr_input.flush()
        arr_target.flush()
        logging.info(f"{split}: Сохранено {idx_input} входных и {idx_target} целевых токенов")

    # Обработка всех сплитов
    process_split("train")
    process_split("validation")
    process_split("test")


if __name__ == "__main__":
    prepare_data()
