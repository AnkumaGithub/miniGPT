from datasets import load_dataset
import numpy as np
import os
import tiktoken
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

BLOCK_SIZE = 512
ENCODING = "gpt2"
OUTPUT_DIR = "data/wiki_qa"
SPECIAL_TOKENS = ["[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]

def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enc = tiktoken.get_encoding(ENCODING)
    dtype = np.uint16

    # Добавляем специальные токены
    enc = tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={**enc._special_tokens, **{token: len(enc._mergeable_ranks) + i for i, token in enumerate(SPECIAL_TOKENS)}}
    )

    # Проверка ID токенов
    for token in SPECIAL_TOKENS:
        token_id = enc.encode_single_token(token)
        logging.info(f"Токен '{token}' имеет ID: {token_id}")

    # Загрузка данных
    logging.info("Loading Microsoft Wiki QA...")
    dataset = load_dataset("microsoft/wiki_qa")

    # Форматирование текста (без фильтрации по меткам)
    def process_example(example):
        text = f"[Q] {example['question']} [SEP] [A] {example['answer']} [EOS]"
        return {'text': text}

    # Обработка данных
    def process_split(split):
        logging.info(f"Обработка {split}...")
        processed = dataset[split].map(
            process_example,
            remove_columns=['question_id', 'question', 'answer', 'label', 'document_title'],
            num_proc=4
        )

        # Токенизация
        def tokenize(example):
            ids = enc.encode_ordinary(example['text'])
            return {'ids': ids, 'len': len(ids)}

        tokenized = processed.map(
            tokenize,
            remove_columns=['text'],
            num_proc=4,
            desc="Tokenizing"
        )

        # Сохранение в бинарный формат
        output_path = os.path.join(OUTPUT_DIR, f"{split}.bin")
        total_tokens = sum(example['len'] for example in tokenized)
        arr = np.memmap(output_path, dtype=dtype, mode='w+', shape=(total_tokens,))
        idx = 0

        for example in tokenized:
            tokens = example['ids']
            arr[idx: idx + len(tokens)] = tokens
            idx += len(tokens)
        arr.flush()
        logging.info(f"{split}: Сохранено {idx} токенов")

    # Обработка всех сплитов
    process_split('train')
    process_split('test')
    process_split('validation')

if __name__ == "__main__":
    #prepare_data()
    arr = np.memmap("data/wiki_qa/train.bin", dtype=np.uint16, mode='r')
    print(arr[:900])