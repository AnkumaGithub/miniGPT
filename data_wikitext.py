from datasets import load_dataset
import numpy as np
import os
import tiktoken
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

BLOCK_SIZE = 512
ENCODING = "gpt2"
OUTPUT_DIR = "data/wikitext"
SPECIAL_TOKENS = ["[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]

def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    enc = tiktoken.get_encoding(ENCODING)
    # Добавляем специальные токены с уникальными ID
    special_tokens_dict = {token: len(enc._mergeable_ranks) + idx for idx, token in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={**enc._special_tokens, **special_tokens_dict}
    )

    # Получаем ID токенов для проверки
    eos_id = enc.encode_single_token("[EOS]")
    for token in SPECIAL_TOKENS:
        token_id = enc.encode_single_token(token)
        logging.info(f"Токен '{token}' имеет ID: {token_id}")

    dtype = np.uint16  # GPT-2 vocab_size (50257) + 4 < 65535
    logging.info(f"Using tokenizer: {ENCODING}, dtype: {dtype}")

    # Загрузка данных
    logging.info("Loading wikitext...")
    try:
        dataset_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        dataset_val = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    except Exception as e:
        logging.error(f"Dataset loading failed: {str(e)}")
        return

    # Обработка текста с добавлением EOS
    def process_text(example):
        text = example['text'].replace('\n', ' ').strip()
        ids = enc.encode_ordinary(text)
        ids.append(eos_id)  # Один EOS в конце текста

        chunks = []
        for i in range(0, len(ids), BLOCK_SIZE):
            chunk = ids[i:i + BLOCK_SIZE]
            chunks.append(chunk)
        return {'ids': chunks, 'len': len(ids)}

    # Сохранение данных
    def process_split(dataset, split_name):
        tokenized = dataset.map(
            process_text,
            remove_columns=['text'],
            desc=f"Tokenizing {split_name}",
            num_proc=4
        ).filter(lambda x: len(x['ids']) > 0)

        output_path = os.path.join(OUTPUT_DIR, f'{split_name}.bin')
        total_tokens = sum(len(chunk) for example in tokenized for chunk in example['ids'])

        arr = np.memmap(output_path, dtype=dtype, mode='w+', shape=(total_tokens,))
        idx = 0
        for example in tokenized:
            for chunk in example['ids']:
                arr[idx:idx + len(chunk)] = chunk
                idx += len(chunk)
        arr.flush()
        logging.info(f"Saved {idx} tokens to {output_path}")

    process_split(dataset_train, "wiki_train_256")
    process_split(dataset_val, "wiki_val_256")

if __name__ == "__main__":
    #prepare_data()
    arr = np.memmap("data/wikitext/wiki_train_256.bin", dtype=np.uint16, mode='r')
    print(arr[:900])