import os

import comet_ml
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, GPT2TokenizerFast, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from datasets import load_from_disk
import tqdm

os.environ["TMPDIR"] = "E:/temp_pytorch"
os.environ["TEMP"] = "E:/temp_pytorch"

from datasets import load_from_disk

load_dotenv(dotenv_path='../../.env')

os.environ["COMET_API_KEY"] = os.getenv("COMET_API_KEY")

dataset = load_from_disk("/data/tinystories_HF")
tokenizer = GPT2TokenizerFast.from_pretrained("/data/tinystories_HF")
SPECIAL_TOKENS = ["[EOS]", "[PAD]"]
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[EOS]"

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    fan_in_fan_out=True,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="../../results",
    per_device_train_batch_size=20,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    logging_dir="./logs",
    learning_rate=5e-5,
    fp16=True,
    save_steps=1000,
    gradient_checkpointing=False,
    logging_strategy="steps",
    logging_steps=5,
    disable_tqdm=False,
    report_to=["comet_ml"],
    label_names=["labels"]
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

experiment = comet_ml.Experiment(
    project_name="gpt2_FT",
    auto_param_logging=False,
)
experiment.log_parameters({
    "model": "gpt2",
    "peft": "LoRA",
    "block_size": 256,
})

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)
trainer.train()