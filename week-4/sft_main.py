import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from torch.optim import AdamW
from trl import SFTTrainer, SFTConfig

# --------- 1. config 정의 ---------
train_config = {
    "num_train_epochs": 5,
    "save_strategy": "steps",
    "logging_steps": 50,
    "save_steps": 100,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True,
    "bf16": True,
    "max_grad_norm": 1.0,
    "eval_steps": 100,
    "warmup_ratio": 0.2,
    "do_eval": True,
    "save_only_model": True,
    "metric_for_best_model": "eval_loss",
    "output_dir": "./checkpoints",

}

# --------- 2. 모델 및 토크나이저 로드 ---------
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --------- 3. 로컬 데이터셋 로드 & 전처리 ---------
dataset_path = "local_data" # private path
dataset = load_from_disk(dataset_path)

print("✅ 로드된 데이터셋 예시:", dataset[0])

def preprocess_chat(example):
    full_text = example["prompt"]  # 이미 query+response 포함됨
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train = dataset.map(preprocess_chat, remove_columns=dataset.column_names)

# --------- 4. Optimizer & Scheduler ---------
batch_size = 1
steps_per_epoch = len(tokenized_train) // batch_size
num_training_steps = steps_per_epoch * train_config["num_train_epochs"]

optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.005
)

lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=int(train_config["warmup_ratio"] * num_training_steps),
    num_training_steps=num_training_steps
)

# --------- 5. SFTConfig & Trainer ---------
sft_args = SFTConfig(**train_config)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=tokenized_train,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    optimizers=(optimizer, lr_scheduler)
)

trainer.train()
trainer.save_model(sft_args.output_dir)
