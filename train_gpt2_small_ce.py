from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

from pathlib import Path
import torch
from torch.utils.data import Subset
from random import sample

from custom_dataset import CustomDataset

LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128
EVAL_SAMPLES = 8192

PATH = Path("./")

# tokenizer dir (same as teacher tokenizer, for consistency)
TOKENIZER_DIR = PATH / "models/GPT2-Large-BabyLM"

MODEL_NAME = "GPT2-Small-BabyLM-CE"
MODEL_OUTPUT = PATH / "models" / MODEL_NAME

BABYLM_TRAIN_PATH = "corpus_split/train_babylm.txt"
BABYLM_VAL_PATH = "corpus_split/val_babylm.txt"

print(f"Loading GPT-2 tokenizer from: {TOKENIZER_DIR}")
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = SEQ_LENGTH

print("Building BabyLM train dataset...")
train_dataset = CustomDataset(
    data_path=BABYLM_TRAIN_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=True,
)

print("Building BabyLM val dataset...")
val_dataset = CustomDataset(
    data_path=BABYLM_VAL_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=False,
)

eval_indices = sample(range(len(val_dataset)), min(EVAL_SAMPLES, len(val_dataset)))
eval_dataset = Subset(val_dataset, eval_indices)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

student_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=2 * SEQ_LENGTH,
    n_embd=768,
    n_layer=12,
    n_head=12,
    pad_token_id=tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(student_config)
print(f"Student (baseline) model parameters: {model.num_parameters()}")

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",
    eval_strategy="epoch",  # keep same style as your distillation script
    num_train_epochs=6,
    report_to=[],
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)

print("Training finished. Model saved to:", MODEL_OUTPUT)
