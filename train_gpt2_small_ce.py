from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

from pathlib import Path
import torch
from torch.utils.data import Subset
from random import sample

# Ensure this file exists in your directory, or copy its class definition here
from custom_dataset import CustomDataset

# ============================================================
# Hyperparameters (Updated for 100M Strategy)
# ============================================================
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 512  # [Critical] Must be 512, adapt to subsequent RL and Teacher
EVAL_SAMPLES = 8192
EPOCHS = 10       # [Critical] Run full 10 Epochs

PATH = Path("./")

# ============================================================
# Paths & Configuration
# ============================================================
MODEL_NAME = "GPT2-Small-BabyLM-CE"
MODEL_OUTPUT = PATH / "models" / MODEL_NAME

# [Critical] Ensure this points to your 100M data folder
BABYLM_TRAIN_PATH = "corpus_split_100M/train_babylm.txt"
BABYLM_VAL_PATH = "corpus_split_100M/val_babylm.txt"

# ============================================================
# 1. Load Tokenizer (Modified)
# ============================================================
# [Modified] No longer rely on local teacher path, use official GPT-2 tokenizer directly
# This way, even if Teacher is not trained yet, this can run
print("Loading standard GPT-2 tokenizer from Hugging Face...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# GPT-2 has no pad token by default, must specify manually, otherwise batch training will error
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = SEQ_LENGTH

# ============================================================
# 2. Prepare Datasets
# ============================================================
print(f"Building BabyLM train dataset from: {BABYLM_TRAIN_PATH}")
train_dataset = CustomDataset(
    data_path=BABYLM_TRAIN_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=True,
)

print(f"Building BabyLM val dataset from: {BABYLM_VAL_PATH}")
val_dataset = CustomDataset(
    data_path=BABYLM_VAL_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=False,
)

# Randomly sample a subset for evaluation to save time
eval_indices = sample(range(len(val_dataset)), min(EVAL_SAMPLES, len(val_dataset)))
eval_dataset = Subset(val_dataset, eval_indices)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ============================================================
# 3. Initialize Student Model (Random Init)
# ============================================================
student_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,  # Give enough capacity, greater than 512 is fine
    n_embd=768,        # GPT-2 Small
    n_layer=12,
    n_head=12,
    pad_token_id=tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(student_config)
print(f"Student (baseline) model initialized from scratch. Parameters: {model.num_parameters()}")

# ============================================================
# 4. Training Arguments
# ============================================================
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    
    # Saving & Eval Strategy
    save_strategy="epoch",
    eval_strategy="epoch",
    
    # [Critical Modification] Set to None, keep checkpoint for every Epoch
    # This way you can use Epoch 2 model for distillation, and Epoch 10 for Baseline later
    save_total_limit=None, 
    
    # Training Hyperparameters
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,  # [Critical] 32 * 4 = 128 effective batch size
    
    # Optimization
    learning_rate=LR,
    weight_decay=0.1,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    
    # System & Logging
    logging_steps=20,
    fp16=True,
    report_to=[],  # Can fill ["wandb"] if you need
    
    # Load Best Model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ============================================================
# 5. Run Training
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Starting training...")
trainer.train()

# Save final model
print(f"Saving final model to {MODEL_OUTPUT}...")
trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)

print("Training finished successfully!")