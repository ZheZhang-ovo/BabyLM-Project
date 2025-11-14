# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from pathlib import Path
from custom_dataset import CustomDataset  # Use your own dataset
import math

# 1. Paths
# model_dir = "/home/zhezhang/BabyLM/models/GPT2-Large-BabyLM"  # Your teacher directory
model_dir = "gpt2-large"
val_path = "./corpus_split/val_babylm.txt"
SEQ_LENGTH = 128  # Keep consistent with training

# 2. Load tokenizer & model (use original GPT-2 tokenizer)
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have pad, use eos instead
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Build validation dataset
print("Building eval dataset...")
eval_dataset = CustomDataset(val_path, SEQ_LENGTH, tokenizer=tokenizer, offset=0)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 4. Build Trainer, only for evaluation, no training
args = TrainingArguments(
    output_dir="./tmp_eval",
    per_device_eval_batch_size=8,
    dataloader_num_workers=2,
    fp16=False,  # Adjust based on your environment
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("Running evaluation...")
metrics = trainer.evaluate()
print(metrics)

eval_loss = metrics["eval_loss"]
ppl = math.exp(eval_loss)
print(f"\nEval loss = {eval_loss:.4f}")
print(f"Perplexity = {ppl:.2f}")
