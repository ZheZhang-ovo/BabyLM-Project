from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

from pathlib import Path
import argparse # Import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from random import sample

from custom_dataset import CustomDataset

# ----------------------------------------------
# 1. Parse command line arguments
# ----------------------------------------------
parser = argparse.ArgumentParser(description="Knowledge Distillation Training Script")
parser.add_argument(
    "--resume_path", 
    type=str, 
    default=None, 
    help="Path to a checkpoint folder (e.g., '../models/GPT2-Small-Distilled-100M-9Epochs/checkpoint-4960') to resume training from."
)
args = parser.parse_args()
RESUME_PATH_INPUT = args.resume_path


torch.cuda.empty_cache()
torch.cuda.synchronize()

#############
LR = 2.5e-4
BATCH_SIZE = 16
SEQ_LENGTH = 512
TEMPERATURE = 2.0
ALPHA = 0.5
#############

# Teacher model: Fine-tuned GPT-2 Large
teacher_dir = "./models/GPT2-Large-BabyLM-100M-Unsloth-Merged" # Pretrained GPT-2 Large

# Student model: GPT-2 Small (random initialization)
MODEL_NAME = f'GPT2-Unsloth-900M'
MODEL_OUTPUT = Path('../models') / MODEL_NAME
EVAL_SAMPLES = 8192


# Load tokenizer 
print(f"Loading GPT-2 tokenizer from teacher model: {teacher_dir}")
tokenizer = GPT2TokenizerFast.from_pretrained(
    teacher_dir,
    model_max_length=SEQ_LENGTH
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load datasets
BABYLM_TRAIN_PATH = "corpus_split_100M/train_babylm.txt"  
BABYLM_VAL_PATH = "corpus_split_100M/val_babylm.txt"      

train_dataset = CustomDataset(
    data_path=BABYLM_TRAIN_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=True 
)

val_dataset = CustomDataset(
    data_path=BABYLM_VAL_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=False
)

# Sample evaluation subset
eval_indices = sample(range(len(val_dataset)), min(EVAL_SAMPLES, len(val_dataset)))
eval_dataset = Subset(val_dataset, eval_indices)

# Student model: GPT-2 Small architecture, random initialization
student_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=2*SEQ_LENGTH,
    n_embd=768,      # GPT-2 Small hidden size
    n_layer=12,      # GPT-2 Small number of layers
    n_head=12,       # GPT-2 Small number of attention heads
    pad_token_id=tokenizer.pad_token_id, 
)

student = GPT2LMHeadModel(student_config)
print(f'Student model parameters = {student.num_parameters()}')

# Teacher model: Fine-tuned GPT-2 Large
teacher = GPT2LMHeadModel.from_pretrained(teacher_dir)
print(f'Teacher model parameters = {teacher.num_parameters()}')
teachers = [teacher]


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Distillation Trainer Arguments
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


from transformers import TrainerCallback

# Word Milestone Callback
class WordMilestoneCB(TrainerCallback):
    def __init__(self, seq_len, grad_acc, bsz, tok_per_word=1.33):
        self.toks = 0
        self.seq_len = seq_len
        self.grad_acc = grad_acc
        self.bsz = bsz
        self.tok_per_word = tok_per_word
        
        # Milestones in millions of words
        ms = list(range(1, 11)) + [i * 10 for i in range(2, 11)] + [i * 100 for i in range(2, 11)]
        self.milestones_m = ms
        self.milestone_toks = [int(m * 1e6 * self.tok_per_word) for m in ms]
        self.next_milestone_idx = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Check if this is the first step of a resumed run, 
        # in which case state.global_step should be used to initialize self.toks
        if self.toks == 0 and state.global_step > 0:
            # Calculate toks based on effective batch size (per device) and current step
            effective_bs_per_device = self.bsz * self.seq_len * self.grad_acc
            self.toks = state.global_step * effective_bs_per_device
            
            # Adjust next milestone index based on current tokens
            while self.next_milestone_idx < len(self.milestone_toks) and self.toks >= self.milestone_toks[self.next_milestone_idx]:
                 self.next_milestone_idx += 1


        # Calculate tokens processed in this step
        step_toks = self.bsz * self.seq_len * self.grad_acc
        self.toks += step_toks
        
        if self.next_milestone_idx < len(self.milestone_toks):
            next_milestone = self.milestone_toks[self.next_milestone_idx]
            if self.toks >= next_milestone:
                
                model = kwargs.get('model')
                tokenizer = kwargs.get('tokenizer')
                
                milestone_m = self.milestones_m[self.next_milestone_idx]
                checkpoint_name = f"checkpoint_{milestone_m}M"
                checkpoint_path = Path(args.output_dir) / checkpoint_name
                
                print(f"Milestone reached: {milestone_m}M words ({self.toks} tokens). Saving checkpoint to {checkpoint_name}...")
                model.save_pretrained(checkpoint_path)
                if tokenizer is not None:
                    tokenizer.save_pretrained(checkpoint_path)
                
                self.next_milestone_idx += 1


# Distillation Trainer
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            # place each teacher on same device as student
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # compute teacher output
        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # assert size
        assert outputs_student.logits.size() == avg_teacher_logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch", # Trainer epoch saving is fine as a backup
    eval_strategy="epoch",
    num_train_epochs=9,
    report_to=[],
    gradient_accumulation_steps=8,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=None,
    warmup_steps=200, 
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=True,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
)


trainer = DistillationTrainer(
        student,
        training_args,
        teacher_models=teachers,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WordMilestoneCB(SEQ_LENGTH, training_args.gradient_accumulation_steps, BATCH_SIZE)]
    )

# ----------------------------------------------
# 2. Modify trainer.train() call to support resuming
# ----------------------------------------------
resume_checkpoint = None
if RESUME_PATH_INPUT:
    # Construct complete Checkpoint path
    if Path(RESUME_PATH_INPUT).is_dir():
        # If user provides a complete path, use it directly
        resume_checkpoint = RESUME_PATH_INPUT
    elif (MODEL_OUTPUT / RESUME_PATH_INPUT).is_dir():
        # If user only provides folder name (e.g., 'checkpoint-4960'), construct complete path
        resume_checkpoint = str(MODEL_OUTPUT / RESUME_PATH_INPUT)
    
    if resume_checkpoint:
        print(f"✅ Found Checkpoint. Resuming training from: {resume_checkpoint}")
    else:
        print(f"⚠️ Checkpoint folder not found at {RESUME_PATH_INPUT}. Starting training from scratch.")

trainer.train(resume_from_checkpoint=resume_checkpoint)

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)