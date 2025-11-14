from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config,
    LlamaForCausalLM, LlamaConfig, GPTJForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from random import sample

from custom_dataset import CustomDataset

#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128
TEMPERATURE = 2.0
ALPHA = 0.5
#############

PATH = Path("./")

# Teacher model: Fine-tuned GPT-2 Large
teacher_dir = PATH / 'models/GPT2-Large-BabyLM'

# Student model: GPT-2 Small (random initialization)
MODEL_NAME = f'GPT2-Small-Distilled'
MODEL_OUTPUT = Path('./models') / MODEL_NAME
EVAL_SAMPLES = 8192


# Load tokenizer - Use GPT-2 original tokenizer (consistent with teacher model)
# Teacher model was trained with GPT-2 original tokenizer, so use the same tokenizer here
print(f"Loading GPT-2 tokenizer from teacher model: {teacher_dir}")
tokenizer = GPT2TokenizerFast.from_pretrained(teacher_dir)
# GPT-2 doesn't have pad_token, use eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = SEQ_LENGTH

# Load datasets - Use BabyLM data
BABYLM_TRAIN_PATH = "corpus_split/train_babylm.txt"  # BabyLM training set
BABYLM_VAL_PATH = "corpus_split/val_babylm.txt"      # BabyLM validation set

train_dataset = CustomDataset(
    data_path=BABYLM_TRAIN_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=True  # Using random chunks for training
)

val_dataset = CustomDataset(
    data_path=BABYLM_VAL_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=False  # No random chunks for validation
)

# Sample evaluation subset if needed
eval_indices = sample(range(len(val_dataset)), min(EVAL_SAMPLES, len(val_dataset)))
eval_dataset = Subset(val_dataset, eval_indices)

tokenizer.model_max_length = SEQ_LENGTH

# Student model: GPT-2 Small architecture, random initialization
student_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=2*SEQ_LENGTH,
    n_embd=768,      # GPT-2 Small hidden size
    n_layer=12,      # GPT-2 Small number of layers
    n_head=12,       # GPT-2 Small number of attention heads
    pad_token_id=tokenizer.pad_token_id,  # Use tokenizer's pad_token_id (usually eos_token_id)
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


print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher = {teacher.num_parameters()}')



#  Distillation Trainer
#  We modified the Trainer from this repo https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
# to work with an ensemble of teachers


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


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
    save_strategy = "epoch",
    eval_strategy = "epoch",  # Fixed: evaluation_strategy has been changed to eval_strategy
    num_train_epochs=6,
    report_to=[],
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,  # Set to zero to avoid saving
    warmup_steps=200, 
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # eval_loss should be minimized
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

    )

trainer.train()

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)
