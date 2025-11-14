# -*- coding: utf-8 -*-
"""
Evaluate and compare baseline and distilled student models on BabyLM validation set.
Compares a randomly initialized GPT-2 Small trained with cross-entropy loss
against a randomly initialized GPT-2 Small trained with knowledge distillation.
"""

import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, DataCollatorForLanguageModeling
from custom_dataset import CustomDataset

# Configuration
PATH = Path("./")
SEQ_LENGTH = 128  # Sequence length (must match training configuration)
BATCH_SIZE = 32
STUDENT_MODEL_DIR = PATH / "models/GPT2-Small-Distilled"  # Distilled student model
TEACHER_TOKENIZER_DIR = PATH / "models/GPT2-Large-BabyLM"  # Tokenizer directory (same as teacher)
BABYLM_VAL_PATH = "corpus_split/val_babylm.txt"  # Validation dataset path

def build_val_loader(tokenizer):
    """
    Build validation data loader.
    
    Args:
        tokenizer: Tokenizer to use for encoding
        
    Returns:
        DataLoader: Validation data loader
    """
    val_dataset = CustomDataset(
        data_path=BABYLM_VAL_PATH,
        seq_length=SEQ_LENGTH,
        tokenizer=tokenizer,
        random_chunk=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    return val_loader

def evaluate_model(model, dataloader, device):
    """
    Evaluate a model on the validation dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for validation set
        device: Device to run evaluation on (cuda/cpu)
        
    Returns:
        tuple: (average_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )
            loss = outputs.loss
            # Count only valid tokens (ignore padding tokens with label -100)
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)  # Perplexity = exp(cross-entropy loss)
    return avg_loss, ppl


def main():
    """
    Main evaluation function.
    Compares baseline (random init + cross-entropy) vs distilled student model.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer (same as used for training)
    print(f"Loading tokenizer from: {TEACHER_TOKENIZER_DIR}")
    tokenizer = GPT2TokenizerFast.from_pretrained(TEACHER_TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = SEQ_LENGTH

    # Build validation data loader
    print("Building validation data loader...")
    val_loader = build_val_loader(tokenizer)

    # Load baseline model (random init + cross-entropy training)
    print(f"\nLoading baseline model from: {PATH / 'models/GPT2-Small-BabyLM-CE'}")
    baseline_model_dir = PATH / "models/GPT2-Small-BabyLM-CE"
    baseline_model = GPT2LMHeadModel.from_pretrained(baseline_model_dir)
    baseline_model.to(device)

    # Load distilled student model
    print(f"Loading distilled student model from: {STUDENT_MODEL_DIR}")
    student_model = GPT2LMHeadModel.from_pretrained(STUDENT_MODEL_DIR)
    student_model.to(device)

    # Evaluate baseline model
    print("\n" + "="*50)
    print("Evaluating baseline model (random init + CE)...")
    print("="*50)
    base_loss, base_ppl = evaluate_model(baseline_model, val_loader, device)
    print("Baseline GPT2 small:")
    print(f"  Loss: {base_loss:.6f}")
    print(f"  PPL:  {base_ppl:.2f}")

    # Evaluate distilled student model
    print("\n" + "="*50)
    print("Evaluating distilled student model...")
    print("="*50)
    student_loss, student_ppl = evaluate_model(student_model, val_loader, device)
    print("Distilled student:")
    print(f"  Loss: {student_loss:.6f}")
    print(f"  PPL:  {student_ppl:.2f}")

    # Print comparison
    print("\n" + "="*50)
    print("Comparison (student - baseline):")
    print("="*50)
    loss_diff = student_loss - base_loss
    ppl_diff = student_ppl - base_ppl
    print(f"  Loss diff: {loss_diff:.6f} ({'↓' if loss_diff < 0 else '↑'})")
    print(f"  PPL diff:  {ppl_diff:.2f} ({'↓' if ppl_diff < 0 else '↑'})")
    
    if ppl_diff < 0:
        improvement = abs(ppl_diff / base_ppl * 100)
        print(f"\n✓ Distilled model shows {improvement:.1f}% improvement in perplexity!")

if __name__ == "__main__":
    main()
