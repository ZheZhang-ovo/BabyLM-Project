from transformers import (
    GPT2Config, GPT2LMHeadModel,
    BitsAndBytesConfig,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2TokenizerFast,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch.utils.data import Subset
from random import sample, seed
from pathlib import Path
import yaml
import argparse
import torch
import torch.nn as nn

from custom_dataset import CustomDataset


def build_model(config, tokenizer):
    model_type = config['model']['type']

    if model_type != "GPT2":
        raise ValueError(f"Unsupported model type: {model_type}")

    # GPT-2 branch with optional LoRA / QLoRA
    use_pretrained = config['model'].get('use_pretrained', False)
    pretrained_model = config['model'].get('pretrained_model', 'gpt2-large')
    use_lora = config['model'].get('use_lora', False)
    use_qlora = config['model'].get('use_qlora', False)

    if use_pretrained:
        print(f"Loading pretrained GPT-2 model: {pretrained_model}")

        # QLoRA path
        if use_qlora:
            print("Using QLoRA (4-bit quantization + LoRA)")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = GPT2LMHeadModel.from_pretrained(
                    pretrained_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                model = prepare_model_for_kbit_training(model)
            except Exception as exc:
                print(f"Warning: QLoRA initialization failed ({exc}). Falling back to regular LoRA.")
                use_qlora = False
                use_lora = True
                model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        else:
            model = GPT2LMHeadModel.from_pretrained(pretrained_model)

        model.config.pad_token_id = tokenizer.pad_token_id

        if use_lora or use_qlora:
            print("Applying LoRA adapter (Unsloth Style)...")
            lora_config = LoraConfig(
                r=config['model'].get('lora_r', 16),
                lora_alpha=config['model'].get('lora_alpha', 32),
                target_modules=config['model'].get('lora_target_modules', ["c_attn", "c_proj", "c_fc"]),
                lora_dropout=config['model'].get('lora_dropout', 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                # [Modification 1] Pass modules_to_save here, PEFT will automatically unfreeze them and make them trainable
                modules_to_save=config['model'].get('modules_to_save', None) 
            )
            model = get_peft_model(model, lora_config)
            
            # Print trainable parameters. You will find that the parameter count increased (because wte was added), this is normal
            model.print_trainable_parameters()
        else:
            print("Using standard full fine-tuning (no LoRA).")
            
        return model

    # Randomly initialized GPT-2
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=2 * tokenizer.model_max_length,
        n_embd=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    return GPT2LMHeadModel(model_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./gpt-705M.yaml", help="Configuration file path")
    # ... (args definitions same as before) ...
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--model_name", type=str, default=None, help="Model name override")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA training")
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA training")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint folder to resume training from")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.model_name is not None:
        config['model']['name'] = args.model_name
    if args.use_lora:
        config['model']['use_lora'] = True
    if args.use_qlora:
        config['model']['use_qlora'] = True

    seq_length = config['data']['seq_length']

    # Use GPT-2's original tokenizer
    use_pretrained = config['model'].get('use_pretrained', False)
    pretrained_model = config['model'].get('pretrained_model', 'gpt2-large')

    if use_pretrained:
        print(f"Loading GPT-2 tokenizer from: {pretrained_model}")
        tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer_path = config['data']['tokenizer_path']
        tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "<pad>"

    train_dataset = CustomDataset(
        config['data']['train_path'],
        seq_length,
        tokenizer=tokenizer,
        random_chunk=True,
    )
    full_eval_dataset = CustomDataset(
        config['data']['eval_path'],
        seq_length,
        tokenizer=tokenizer,
        offset=0,
    )

    seed(2023)
    requested_eval = config['data']['eval_samples']
    available_eval = len(full_eval_dataset)
    eval_sample_size = min(requested_eval, available_eval)
    eval_indices = sample(range(available_eval), eval_sample_size)
    eval_dataset = Subset(full_eval_dataset, eval_indices)

    tokenizer.model_max_length = seq_length
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = build_model(config, tokenizer)
    
    # Explicitly update config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    output_dir = Path(config['logging']['output_dir']) / config['model']['name']
    output_dir.mkdir(parents=True, exist_ok=True)

    accumulation_steps = config['training']['gradient_accumulation_steps']
    per_device_bsz = config['training']['batch_size'] // accumulation_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        num_train_epochs=config['training']['num_epochs'],
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        save_total_limit=1,
        warmup_steps=config['training']['warmup_steps'],
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        logging_steps=20,
        report_to=[],
        fp16=config['training']['fp16'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        torch_compile=config['training'].get('torch_compile', False),
    )

    # === [Modification 2] Create optimizer supporting differential learning rates ===
    optimizer = None
    if config['model'].get('use_lora', False):
        embedding_lr = float(config['training'].get('embedding_lr', 1e-5)) # Default to a small value in case it's not set in yaml
        base_lr = float(config['training']['lr'])
        weight_decay = 0.01 # Default weight decay

        print(f"Applying differential Learning Rates: Base={base_lr}, Embeddings={embedding_lr}")

        # Separate parameter groups
        # 1. Embeddings (wte, lm_head): Use very small embedding_lr
        # 2. Others (LoRA layers): Use normal base_lr
        
        # Filter out parameters that do not require gradients
        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        
        # Define groups
        embedding_params = [p for n, p in model.named_parameters() if ("wte" in n or "lm_head" in n) and p.requires_grad]
        lora_params = [p for n, p in model.named_parameters() if ("wte" not in n and "lm_head" not in n) and p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": embedding_params,
                "lr": embedding_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": lora_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        ]
        
        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Pass custom optimizer
    # Pass (optimizer, None) to let Trainer automatically create Scheduler for us
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None) if optimizer is not None else (None, None)
    )

    # ... (Checkpoint loading logic same as before) ...
    resume_checkpoint = None
    if args.resume_path:
        if Path(args.resume_path).is_dir():
            resume_checkpoint = args.resume_path
        elif (output_dir / args.resume_path).is_dir():
            resume_checkpoint = str(output_dir / args.resume_path)
        
        if resume_checkpoint:
            print(f"✅ Found checkpoint. Resuming training from: {resume_checkpoint}")
        else:
            print(f"⚠️ Checkpoint folder not found at {args.resume_path}. Starting training from scratch.")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save model
    if config['model'].get('use_lora', False) or config['model'].get('use_qlora', False):
        model.save_pretrained(output_dir)
        print(f"LoRA adapter (including embeddings) saved to {output_dir}")
    else:
        trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()