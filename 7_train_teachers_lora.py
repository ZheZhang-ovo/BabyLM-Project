from transformers import (
    GPT2Config, GPT2LMHeadModel,
    LlamaConfig, LlamaForCausalLM,
    GPTJConfig, GPTJForCausalLM,
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

from custom_dataset import CustomDataset


def build_model(config, tokenizer):
    model_type = config['model']['type']

    if model_type == "Llama":
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=2 * tokenizer.model_max_length,
            hidden_size=config['model']['hidden_size'],
            intermediate_size=config['model']['intermediate_size'],
            num_hidden_layers=config['model']['n_layer'],
            num_attention_heads=config['model']['n_head'],
            tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        return LlamaForCausalLM(model_config)

    if model_type == "GPTJ":
        model_config = GPTJConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=2 * tokenizer.model_max_length,
            n_embd=config['model']['hidden_size'],
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        return GPTJForCausalLM(model_config)

    if model_type != "GPT2":
        raise ValueError(f"Unsupported model type: {model_type}")

    # GPT-2 branch with optional LoRA / QLoRA
    use_pretrained = config['model'].get('use_pretrained', False)
    pretrained_model = config['model'].get('pretrained_model', 'gpt2-large')
    use_lora = config['model'].get('use_lora', False)
    use_qlora = config['model'].get('use_qlora', False)

    if use_pretrained:
        print(f"Loading pretrained GPT-2 model: {pretrained_model}")

        # QLoRA path: 4-bit quantization + LoRA adapter
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

        # Using GPT-2 original tokenizer, vocab_size fully matches, no adjustment needed
        model.config.pad_token_id = tokenizer.pad_token_id

        if use_lora or use_qlora:
            lora_config = LoraConfig(
                r=config['model'].get('lora_r', 16),
                lora_alpha=config['model'].get('lora_alpha', 32),
                target_modules=config['model'].get('lora_target_modules', ["c_attn", "c_proj", "c_fc"]),
                lora_dropout=config['model'].get('lora_dropout', 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
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
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--model_name", type=str, default=None, help="Model name override")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.model_name is not None:
        config['model']['name'] = args.model_name

    seq_length = config['data']['seq_length']

    # Use GPT-2's original tokenizer (fully compatible with pre-trained models)
    use_pretrained = config['model'].get('use_pretrained', False)
    pretrained_model = config['model'].get('pretrained_model', 'gpt2-large')

    if use_pretrained:
        print(f"Loading GPT-2 tokenizer from: {pretrained_model}")
        tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model)
        # GPT-2 doesn't have pad_token, use eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # If not using pre-trained model, use custom tokenizer
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
    if eval_sample_size < requested_eval:
        print(
            f"Warning: requested {requested_eval} eval samples but only {available_eval} available. "
            f"Using {eval_sample_size} instead."
        )
    eval_indices = sample(range(available_eval), eval_sample_size)
    eval_dataset = Subset(full_eval_dataset, eval_indices)

    tokenizer.model_max_length = seq_length

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = build_model(config, tokenizer)
    print(f"Trainable parameters: {model.num_parameters()} (total)")

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

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    if config['model'].get('use_lora', False) or config['model'].get('use_qlora', False):
        model.save_pretrained(output_dir)
        print(f"LoRA/QLoRA adapter saved to {output_dir}")
    else:
        trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
