import os
# Use environment TMPDIR for all caches and outputs
scratch = os.environ.get('TMPDIR', '/tmp')
hf_cache = os.path.join(scratch, 'hf_cache')
pip_cache = os.path.join(scratch, 'pip_cache')
wandb_cache = os.path.join(scratch, 'wandb_cache')
torch_compile_cache = os.path.join(scratch, 'torch_compile_cache')

# Set HuggingFace, pip, and output directories to local temp
# os.environ['HF_HOME'] = hf_cache
os.environ['TRANSFORMERS_CACHE'] = hf_cache
os.environ['HUGGINGFACE_HUB_CACHE'] = hf_cache
os.environ['PIP_CACHE_DIR'] = pip_cache
os.environ['WANDB_DIR'] = wandb_cache
os.environ['TORCH_COMPILE_CACHE_DIR'] = torch_compile_cache

os.makedirs(hf_cache, exist_ok=True)
os.makedirs(pip_cache, exist_ok=True)
os.makedirs(wandb_cache, exist_ok=True)
os.makedirs(torch_compile_cache, exist_ok=True)
print(f"[INFO] Using HF and pip cache at {hf_cache} and {pip_cache} and {wandb_cache} and {torch_compile_cache}.")

import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from interactive.grpotrainer import CustomGRPOTrainer
from interactive.reward import Llama3RewardModel
from interactive.datasetbuilder import TinyStoriesDatasetBuilder, DatasetCombiner
from interactive.ppoconfig import CustomPPOConfig
from interactive.utils import load_yaml_config

def main(grpo_cfg, teacher_cfg):
    # 1. Setup Randomness
    seed = teacher_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Config & Dataset
    grpo_config = CustomPPOConfig(
        model_name=grpo_cfg["model_name"],
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-6)),
        log_with=grpo_cfg.get("log_with", None),
        batch_size=grpo_cfg.get("batch_size", 4),
        mini_batch_size=grpo_cfg.get("mini_batch_size", 4),
        output_min_length=grpo_cfg.get("output_min_length", 64),
        output_max_length=grpo_cfg.get("output_max_length", 128),
    )

    builder = TinyStoriesDatasetBuilder(grpo_config)
    dataset = DatasetCombiner([builder])
    dataset.set_token_limit(token_limit=grpo_cfg.get("token_limit", 10000000))
    dataset = dataset.load()

    # 3. Models (Standard CausalLM)
    print(f"Loading model: {grpo_cfg['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(grpo_cfg["model_name"])
    ref_model = AutoModelForCausalLM.from_pretrained(grpo_cfg["model_name"])
    ref_model.eval()

    tokenizer = builder.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token 

    # 4. Reward
    reward_model = Llama3RewardModel(config=teacher_cfg)

    # 5. Trainer
    trainer = CustomGRPOTrainer(
        num_generations=grpo_cfg.get("num_generations", 4),
        beta=grpo_cfg.get("beta", 0.04),
        config=grpo_config, 
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=reward_model,
        word_budget=grpo_cfg.get("token_limit", 10000000),
        hf_org=grpo_cfg.get("hf_org", "llm-slice"),
        wandb_project="grpo-training",
        push_to_hub=True,
    )

    # 6. Run
    trainer.run_training_loop()

if __name__ == "__main__":
    grpo_cfg = load_yaml_config("config/grpo.yaml")
    teacher_cfg = load_yaml_config("config/teacher.yaml")
    main(grpo_cfg, teacher_cfg)