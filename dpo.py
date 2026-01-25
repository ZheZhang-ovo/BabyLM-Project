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
from interactive.dpotrainer import CustomDPOTrainer
from interactive.reward import Llama3RewardModel
from interactive.datasetbuilder import TinyStoriesDatasetBuilder, DatasetCombiner
from interactive.ppoconfig import CustomPPOConfig
from interactive.utils import load_yaml_config

def main(dpo_cfg, teacher_cfg):
    # 1. Setup Randomness
    seed = teacher_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Config & Dataset
    dpo_config = CustomPPOConfig(
        model_name=dpo_cfg["model_name"],
        learning_rate=float(dpo_cfg.get("learning_rate", 1e-6)),
        log_with=dpo_cfg.get("log_with", None),
        batch_size=dpo_cfg.get("batch_size", 4),
        mini_batch_size=dpo_cfg.get("mini_batch_size", 4),
        output_min_length=dpo_cfg.get("output_min_length", 64),
        output_max_length=dpo_cfg.get("output_max_length", 128),
    )

    builder = TinyStoriesDatasetBuilder(dpo_config)
    dataset = DatasetCombiner([builder])
    dataset.set_token_limit(token_limit=dpo_cfg.get("token_limit", 10000000))
    dataset = dataset.load()

    # Models
    print(f"Loading model: {dpo_cfg['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(dpo_cfg["model_name"])
    ref_model = AutoModelForCausalLM.from_pretrained(dpo_cfg["model_name"])
    ref_model.eval()

    tokenizer = builder.tokenizer
    tokenizer.padding_side = "left" # Critical for generation
    tokenizer.pad_token = tokenizer.eos_token

    # Reward
    reward_model = Llama3RewardModel(config=teacher_cfg)

    # Trainer
    trainer = CustomDPOTrainer(
        beta=dpo_cfg.get("beta", 0.1),
        config=dpo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_fn=reward_model,
        word_budget=dpo_cfg.get("token_limit", 1000000),
        hf_org=dpo_cfg.get("hf_org", "llm-slice"),
        wandb_project="dpo-training",
        push_to_hub=True,
    )

    trainer.run_training_loop()

if __name__ == "__main__":
    dpo_cfg = load_yaml_config("config/dpo.yaml")
    teacher_cfg = load_yaml_config("config/teacher.yaml")
    main(dpo_cfg, teacher_cfg)