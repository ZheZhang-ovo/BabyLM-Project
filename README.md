# Distill-and-Reinforce: Empowering BabyLM with Large Model Supervision

This project implements a comprehensive knowledge distillation pipeline for the BabyLM Challenge. The goal is to train efficient small language models by transferring knowledge from a fine-tuned GPT-2 Large teacher model to smaller student models.

## ✨ Key Features

- **Knowledge Distillation**: Transfer learning from GPT-2 Large (774M) to GPT-2 Small (124M) using KL Divergence and Cross-Entropy loss.
- **Reinforcement Learning (RLAIF)**: PPO (Proximal Policy Optimization) training pipeline with Llama-3 based reward modeling.
- **100M Dataset Support**: Scalable data processing pipelines handling both 10M and 100M BabyLM datasets.
- **Interactive Evaluation**: Integrated chatbot interface.
- **Flexible Training**: Supports Full Fine-tuning, LoRA, and QLoRA for teacher models adaptability to various hardware constraints.

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

*Note: For QLoRA, `bitsandbytes` is required.*

### 1. Data Preparation

Combine and split the raw BabyLM data (supports 10M or 100M versions):

```bash
# Processes files from train_10M/ or train_100M/
python combine_babylm.py
```
*Outputs to `corpus_split/` or `corpus_split_100M/` with 95/5 train/val split.*

### 2. Teacher Model Training

Fine-tune the teacher (GPT-2 Large) on the domain data. Choose based on your GPU memory:

| Method | Memory | Command |
|--------|--------|---------|
| **Full Fine-tuning** | ≥8GB | `python train_teachers.py --config gpt2-large-babylm.yaml` |
| **LoRA** | 4-8GB | `python train_teachers.py --config gpt2-large-babylm.yaml --use_lora` |
| **QLoRA** | <4GB | `python train_teachers.py --config gpt2-large-babylm.yaml --use_qlora` |

**(Optional) Merge LoRA Adapter**:
If you trained with LoRA/QLoRA, merge the adapter for efficient inference/distillation:
```bash
python merge_lora.py --lora_path ./models/GPT2-Large-BabyLM-100M-LoRA --output_path ./models/GPT2-Large-BabyLM-100M-Merged --base_model gpt2-large
```

### 3. Student Model Training

#### Option A: Standard Distillation (GPT-2 Small)
Train a randomly initialized GPT-2 Small using knowledge from your teacher.

```bash
python train_student.py
```

#### Option B: Baseline (No Distillation)
Train a GPT-2 Small from scratch using only Cross-Entropy loss for comparison.

```bash
python train_gpt2_small_ce.py
```

### 4. Reinforcement Learning (PPO)

Refine the student model using PPO with feedback from a Llama-3 based reward model.

```bash
python ppo.py
```

## 🧪 Evaluation & Analysis

### Zero-Shot Evaluation
We support the official BabyLM evaluation suite (BLiMP, EWoK, etc.).

```bash
cd eval
# Usage: ./eval_zero_shot.sh <Absolute Model Path> <Backend>
./eval_zero_shot.sh /abs/path/to/models/GPT2-Small-Distilled causal
```

### Interactive Demo
Test your trained models using the provided interface or visit our [LMseed Hugging Face Space](https://huggingface.co/LMseed).

## 📂 Project Structure

```
BabyLM/
├── corpus_split_100M/         # Processed training data
├── models/                    # Saved model checkpoints
├── eval/                      # Evaluation suite
├── interactive/               # RL & Chatbot components
├── 7_train_teachers.py        # Teacher training script
├── 8_train_student.py         # Student distillation script
├── train_gpt2_small_ce.py     # Baseline training
├── ppo.py                     # RL training entry point
└── gpt2-large-babylm.yaml     # Configuration file
```

## 🔬 Methodology Details

- **Distillation Loss**: `Loss = α * L_CE + (1-α) * L_KL`
    - `Temperature=2.0`, `Alpha=0.5`
- **Checkpointing**: Data-volume based (milestones at 1M, 10M, 100M words) rather than epoch-based, allowing for granular "scaling law" analysis.
- **Tokenizer**: Standard GPT-2 tokenizer (50k vocab) used consistently across all models.


## 🔗 References

- [BabyLM Challenge](https://babylm.github.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Knowledge Distillation (Hinton et al.)](https://arxiv.org/abs/1503.02531)
