# BabyLM Knowledge Distillation

A knowledge distillation project for training small language models on the BabyLM dataset, transferring knowledge from a fine-tuned GPT-2 Large teacher model to a randomly initialized GPT-2 Small student model.

> **Project Period**: November 11-18, 2025  
> **Status**: Week 1 Implementation - Initial approach completed. The project is under active development and will be updated with improvements and refinements.

## 📊 Results

Our knowledge distillation approach achieves significant improvements over a randomly initialized baseline:

| Model | Loss | Perplexity (PPL) |
|-------|------|------------------|
| **Baseline** (Random Init + CE) | 3.777 | 43.69 |
| **Distilled Student** | 3.582 | 35.94 |
| **Improvement** | **-0.195** | **-7.75** (↓17.7%) |

The distilled student model shows a **17.7% reduction in perplexity** compared to the baseline, demonstrating the effectiveness of knowledge distillation.

## 🎯 Overview

This project implements a complete knowledge distillation pipeline:

1. **Teacher Model**: Fine-tune GPT-2 Large on BabyLM dataset
2. **Student Model**: Train a randomly initialized GPT-2 Small using knowledge distillation
3. **Baseline Comparison**: Train a randomly initialized GPT-2 Small with standard cross-entropy loss

### Architecture

- **Teacher**: GPT-2 Large (774M parameters)
- **Student**: GPT-2 Small (124M parameters)
- **Distillation Method**: KL divergence loss + student cross-entropy loss
- **Temperature**: 2.0
- **Alpha**: 0.5 (weighting between distillation and student loss)

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Training Pipeline

#### Step 1: Prepare Data

```bash
python combine_babylm.py
```

This merges all `.train` files from `train_10M/` directory and splits into:
- Training set: `corpus_split/train_babylm.txt` (95%)
- Validation set: `corpus_split/val_babylm.txt` (5%)

#### Step 2: Fine-tune Teacher Model

**Option A: Full Fine-tuning** (≥8GB GPU memory)

```bash
python 7_train_teachers.py --config gpt2-large-babylm.yaml
```

**Option B: LoRA/QLoRA Fine-tuning** (<8GB GPU memory)

```bash
python 7_train_teachers_lora.py --config gpt2-large-babylm.yaml
```

Enable LoRA/QLoRA in `gpt2-large-babylm.yaml`:
```yaml
model:
  use_lora: true      # or use_qlora: true for QLoRA
```

#### Step 3: Train Baseline Model (for comparison)

```bash
python train_gpt2_small_ce.py
```

This trains a randomly initialized GPT-2 Small using standard cross-entropy loss.

#### Step 4: Distill Student Model

```bash
python 8_train_student.py
```

This trains a randomly initialized GPT-2 Small using knowledge distillation from the teacher.

#### Step 5: Evaluate Models

```bash
python evaluate_student.py
```

This compares the baseline and distilled student models on the validation set.

## 📁 Project Structure

```
BabyLM/
├── train_10M/                    # BabyLM training data
│   ├── bnc_spoken.train
│   ├── childes.train
│   ├── gutenberg.train
│   ├── open_subtitles.train
│   ├── simple_wiki.train
│   └── switchboard.train
├── corpus_split/                  # Processed data
│   ├── train_babylm.txt
│   └── val_babylm.txt
├── models/                        # Trained models
│   ├── GPT2-Large-BabyLM/         # Fine-tuned teacher
│   ├── GPT2-Small-BabyLM-CE/      # Baseline student
│   └── GPT2-Small-Distilled/      # Distilled student
├── combine_babylm.py              # Data preparation
├── 6_tokenizer.py                # Custom tokenizer training (optional)
├── 7_train_teachers.py            # Teacher fine-tuning (full)
├── 7_train_teachers_lora.py      # Teacher fine-tuning (LoRA/QLoRA)
├── train_gpt2_small_ce.py        # Baseline student training
├── 8_train_student.py             # Student distillation
├── evaluate_student.py            # Model evaluation
├── evaluate_teacher.py            # Teacher model evaluation
├── custom_dataset.py              # PyTorch dataset class
├── gpt2-large-babylm.yaml        # Training configuration
└── README.md
```

## ⚙️ Configuration

### Teacher Model Configuration (`gpt2-large-babylm.yaml`)

```yaml
data:
  train_path: "./corpus_split/train_babylm.txt"
  eval_path: "./corpus_split/val_babylm.txt"
  seq_length: 128
  eval_samples: 8192

model:
  type: "GPT2"
  name: "GPT2-Large-BabyLM"
  use_pretrained: true
  pretrained_model: "gpt2-large"
  use_lora: false      # Set to true for LoRA
  use_qlora: false     # Set to true for QLoRA

training:
  lr: 2.5e-4
  batch_size: 128
  num_epochs: 4
  gradient_accumulation_steps: 16
  warmup_steps: 300
  fp16: True
```

### Student Model Configuration

**Baseline** (`train_gpt2_small_ce.py`):
- Learning rate: 2.5e-4
- Batch size: 32
- Epochs: 6
- Loss: Cross-entropy only

**Distilled** (`8_train_student.py`):
- Learning rate: 2.5e-4
- Batch size: 32
- Epochs: 6
- Temperature: 2.0
- Alpha: 0.5 (distillation loss weight)
- Loss: KL divergence + cross-entropy

## 🔬 Methodology

### Knowledge Distillation

The student model learns from the teacher's soft labels (logits) using:

```
Loss = α × L_CE(student) + (1-α) × L_KL(student, teacher)
```

Where:
- `L_CE`: Cross-entropy loss on ground truth labels
- `L_KL`: KL divergence between student and teacher logits (softened by temperature)
- `α = 0.5`: Weighting factor
- `Temperature = 2.0`: Softens the teacher's probability distribution

### Tokenizer

We use GPT-2's original tokenizer (50,257 vocabulary) for full compatibility with pre-trained models. This ensures:
- ✅ Perfect compatibility with GPT-2 pre-trained weights
- ✅ No vocabulary mapping needed
- ✅ Stable training

## 📈 Training Details

### Teacher Model
- **Base Model**: GPT-2 Large (pre-trained)
- **Fine-tuning**: Full fine-tuning or LoRA/QLoRA on BabyLM dataset
- **Output**: `models/GPT2-Large-BabyLM/`

### Baseline Student
- **Architecture**: GPT-2 Small (random initialization)
- **Training**: Standard cross-entropy loss
- **Output**: `models/GPT2-Small-BabyLM-CE/`

### Distilled Student
- **Architecture**: GPT-2 Small (random initialization)
- **Training**: Knowledge distillation from teacher
- **Output**: `models/GPT2-Small-Distilled/`

## 🖥️ System Requirements

### GPU Memory Recommendations
- **Full Fine-tuning**: ≥8GB (e.g., RTX 3070, A100)
- **LoRA**: 4-8GB (e.g., RTX 3060)
- **QLoRA**: <4GB (e.g., RTX 2060)

### Example SLURM Commands

```bash
# Request GPU
salloc -p gpua100 -t 4:00:00 --gres=gpu:1

# Enter interactive session
srun --pty bash

# Load modules
module load GCCcore/12.2.0
module load Python/3.10.8

# Activate environment
source ~/BabyLM/babylm/bin/activate
```

## 📝 Key Files

- `combine_babylm.py`: Merges and splits BabyLM data files while maintaining sentence coherence
- `6_tokenizer.py`: Trains a custom BPE tokenizer (optional, GPT-2 tokenizer is recommended)
- `7_train_teachers.py`: Full fine-tuning of teacher model
- `7_train_teachers_lora.py`: LoRA/QLoRA fine-tuning of teacher model
- `train_gpt2_small_ce.py`: Baseline student training (cross-entropy only)
- `8_train_student.py`: Student distillation training
- `evaluate_student.py`: Compares baseline vs. distilled models
- `evaluate_teacher.py`: Evaluates teacher model performance
- `custom_dataset.py`: PyTorch dataset for text data with caching support

## ⚠️ Notes

1. **LoRA/QLoRA**: When using parameter-efficient fine-tuning, only adapter weights are saved. For inference, load the base model + adapter.

2. **QLoRA Dependency**: Requires `bitsandbytes` package for 4-bit quantization.

3. **Memory Optimization**: If running out of memory, reduce `batch_size` and increase `gradient_accumulation_steps`.

4. **Tokenizer Consistency**: All models use the same GPT-2 tokenizer for consistency.

5. **Data Coherence**: The `combine_babylm.py` script maintains sentence order within each file to preserve dialogue and text coherence.

## 📊 Evaluation

The evaluation script (`evaluate_student.py`) compares:
- **Baseline**: Randomly initialized GPT-2 Small trained with cross-entropy
- **Distilled**: Randomly initialized GPT-2 Small trained with knowledge distillation

Metrics reported:
- Cross-entropy loss
- Perplexity (PPL)

## 📅 Implementation Status (Nov 11-18, 2024)

This is our initial implementation completed during Week 1 (November 11-18, 2024). The current version includes:

### ✅ Completed
- Basic knowledge distillation pipeline
- Teacher model fine-tuning (full and LoRA/QLoRA)
- Student model training with distillation
- Baseline comparison model
- Evaluation framework
- Data preparation with coherence preservation

### 📝 Notes
- Results are preliminary and subject to change
- Configuration parameters may be adjusted in future iterations
- Evaluation metrics and methodology may be refined

## 🔗 References

- [BabyLM Challenge](https://babylm.github.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531)
