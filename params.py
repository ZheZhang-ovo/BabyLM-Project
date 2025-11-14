from transformers import AutoModelForCausalLM
import torch, os

model_path = "/home/zhezhang/BabyLM/models/GPT2-Large-BabyLM"

print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
nontrainable_params = total_params - trainable_params

print("\n===== Parameter Count =====")
print(f"Total parameters       : {total_params:,}")
print(f"Trainable parameters   : {trainable_params:,}")
print(f"Non-trainable parameters: {nontrainable_params:,}")

# Estimated size
bytes_total = total_params * 4   # float32 assumed
print(f"\nApprox total size (float32): {bytes_total / (1024**3):.2f} GB")

# official GPT-2 Large number
official_gpt2_large_params = 774_030_080
print(f"\nGPT-2 Large official parameter count: {official_gpt2_large_params:,}")
print(f"Difference: {total_params - official_gpt2_large_params:,}")
