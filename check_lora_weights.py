#!/usr/bin/env python3
"""
Check if LoRA adapter weights contain NaN
"""

import torch
import os
from safetensors import safe_open

ADAPTER_PATH = "/home/inanna/dev/gemma/checkpoints/final_lora"

print("=" * 60)
print("Checking LoRA Adapter Weights")
print("=" * 60)

# Find safetensors file
adapter_file = None
for file in os.listdir(ADAPTER_PATH):
    if file.endswith('.safetensors') or file.endswith('.bin'):
        adapter_file = os.path.join(ADAPTER_PATH, file)
        break

if not adapter_file:
    print("❌ No adapter weight file found!")
    exit(1)

print(f"\n📂 Checking: {adapter_file}")

if adapter_file.endswith('.safetensors'):
    with safe_open(adapter_file, framework="pt", device="cpu") as f:
        print(f"\n🔍 Tensors in adapter:")
        for key in f.keys():
            tensor = f.get_tensor(key)
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            print(f"   {key}:")
            print(f"      Shape: {tensor.shape}")
            print(f"      Mean: {tensor.mean().item():.6f}")
            print(f"      NaN: {has_nan}, Inf: {has_inf}")

            if has_nan or has_inf:
                print(f"      ❌ PROBLEM: Contains NaN or Inf!")
else:
    # .bin file
    weights = torch.load(adapter_file, map_location="cpu")
    print(f"\n🔍 Checking {len(weights)} tensors...")
    for key, tensor in weights.items():
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            print(f"   ❌ {key}: NaN={has_nan}, Inf={has_inf}")

print("\n" + "=" * 60)
