#!/usr/bin/env python3
"""
Merge LoRA adapters using PEFT's native merge (not Unsloth)
This should avoid NaN issues from Unsloth's merge
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

print("=" * 60)
print("LoRA Merge (PEFT Native Method)")
print("=" * 60)

ADAPTER_PATH = "/home/inanna/dev/gemma/checkpoints/final_lora"
BASE_MODEL = "google/functiongemma-270m-it"
OUTPUT_PATH = "/home/inanna/dev/gemma/merged_model_fp16_peft"

print(f"\n📥 Loading base model: {BASE_MODEL}")

# Load base model in FP16 (NOT 4-bit for merge)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(f"✅ Base model loaded in FP16")

print(f"\n📥 Loading LoRA adapters from: {ADAPTER_PATH}")

# Load PEFT model (adapters on top of base)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print(f"✅ LoRA adapters loaded")

print(f"\n🔧 Merging adapters into base model...")

# Merge and unload - this is PEFT's native merge
model = model.merge_and_unload()

print(f"✅ Merge complete")

print(f"\n🧪 Testing merged model (quick sanity check)...")

# Quick test
test_input = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits mean: {logits.mean().item():.4f}")
    print(f"   Logits contains NaN: {torch.isnan(logits).any().item()}")
    print(f"   Logits contains Inf: {torch.isinf(logits).any().item()}")

    if torch.isnan(logits).any():
        print(f"❌ ERROR: Model still contains NaN values!")
        print(f"   The LoRA merge failed. This might be a training issue.")
        exit(1)
    else:
        print(f"✅ Logits look healthy!")

print(f"\n💾 Saving merged model to: {OUTPUT_PATH}")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Save with PEFT's save method
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"✅ Model saved")

# Check file size
import subprocess
result = subprocess.run(['du', '-sh', OUTPUT_PATH], capture_output=True, text=True)
if result.returncode == 0:
    size = result.stdout.split()[0]
    print(f"   Total size: {size}")

print("\n" + "=" * 60)
print("✅ PEFT Merge Complete!")
print("=" * 60)
print(f"📍 Merged model: {OUTPUT_PATH}")
print(f"\nNext steps:")
print(f"1. Test: python test_merged_model.py")
print(f"   (Update path to: {OUTPUT_PATH})")
print(f"2. Convert to GGUF")
print("=" * 60)
