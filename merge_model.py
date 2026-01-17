#!/usr/bin/env python3
"""
Merge LoRA Adapters into Base Model
Converts fine-tuned LoRA adapters back into a full FP16 model
"""

import torch
from unsloth import FastLanguageModel
import os

# ============================================================================
# Configuration
# ============================================================================

LORA_ADAPTER_PATH = "/home/inanna/dev/gemma/checkpoints/final_lora"
OUTPUT_PATH = "/home/inanna/dev/gemma/merged_model_fp16"
MAX_SEQ_LENGTH = 2048

# ============================================================================
# Main
# ============================================================================

print("=" * 60)
print("LoRA Adapter Merging")
print("=" * 60)

# Check if adapter path exists
if not os.path.exists(LORA_ADAPTER_PATH):
    print(f"❌ Error: LoRA adapter path not found: {LORA_ADAPTER_PATH}")
    print(f"\n💡 Make sure training has completed and adapters are saved.")
    exit(1)

print(f"📥 Loading model and LoRA adapters...")
print(f"   Adapter path: {LORA_ADAPTER_PATH}")

# Load the fine-tuned model with adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,  # Load in 4-bit first
)

print(f"✅ Model and adapters loaded")

# Prepare for inference (merges adapters)
print(f"\n🔧 Merging LoRA adapters into base model...")
model = FastLanguageModel.for_inference(model)

# Save merged model in FP16
print(f"\n💾 Saving merged model to {OUTPUT_PATH}...")
print(f"   Format: FP16 (full precision)")
print(f"   Expected size: ~550-600 MB")

model.save_pretrained_merged(
    OUTPUT_PATH,
    tokenizer,
    save_method="merged_16bit",  # Save as FP16
)

print(f"✅ Merged model saved successfully!")

# Verify model files
print(f"\n📂 Checking saved files...")
files = os.listdir(OUTPUT_PATH)
print(f"   Files: {', '.join(files)}")

# Check model size
import subprocess
result = subprocess.run(['du', '-sh', OUTPUT_PATH], capture_output=True, text=True)
if result.returncode == 0:
    size = result.stdout.split()[0]
    print(f"   Total size: {size}")

print("\n" + "=" * 60)
print("✅ Merge Complete!")
print("=" * 60)
print(f"📍 Merged model location: {OUTPUT_PATH}")
print(f"\nNext steps:")
print(f"1. Test merged model (optional): python test_merged_model.py")
print(f"2. Convert to GGUF: cd /home/inanna/dev/llama.cpp && python convert_hf_to_gguf.py {OUTPUT_PATH}")
print("=" * 60)
