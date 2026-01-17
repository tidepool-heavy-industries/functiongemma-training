#!/usr/bin/env python3
"""
Merge LoRA adapters for conversation tree model
Based on merge_model_fp32.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL = "google/functiongemma-270m-it"
ADAPTER_PATH = "./training-runs/conversation/2026-01-17/checkpoints/final_lora"
OUTPUT_PATH = "./training-runs/conversation/2026-01-17/merged_model_fp32"

print("=" * 80)
print("Merging LoRA Adapters for Conversation Tree Model")
print("=" * 80)

print(f"\n📥 Loading base model in FP32...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,  # CRITICAL: Must use FP32 for Gemma3
    device_map="auto",
    trust_remote_code=True
)

print(f"\n📥 Loading LoRA adapters from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print(f"\n🔀 Merging adapters...")
model = model.merge_and_unload()

print(f"\n💾 Saving merged model to: {OUTPUT_PATH}")
os.makedirs(OUTPUT_PATH, exist_ok=True)
model.save_pretrained(OUTPUT_PATH)

print(f"\n💾 Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, fix_mistral_regex=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\n" + "=" * 80)
print("✅ Merge Complete!")
print("=" * 80)
print(f"\nMerged model saved to: {OUTPUT_PATH}")
print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f} GB")
print("\nNext: python test_conversation_model.py")
print("=" * 80)
