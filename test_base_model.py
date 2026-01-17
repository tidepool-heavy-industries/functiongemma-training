#!/usr/bin/env python3
"""
Test if base model works (before any LoRA merge)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("Testing Base FunctionGemma Model")
print("=" * 60)

BASE_MODEL = "google/functiongemma-270m-it"

print(f"\n📥 Loading base model in FP16...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

test_input = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(test_input, return_tensors="pt").to(model_fp16.device)

print(f"\n🧪 Testing FP16 base model...")
with torch.no_grad():
    logits = model_fp16(**inputs).logits
    print(f"   Logits mean: {logits.mean().item():.4f}")
    print(f"   Logits contains NaN: {torch.isnan(logits).any().item()}")

if torch.isnan(logits).any():
    print(f"❌ Base model in FP16 has NaN - this is a model issue")
else:
    print(f"✅ Base model in FP16 works!")

    # Try generation
    print(f"\n🔄 Generating with base model...")
    outputs = model_fp16.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"   Generated: {tokenizer.decode(outputs[0])}")

# Now test FP32
print(f"\n📥 Loading base model in FP32...")
model_fp32 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="auto"
)

print(f"\n🧪 Testing FP32 base model...")
with torch.no_grad():
    logits = model_fp32(**inputs).logits
    print(f"   Logits mean: {logits.mean().item():.4f}")
    print(f"   Logits contains NaN: {torch.isnan(logits).any().item()}")

print("\n" + "=" * 60)
