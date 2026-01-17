#!/usr/bin/env python3
"""
Merge LoRA adapters in FP32 (FP16 doesn't work for Gemma3)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

print("=" * 60)
print("LoRA Merge in FP32 (Gemma3 Compatible)")
print("=" * 60)

ADAPTER_PATH = "/home/inanna/dev/gemma/checkpoints/final_lora"
BASE_MODEL = "google/functiongemma-270m-it"
OUTPUT_PATH = "/home/inanna/dev/gemma/merged_model_fp32"

print(f"\n📥 Loading base model in FP32...")
print(f"   (FP16 causes NaN for Gemma3)")

# Load in FP32 - this is what training used
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,  # CRITICAL: Must use FP32
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(f"✅ Base model loaded in FP32")
print(f"   Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print(f"\n📥 Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print(f"✅ LoRA adapters loaded")

print(f"\n🔧 Merging...")
model = model.merge_and_unload()

print(f"✅ Merge complete")

print(f"\n🧪 Testing merged model...")
test_input = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits
    print(f"   Logits mean: {logits.mean().item():.4f}")
    print(f"   Logits NaN: {torch.isnan(logits).any().item()}")
    print(f"   Logits Inf: {torch.isinf(logits).any().item()}")

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"❌ ERROR: Merge failed!")
        exit(1)

    # Try generating
    print(f"\n🔄 Testing generation...")
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])
    print(f"   Generated: {generated}")

    if "<pad>" in generated and len(generated.replace("<pad>", "").strip()) == 0:
        print(f"   ⚠️ WARNING: Only generating padding tokens")
    else:
        print(f"   ✅ Generation works!")

print(f"\n💾 Saving to: {OUTPUT_PATH}")
os.makedirs(OUTPUT_PATH, exist_ok=True)

model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"✅ Model saved")

import subprocess
result = subprocess.run(['du', '-sh', OUTPUT_PATH], capture_output=True, text=True)
if result.returncode == 0:
    print(f"   Size: {result.stdout.split()[0]}")

print("\n" + "=" * 60)
print("✅ Merge Complete!")
print("=" * 60)
print(f"📍 Merged model: {OUTPUT_PATH}")
print(f"\nNOTE: Model is in FP32 (~1.1GB)")
print(f"GGUF conversion will quantize it to Q4_K_M (~150-200MB)")
print("=" * 60)
