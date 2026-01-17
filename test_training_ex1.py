#!/usr/bin/env python3
"""Test model on Training Example 1 (which it saw during training)"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

MODEL_PATH = "./training-runs/lsp/2026-01-17/merged_model_fp32"

print("Testing model on Training Example 1 (semantic example)")
print("=" * 80)

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)

# Load exact prompt from training
with open('/tmp/test_prompt.txt') as f:
    prompt = f.read()

print("Topic: How do I send as a response?")
print("Expected: Parse,Ollama")
print()

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

# Parse
match = re.search(r'selected:<escape>([^<]*)<escape>', result)
selected = match.group(1).strip() if match else None

print(f"Model output: {result[:150]}")
print()
print(f"Parsed: {selected}")
print(f"Expected: Parse,Ollama")
print()
if selected == 'Parse,Ollama':
    print("✅ PASS - Model remembered this training example!")
else:
    print("❌ FAIL - Model forgot even training examples")
