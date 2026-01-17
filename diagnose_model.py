#!/usr/bin/env python3
"""
Diagnose merged model issues
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("Model Diagnostics")
print("=" * 60)

model_path = "/home/inanna/dev/gemma/merged_model_fp16"

print(f"\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"\n🔍 Tokenizer info:")
print(f"   Vocab size: {len(tokenizer)}")
print(f"   PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"   EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"   BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")

print(f"\n📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"✅ Model loaded")
print(f"   Device: {model.device}")
print(f"   Dtype: {model.dtype}")

# Simple test
test_prompt = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"

print(f"\n🧪 Testing generation...")
print(f"Prompt: {test_prompt}")

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
print(f"\n📊 Input tokens:")
print(f"   IDs: {inputs['input_ids'][0].tolist()[:20]}")
print(f"   Decoded: {tokenizer.decode(inputs['input_ids'][0][:20])}")

print(f"\n🔄 Generating with different configs...")

# Test 1: Simple greedy
print(f"\n1. Greedy decoding:")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )
print(f"   Output IDs: {outputs[0].tolist()[-20:]}")
print(f"   Decoded: {tokenizer.decode(outputs[0][-20:])}")

# Test 2: With explicit eos
print(f"\n2. With EOS token:")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id  # Use EOS as pad
    )
print(f"   Output IDs: {outputs[0].tolist()[-20:]}")
print(f"   Decoded: {tokenizer.decode(outputs[0][-20:])}")

# Test 3: Check if model can forward pass
print(f"\n3. Raw forward pass test:")
with torch.no_grad():
    logits = model(**inputs).logits
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits mean: {logits.mean().item():.4f}")
    print(f"   Logits std: {logits.std().item():.4f}")
    print(f"   Logits contains NaN: {torch.isnan(logits).any().item()}")
    print(f"   Logits contains Inf: {torch.isinf(logits).any().item()}")

    # Get top predicted token
    next_token = logits[0, -1].argmax().item()
    print(f"   Next token prediction: {next_token} -> '{tokenizer.decode([next_token])}'")

print("\n" + "=" * 60)
