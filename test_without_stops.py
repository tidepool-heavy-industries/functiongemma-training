#!/usr/bin/env python3
"""
Test generation without stop tokens
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

print("=" * 80)
print("Testing Generation Without Stop Tokens")
print("=" * 80)

# Load test dataset
test_dataset = load_from_disk("/home/inanna/dev/gemma/data/mobile_actions_curated_test")
example = test_dataset[0]

# Extract prompt
full_text = example['text']
parts = full_text.split('<start_of_turn>model\n')
if len(parts) >= 2:
    prompt = '<start_of_turn>model\n'.join(parts[:-1]) + '<start_of_turn>model\n'
    expected = parts[-1].replace('<end_of_turn>', '').strip()
else:
    prompt = full_text
    expected = "N/A"

print("\nExpected output:")
print(expected[:200])

# Load fine-tuned model
print("\n📥 Loading fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/inanna/dev/gemma/merged_model_fp32",
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/home/inanna/dev/gemma/merged_model_fp32")
print("✅ Model loaded")

# Check generation config
print("\n" + "=" * 80)
print("Generation Config")
print("=" * 80)
if hasattr(model, 'generation_config'):
    print(f"EOS token ID: {model.generation_config.eos_token_id}")
    if hasattr(model.generation_config, 'stop_strings'):
        print(f"Stop strings: {model.generation_config.stop_strings}")
print(f"Tokenizer EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"Tokenizer PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# Try encoding the stop tokens to see their IDs
test_strings = ["<end_of_turn>", "<start_of_turn>", "<start_function_call>", "<end_function_call>"]
print("\nToken IDs:")
for s in test_strings:
    ids = tokenizer.encode(s, add_special_tokens=False)
    print(f"  {s}: {ids}")

print("\n" + "=" * 80)
print("Generation WITHOUT stop tokens")
print("=" * 80)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        eos_token_id=None,  # Disable EOS stopping
        pad_token_id=tokenizer.pad_token_id
    )

generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
print(f"\nGenerated ({len(generated)} chars):")
print(generated[:600])
if len(generated) > 600:
    print("\n... [truncated]")

# Check for function call markers
has_start_call = "<start_function_call>" in generated
has_call_syntax = "call:" in generated
has_end_call = "<end_function_call>" in generated
print(f"\n✓ Has <start_function_call>: {has_start_call}")
print(f"✓ Has call: syntax: {has_call_syntax}")
print(f"✓ Has <end_function_call>: {has_end_call}")

print("\n" + "=" * 80)
