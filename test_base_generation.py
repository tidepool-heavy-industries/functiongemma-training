#!/usr/bin/env python3
"""
Test base FunctionGemma model generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

print("=" * 80)
print("Testing Base FunctionGemma Model")
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
print(expected[:300])

# Load BASE model (not fine-tuned)
print("\n📥 Loading base FunctionGemma model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/functiongemma-270m-it",
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
print("✅ Base model loaded")

print("\n" + "=" * 80)
print("BASE MODEL Generation (Greedy)")
print("=" * 80)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
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
has_escape = "<escape>" in generated

print(f"\n✓ Has <start_function_call>: {has_start_call}")
print(f"✓ Has call: syntax: {has_call_syntax}")
print(f"✓ Has <end_function_call>: {has_end_call}")
print(f"✓ Has <escape> syntax: {has_escape}")

if has_call_syntax:
    # Extract first function call
    if "call:" in generated:
        call_start = generated.find("call:")
        call_end = generated.find("<end_function_call>", call_start)
        if call_end > call_start:
            function_call = generated[call_start:call_end]
            print(f"\nFirst function call:")
            print(f"  {function_call[:200]}")

print("\n" + "=" * 80)
