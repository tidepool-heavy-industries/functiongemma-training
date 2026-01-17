#!/usr/bin/env python3
"""
Debug why the fine-tuned model isn't generating function calls
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

print("=" * 80)
print("Debugging Fine-Tuned Model Generation")
print("=" * 80)

# Load test dataset
print("\n📥 Loading test dataset...")
test_dataset = load_from_disk("/home/inanna/dev/gemma/data/mobile_actions_curated_test")
example = test_dataset[0]

# Show full example
print("\n" + "=" * 80)
print("FULL EXAMPLE TEXT")
print("=" * 80)
full_text = example['text']
print(full_text[:1000])  # First 1000 chars
print("\n... [truncated]")

# Extract prompt
parts = full_text.split('<start_of_turn>model\n')
if len(parts) >= 2:
    prompt = '<start_of_turn>model\n'.join(parts[:-1]) + '<start_of_turn>model\n'
    expected = parts[-1].replace('<end_of_turn>', '').strip()
else:
    prompt = full_text
    expected = "N/A"

print("\n" + "=" * 80)
print("EXTRACTED PROMPT")
print("=" * 80)
print(prompt[:1000])
print("\n... [truncated]")

print("\n" + "=" * 80)
print("EXPECTED OUTPUT")
print("=" * 80)
print(expected[:500])

# Load fine-tuned model
print("\n" + "=" * 80)
print("LOADING FINE-TUNED MODEL")
print("=" * 80)
model = AutoModelForCausalLM.from_pretrained(
    "/home/inanna/dev/gemma/merged_model_fp32",
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/home/inanna/dev/gemma/merged_model_fp32")
print("✅ Model loaded")

# Try different generation strategies
print("\n" + "=" * 80)
print("TESTING DIFFERENT GENERATION STRATEGIES")
print("=" * 80)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

strategies = [
    {"name": "Greedy (temp=0)", "params": {"do_sample": False, "max_new_tokens": 200}},
    {"name": "Low temp (0.1)", "params": {"do_sample": True, "temperature": 0.1, "max_new_tokens": 200}},
    {"name": "Med temp (0.7)", "params": {"do_sample": True, "temperature": 0.7, "max_new_tokens": 200}},
    {"name": "With top_p", "params": {"do_sample": True, "temperature": 0.3, "top_p": 0.9, "max_new_tokens": 200}},
]

for strategy in strategies:
    print(f"\n{'─' * 80}")
    print(f"Strategy: {strategy['name']}")
    print(f"{'─' * 80}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **strategy['params'],
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    print(f"Generated ({len(generated)} chars):")
    print(generated[:300])
    if len(generated) > 300:
        print("... [truncated]")

    # Check for function call markers
    has_start_call = "<start_function_call>" in generated
    has_call_syntax = "call:" in generated
    print(f"\n✓ Has <start_function_call>: {has_start_call}")
    print(f"✓ Has call: syntax: {has_call_syntax}")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
