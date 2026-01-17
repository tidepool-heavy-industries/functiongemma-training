#!/usr/bin/env python3
"""
Test FunctionGemma with CORRECT inference method using chat template
Based on Gemini's technical report Section 6.1
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

print("=" * 80)
print("Testing FunctionGemma with Proper Chat Template")
print("=" * 80)

# Load fine-tuned model
model_path = "/home/inanna/dev/gemma/merged_model_fp32"
print(f"\n📥 Loading fine-tuned model: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("✅ Model loaded")

# Define tools using proper JSON schema format (not raw text!)
tools = [
    {
        "type": "function",
        "function": {
            "name": "turn_on_flashlight",
            "description": "Turns the flashlight on.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Creates a new calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime": {
                        "type": "string",
                        "description": "The date and time of the event in the format YYYY-MM-DDTHH:MM:SS."
                    },
                    "title": {
                        "type": "string",
                        "description": "The title of the event."
                    }
                },
                "required": ["datetime", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_off_flashlight",
            "description": "Turns the flashlight off.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Test queries
test_queries = [
    "Please turn on the flashlight",
    "Schedule a calendar event titled 'Team Meeting' for November 20th, 2026 at 10:00 AM",
    "Turn on the flashlight and schedule a calendar event called 'Review Q4 Strategy Draft' for Friday, November 20th, 2026 at 10:00 AM"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST {i}/3")
    print(f"{'=' * 80}")
    print(f"Query: {query}")

    # Structure messages correctly
    messages = [
        {"role": "user", "content": query}
    ]

    # Apply chat template with tools parameter (THIS IS THE KEY!)
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

    print(f"\n🟢 Fine-tuned Model Output:")
    print(generated[:400])
    if len(generated) > 400:
        print("... [truncated]")

    # Check for function call syntax
    has_start_call = "<start_function_call>" in generated
    has_call_syntax = "call:" in generated
    has_end_call = "<end_function_call>" in generated

    print(f"\n✓ Has <start_function_call>: {has_start_call}")
    print(f"✓ Has call: syntax: {has_call_syntax}")
    print(f"✓ Has <end_function_call>: {has_end_call}")

    if has_call_syntax:
        print("✅ SUCCESS: Model generated proper function call!")
    else:
        print("❌ FAILURE: Missing call: syntax")

print("\n" + "=" * 80)
print("Testing Complete")
print("=" * 80)
