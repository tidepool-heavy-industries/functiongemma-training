#!/usr/bin/env python3
"""
Quick test of merged fine-tuned model
Tests function calling on a few examples
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("Testing Merged Fine-Tuned Model")
print("=" * 60)

# Load model
model_path = "/home/inanna/dev/gemma/merged_model_fp16"
print(f"\n📥 Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"✅ Model loaded")
print(f"   Device: {model.device}")
print(f"   Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Test prompts
test_cases = [
    {
        "functions": """<start_function_declaration>show_map
Shows a location on the map.
Parameters:
  query (string): The location to search for. (required)
<end_function_declaration>

<start_function_declaration>turn_on_flashlight
Turns the flashlight on.
<end_function_declaration>""",
        "query": "Show me the Eiffel Tower on the map",
        "expected": "show_map"
    },
    {
        "functions": """<start_function_declaration>create_calendar_event
Creates a new calendar event.
Parameters:
  title (string): The title of the event. (required)
  datetime (string): The date and time in YYYY-MM-DDTHH:MM:SS format. (required)
<end_function_declaration>

<start_function_declaration>send_email
Sends an email.
Parameters:
  to (string): Email recipient. (required)
  subject (string): Email subject. (required)
  body (string): Email body.
<end_function_declaration>""",
        "query": "Schedule a meeting called Team Sync for tomorrow at 2 PM",
        "expected": "create_calendar_event"
    },
    {
        "functions": """<start_function_declaration>turn_on_flashlight
Turns the flashlight on.
<end_function_declaration>

<start_function_declaration>turn_off_flashlight
Turns the flashlight off.
<end_function_declaration>""",
        "query": "Turn on my flashlight please",
        "expected": "turn_on_flashlight"
    }
]

print("\n" + "=" * 60)
print("Running Test Cases")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'=' * 60}")
    print(f"TEST {i}/3")
    print(f"{'=' * 60}")
    print(f"Query: {test['query']}")
    print(f"Expected function: {test['expected']}")

    # Build prompt
    prompt = f"""<start_of_turn>developer
You are an expert function calling AI assistant. Current date: 2026-01-16.
You have access to the following functions:

{test['functions']}

<end_of_turn>
<start_of_turn>user
{test['query']}<end_of_turn>
<start_of_turn>model
"""

    # Generate (use greedy decoding to avoid sampling issues)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the model's response
    if "<start_of_turn>model" in response:
        model_response = response.split("<start_of_turn>model")[-1].split("<end_of_turn>")[0].strip()
    else:
        model_response = response

    print(f"\nGenerated response:")
    print(f"  {model_response[:300]}...")

    # Check if expected function is in response
    if test['expected'] in model_response:
        print(f"✅ PASS - Contains expected function: {test['expected']}")
    else:
        print(f"❌ FAIL - Missing expected function: {test['expected']}")

    # Check format
    if "<start_function_call>" in model_response and "<end_function_call>" in model_response:
        print(f"✅ PASS - Proper function call format")
    else:
        print(f"❌ FAIL - Missing function call tags")

print("\n" + "=" * 60)
print("✅ Testing Complete!")
print("=" * 60)
print("\nIf tests look good, proceed with GGUF conversion:")
print("  1. Clone llama.cpp: cd /home/inanna/dev && git clone https://github.com/ggerganov/llama.cpp.git")
print("  2. Build: cd llama.cpp && make")
print("  3. Convert: python convert_hf_to_gguf.py /home/inanna/dev/gemma/merged_model_fp16")
print("=" * 60)
