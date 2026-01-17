#!/usr/bin/env python3
"""
Compare Base vs Fine-Tuned FunctionGemma using PROPER inference method
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import json

def extract_tools_from_text(text):
    """Extract tool definitions from raw training data format"""
    tools = []

    # Common tools in the dataset
    tool_definitions = {
        "turn_on_flashlight": {
            "name": "turn_on_flashlight",
            "description": "Turns the flashlight on.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        },
        "turn_off_flashlight": {
            "name": "turn_off_flashlight",
            "description": "Turns the flashlight off.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        },
        "create_calendar_event": {
            "name": "create_calendar_event",
            "description": "Creates a new calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime": {"type": "string", "description": "The date and time of the event in the format YYYY-MM-DDTHH:MM:SS."},
                    "title": {"type": "string", "description": "The title of the event."}
                },
                "required": ["datetime", "title"]
            }
        },
        "create_contact": {
            "name": "create_contact",
            "description": "Creates a contact in the phone's contact list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string", "description": "The first name of the contact."},
                    "last_name": {"type": "string", "description": "The last name of the contact."},
                    "phone_number": {"type": "string", "description": "The phone number of the contact."},
                    "email": {"type": "string", "description": "The email address of the contact."}
                },
                "required": ["first_name", "last_name"]
            }
        },
        "open_wifi_settings": {
            "name": "open_wifi_settings",
            "description": "Opens the Wi-Fi settings.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }

    # Extract which tools are mentioned in the text
    for tool_name, tool_def in tool_definitions.items():
        if tool_name in text or tool_def["description"] in text:
            tools.append({
                "type": "function",
                "function": tool_def
            })

    return tools

def extract_user_query(text):
    """Extract the user query from raw training format"""
    if "<start_of_turn>user\n" in text:
        user_parts = text.split("<start_of_turn>user\n")
        if len(user_parts) > 1:
            query = user_parts[-1].split("<end_of_turn>")[0].strip()
            return query
    return None

print("=" * 80)
print("Base vs Fine-Tuned Comparison (PROPER INFERENCE)")
print("=" * 80)

# Load test dataset
print("\n📥 Loading test dataset...")
test_dataset = load_from_disk("/home/inanna/dev/gemma/data/mobile_actions_curated_test")
print(f"✅ {len(test_dataset)} examples loaded")

# Load models
print("\n📥 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/functiongemma-270m-it",
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
base_tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
print("✅ Base model loaded")

print("\n📥 Loading fine-tuned model...")
ft_model = AutoModelForCausalLM.from_pretrained(
    "/home/inanna/dev/gemma/merged_model_fp32",
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
ft_tokenizer = AutoTokenizer.from_pretrained("/home/inanna/dev/gemma/merged_model_fp32")
print("✅ Fine-tuned model loaded")

print("\n" + "=" * 80)
print("RUNNING COMPARISONS (10 examples)")
print("=" * 80)

results = []

for i in range(10):
    example = test_dataset[i]
    text = example['text']

    # Extract tools and query
    tools = extract_tools_from_text(text)
    query = extract_user_query(text)

    if not query or not tools:
        continue

    print(f"\n{'─' * 80}")
    print(f"Example {i+1}")
    print(f"{'─' * 80}")
    print(f"Query: {query[:100]}...")
    print(f"Tools available: {len(tools)}")

    # Structure messages
    messages = [{"role": "user", "content": query}]

    # Test base model
    base_inputs = base_tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(base_model.device)

    with torch.no_grad():
        base_outputs = base_model.generate(**base_inputs, max_new_tokens=200, do_sample=False)
    base_generated = base_tokenizer.decode(base_outputs[0][base_inputs['input_ids'].shape[1]:], skip_special_tokens=False)

    # Test fine-tuned model
    ft_inputs = ft_tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(ft_model.device)

    with torch.no_grad():
        ft_outputs = ft_model.generate(**ft_inputs, max_new_tokens=200, do_sample=False)
    ft_generated = ft_tokenizer.decode(ft_outputs[0][ft_inputs['input_ids'].shape[1]:], skip_special_tokens=False)

    # Evaluate
    base_has_call = "call:" in base_generated
    ft_has_call = "call:" in ft_generated

    print(f"\n🔵 Base: {'✓' if base_has_call else '✗'}")
    if base_has_call:
        print(f"  {base_generated[:150]}...")

    print(f"\n🟢 Fine-tuned: {'✓' if ft_has_call else '✗'}")
    if ft_has_call:
        print(f"  {ft_generated[:150]}...")

    results.append({
        "query": query,
        "base_success": base_has_call,
        "ft_success": ft_has_call
    })

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

base_count = sum(1 for r in results if r['base_success'])
ft_count = sum(1 for r in results if r['ft_success'])
total = len(results)

print(f"\n🔵 Base Model: {base_count}/{total} ({base_count/total*100:.1f}%)")
print(f"🟢 Fine-Tuned Model: {ft_count}/{total} ({ft_count/total*100:.1f}%)")
print(f"\n📈 Improvement: +{ft_count - base_count} examples ({(ft_count-base_count)/total*100:.1f}%)")

# Save results
with open("comparison_results_proper.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: comparison_results_proper.json")
print("=" * 80)
