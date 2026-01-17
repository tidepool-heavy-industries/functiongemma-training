#!/usr/bin/env python3
"""
Interactive function calling tester for FunctionGemma fine-tuned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json

print("=" * 80)
print("FunctionGemma Fine-Tuned - Interactive Tester")
print("=" * 80)

# Load model
print("\n📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/inanna/dev/gemma/merged_model_fp32",
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/home/inanna/dev/gemma/merged_model_fp32")
print("✅ Model loaded!")

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "turn_on_flashlight",
            "description": "Turns the flashlight on.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_off_flashlight",
            "description": "Turns the flashlight off.",
            "parameters": {"type": "object", "properties": {}, "required": []}
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
                    "datetime": {"type": "string", "description": "Date/time in YYYY-MM-DDTHH:MM:SS format"},
                    "title": {"type": "string", "description": "Event title"}
                },
                "required": ["datetime", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_contact",
            "description": "Creates a contact in the phone's contact list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string", "description": "First name"},
                    "last_name": {"type": "string", "description": "Last name"},
                    "phone_number": {"type": "string", "description": "Phone number"},
                    "email": {"type": "string", "description": "Email address"}
                },
                "required": ["first_name", "last_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Sends an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"}
                },
                "required": ["to", "subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_wifi_settings",
            "description": "Opens the Wi-Fi settings.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

print("\n📋 Available functions:")
for i, tool in enumerate(tools, 1):
    print(f"  {i}. {tool['function']['name']} - {tool['function']['description']}")

def parse_function_calls(raw_output):
    """Parse function calls from model output"""
    function_calls = []

    # Find all function calls
    pattern = r'<start_function_call>call:([^{]+)\{([^}]*)\}<end_function_call>'
    matches = re.findall(pattern, raw_output)

    for func_name, params_str in matches:
        func_name = func_name.strip()

        # Parse parameters
        params = {}
        if params_str.strip():
            # Split by comma (but not inside <escape> blocks)
            param_parts = re.split(r',(?![^<]*<escape>)', params_str)

            for part in param_parts:
                part = part.strip()
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove <escape> tokens
                    value = re.sub(r'<escape>', '', value)

                    params[key] = value

        function_calls.append({
            "function": func_name,
            "parameters": params
        })

    return function_calls

def run_query(query):
    """Run a query through the model"""
    messages = [{"role": "user", "content": query}]

    # Apply chat template
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
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

    return result

print("\n" + "=" * 80)
print("Ready! Type your query (or 'quit' to exit)")
print("=" * 80)

while True:
    print("\n" + "─" * 80)
    query = input("\n🗣️  You: ").strip()

    if query.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break

    if not query:
        continue

    print("\n🤖 Generating...")
    raw_output = run_query(query)

    # Parse function calls
    function_calls = parse_function_calls(raw_output)

    # Display results
    print("\n" + "=" * 80)
    print("📤 RAW OUTPUT:")
    print("=" * 80)
    print(raw_output[:500])
    if len(raw_output) > 500:
        print("... [truncated]")

    print("\n" + "=" * 80)
    print("🔧 PARSED FUNCTION CALLS:")
    print("=" * 80)

    if function_calls:
        for i, call in enumerate(function_calls, 1):
            print(f"\n{i}. Function: {call['function']}")
            if call['parameters']:
                print("   Parameters:")
                for key, value in call['parameters'].items():
                    print(f"     - {key}: {value}")
            else:
                print("   Parameters: (none)")

        print("\n" + "─" * 80)
        print("📋 JSON FORMAT:")
        print("─" * 80)
        print(json.dumps(function_calls, indent=2))
    else:
        print("⚠️  No function calls detected in output")

    print("\n" + "=" * 80)
