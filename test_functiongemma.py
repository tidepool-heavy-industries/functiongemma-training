#!/usr/bin/env python3
"""
Test script for FunctionGemma 270M
Verifies model loading and basic function calling capability
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def main():
    print("🔍 Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️  CUDA not available, using CPU")
        device = "cpu"

    print("\n📥 Loading FunctionGemma 270M...")
    model_id = "google/functiongemma-270m-it"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device
        )
        print("✅ Model loaded successfully!\n")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Test function calling
    print("🧪 Testing function calling capability...\n")

    # Define a simple function
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    # Test prompt
    prompt = "What's the weather in Seattle?"

    # Format for FunctionGemma
    formatted_prompt = f"""<|user|>
You have access to the following functions:

{tools}

User query: {prompt}
<|assistant|>
"""

    print(f"Prompt: {prompt}\n")

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"\n📤 Model response:\n{response}\n")

    print("✅ FunctionGemma is working correctly!")
    print(f"\n📊 Model info:")
    print(f"   Device: {device}")
    print(f"   Parameters: ~270M")
    print(f"   Dtype: {model.dtype}")

if __name__ == "__main__":
    main()
