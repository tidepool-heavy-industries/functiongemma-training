#!/usr/bin/env python3
"""
Test the fine-tuned LSP symbol selection model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

MODEL_PATH = "./training-runs/lsp/2026-01-17/merged_model_fp32"

print("=" * 80)
print("Testing LSP Symbol Selection Model")
print("=" * 80)

# Load model
print(f"\n📥 Loading model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
print("✅ Model loaded")

# Define tool
tools = [{
    "type": "function",
    "function": {
        "name": "select_symbols",
        "description": "Select relevant symbols from candidates that help understand the topic in context of the current symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "selected": {
                    "type": "array",
                    "description": "Array of selected symbol names from the candidate list."
                }
            },
            "required": ["selected"]
        }
    }
}]

def parse_selection(output):
    """Parse selected symbols from model output"""
    match = re.search(r'selected:<escape>([^<]*)<escape>', output)
    if match:
        selected = match.group(1).strip()
        if selected:
            return selected.split(',')
        return []
    return None

def test_query(user_query):
    """Test a query"""
    messages = [{"role": "user", "content": user_query}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    return result

# Test cases
test_cases = [
    {
        "name": "Positive: Parsing response",
        "query": """Topic: How do I send a response?
Symbol: parseSelectionResponse
Module: Tidepool.Control.Scout.Teach.Gemma
Package: tidepool-control-server
Signature: parseSelectionResponse :: ByteString -> Either Text [Text]

Candidates:
  Fields: (none)
  Inputs: (none)
  Output: Parse, Ollama
  References: Gemma.hs:243""",
        "expected": ["Parse", "Ollama"]
    },
    {
        "name": "Negative: Unrelated topic",
        "query": """Topic: What should I avoid when working with database connections?
Symbol: parseSelectionResponse
Module: Tidepool.Control.Scout.Teach.Gemma
Package: tidepool-control-server
Signature: parseSelectionResponse :: ByteString -> Either Text [Text]

Candidates:
  Fields: (none)
  Inputs: Database, Connection
  Output: Parse, Ollama
  References: Gemma.hs:243""",
        "expected": []
    },
    {
        "name": "Positive: Type transformation",
        "query": """Topic: How do I transform Type and TeachGemma?
Symbol: runTeachGemma
Module: Tidepool.Control.Scout.Teach.Gemma
Package: tidepool-control-server
Signature: runTeachGemma :: Eff (TeachGemma : effs) a -> Eff effs a

Candidates:
  Fields: (none)
  Inputs: Type, TeachGemma
  Output: Mock, Simply
  References: Gemma.hs:39""",
        "expected": ["Type", "TeachGemma"]
    }
]

print("\n" + "=" * 80)
print("RUNNING TESTS")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'─' * 80}")
    print(f"Test {i}: {test['name']}")
    print(f"{'─' * 80}")

    # Show truncated query
    topic = test['query'].split('\n')[0]
    symbol = test['query'].split('\n')[1]
    print(f"{topic}")
    print(f"{symbol}")

    # Generate
    output = test_query(test['query'])

    # Parse result
    selected = parse_selection(output)

    print(f"\n🤖 Model Output:")
    print(output[:200])
    if len(output) > 200:
        print("... [truncated]")

    print(f"\n📋 Parsed Selection: {selected}")
    print(f"🎯 Expected: {test['expected']}")

    # Check correctness
    if selected == test['expected']:
        print("✅ PASS")
    elif selected is not None and set(selected) == set(test['expected']):
        print("✅ PASS (different order)")
    else:
        print("❌ FAIL")

print("\n" + "=" * 80)
print("Testing Complete")
print("=" * 80)
