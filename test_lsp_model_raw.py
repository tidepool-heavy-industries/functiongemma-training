#!/usr/bin/env python3
"""
Test the fine-tuned LSP symbol selection model using RAW format (matching training)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

MODEL_PATH = "./training-runs/lsp/2026-01-17/merged_model_fp32"

print("=" * 80)
print("Testing LSP Symbol Selection Model (RAW FORMAT)")
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

def parse_selection(output):
    """Parse selected symbols from model output"""
    match = re.search(r'selected:<escape>([^<]*)<escape>', output)
    if match:
        selected = match.group(1).strip()
        if selected:
            # Split by comma and strip whitespace
            return [s.strip() for s in selected.split(',')]
        return []
    return None

def test_query_raw(full_prompt):
    """Test a query using raw text format (matching training)"""
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    return result

# Test cases using EXACT training format
test_cases = [
    {
        "name": "Positive: Parsing response",
        "prompt": """<start_of_turn>developer
You are an expert function calling AI assistant.
You have access to the following functions:

<start_function_declaration>select_symbols
Select relevant symbols from candidates that help understand the topic in context of the current symbol.
Parameters:
  selected (array): Array of selected symbol names from the candidate list. (required)
<end_function_declaration>

<end_of_turn>
<start_of_turn>user
Topic: How do I send a response?
Symbol: parseSelectionResponse
Module: Tidepool.Control.Scout.Teach.Gemma
Package: tidepool-control-server
Signature: parseSelectionResponse :: ByteString -> Either Text [Text]

Candidates:
  Fields: (none)
  Inputs: (none)
  Output: Parse, Ollama
  References: Gemma.hs:243
<end_of_turn>
<start_of_turn>model
""",
        "expected": ["Parse", "Ollama"]
    },
    {
        "name": "Negative: Unrelated topic",
        "prompt": """<start_of_turn>developer
You are an expert function calling AI assistant.
You have access to the following functions:

<start_function_declaration>select_symbols
Select relevant symbols from candidates that help understand the topic in context of the current symbol.
Parameters:
  selected (array): Array of selected symbol names from the candidate list. (required)
<end_function_declaration>

<end_of_turn>
<start_of_turn>user
Topic: What should I avoid when working with database connections?
Symbol: parseSelectionResponse
Module: Tidepool.Control.Scout.Teach.Gemma
Package: tidepool-control-server
Signature: parseSelectionResponse :: ByteString -> Either Text [Text]

Candidates:
  Fields: (none)
  Inputs: Database, Connection
  Output: Parse, Ollama
  References: Gemma.hs:243
<end_of_turn>
<start_of_turn>model
""",
        "expected": []
    },
    {
        "name": "Positive: Type transformation",
        "prompt": """<start_of_turn>developer
You are an expert function calling AI assistant.
You have access to the following functions:

<start_function_declaration>select_symbols
Select relevant symbols from candidates that help understand the topic in context of the current symbol.
Parameters:
  selected (array): Array of selected symbol names from the candidate list. (required)
<end_function_declaration>

<end_of_turn>
<start_of_turn>user
Topic: How do I transform Type and TeachGemma?
Symbol: runTeachGemma
Module: Tidepool.Control.Scout.Teach.Gemma
Package: tidepool-control-server
Signature: runTeachGemma :: Eff (TeachGemma : effs) a -> Eff effs a

Candidates:
  Fields: (none)
  Inputs: Type, TeachGemma
  Output: Mock, Simply
  References: Gemma.hs:39
<end_of_turn>
<start_of_turn>model
""",
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

    # Extract topic line for display
    topic_line = [line for line in test['prompt'].split('\n') if line.startswith('Topic:')]
    if topic_line:
        print(topic_line[0])

    # Generate
    output = test_query_raw(test['prompt'])

    # Parse result
    selected = parse_selection(output)

    print(f"\n🤖 Model Output:")
    print(output[:200])
    if len(output) > 200:
        print("... [truncated]")

    print(f"\n📋 Parsed Selection: {selected}")
    print(f"🎯 Expected: {test['expected']}")

    # Check correctness (handle ordering differences)
    if selected == test['expected']:
        print("✅ PASS")
    elif selected is not None and set(selected) == set(test['expected']):
        print("✅ PASS (different order)")
    else:
        print("❌ FAIL")

print("\n" + "=" * 80)
print("Testing Complete")
print("=" * 80)
