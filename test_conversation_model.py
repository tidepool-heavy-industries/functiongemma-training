#!/usr/bin/env python3
"""
Test the fine-tuned conversation tree model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

MODEL_PATH = "./training-runs/conversation/2026-01-17/merged_model_fp32"
BASE_MODEL = "google/functiongemma-270m-it"

print("=" * 80)
print("Testing Conversation Tree Model")
print("=" * 80)

# Load model
print(f"\n📥 Loading model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
# Load tokenizer from base model to avoid saved regex issue
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, fix_mistral_regex=True)
print("✅ Model loaded")

def parse_response(output):
    """Parse add_text and add_choice calls from model output"""
    # Parse add_text calls
    text_pattern = r'call:add_text\{content:<escape>([^<]*)<escape>\}'
    texts = re.findall(text_pattern, output)

    # Parse add_choice calls
    choice_pattern = r'call:add_choice\{text:<escape>([^<]*)<escape>,confidence:<escape>([^<]*)<escape>,target:<escape>([^<]*)<escape>\}'
    choices = re.findall(choice_pattern, output)

    return {
        'texts': texts,
        'choices': [
            {'text': c[0], 'confidence': c[1], 'target': c[2]}
            for c in choices
        ]
    }

def test_query(user_query, conversation_history=""):
    """Test a query using raw text format"""
    # Build prompt in FunctionGemma format
    prompt = f"""<start_of_turn>developer
You are an expert function calling AI assistant. Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2026-01-17T05:25:55
You have access to the following functions:

<start_function_declaration>add_text
Adds AI response text to current node.
Parameters:
  content (string): Response text (required)
<end_function_declaration>

<start_function_declaration>add_choice
Adds one predicted user response choice.
Parameters:
  text (string): User's response text (required)
  confidence (float): Confidence 0.0-1.0 (required)
  target (string): Target node name (required)
<end_function_declaration>
"""

    # Add conversation history if provided
    if conversation_history:
        prompt += f"\nConversation history:\n{conversation_history}\n"

    prompt += f"""
<end_of_turn>
<start_of_turn>user
{user_query}
<end_of_turn>
<start_of_turn>model
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get stop token IDs
    end_of_turn_id = tokenizer.convert_tokens_to_ids('<end_of_turn>')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=end_of_turn_id  # CRITICAL: Stop at <end_of_turn>
        )

    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    return result

# Test cases
test_cases = [
    {
        "name": "Fresh topic: Monads",
        "query": "What's a monad?",
        "history": ""
    },
    {
        "name": "Follow-up question",
        "query": "Example?",
        "history": """  User: "What's a monad?"
  AI: "Pattern for chaining operations with context."
  AI: "Like a box with rules for combining boxes.\""""
    },
    {
        "name": "Fresh topic: Distributed systems",
        "query": "CAP theorem?",
        "history": ""
    }
]

print("\n" + "=" * 80)
print("RUNNING TESTS")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'─' * 80}")
    print(f"Test {i}: {test['name']}")
    print(f"{'─' * 80}")
    print(f"Query: {test['query']}")

    # Generate
    output = test_query(test['query'], test['history'])

    # Parse result
    parsed = parse_response(output)

    print(f"\n🤖 Model Output (raw):")
    print(output[:400])
    if len(output) > 400:
        print("... [truncated]")

    print(f"\n📋 Parsed Response:")
    print(f"   AI Texts ({len(parsed['texts'])}):")
    for j, text in enumerate(parsed['texts'], 1):
        print(f"      {j}. {text}")

    print(f"   User Choices ({len(parsed['choices'])}):")
    for j, choice in enumerate(parsed['choices'], 1):
        print(f"      {j}. \"{choice['text']}\" (conf: {choice['confidence']}, target: {choice['target']})")

    # Validation
    issues = []
    if len(parsed['texts']) < 1:
        issues.append("No AI response texts generated")
    if len(parsed['choices']) != 3:
        issues.append(f"Expected 3 choices, got {len(parsed['choices'])}")

    if issues:
        print(f"\n⚠️ Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ PASS - Valid response structure")

print("\n" + "=" * 80)
print("Testing Complete")
print("=" * 80)
