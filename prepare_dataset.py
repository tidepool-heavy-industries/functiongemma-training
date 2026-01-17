#!/usr/bin/env python3
"""
FunctionGemma Dataset Preparation Script
Formats Google Mobile Actions dataset for FunctionGemma training
"""

import os
from datasets import load_dataset
from datetime import datetime

def format_function_declaration(tool):
    """Convert tool definition to FunctionGemma declaration format"""
    func = tool['function']
    name = func['name']
    description = func['description']
    parameters = func['parameters']

    declaration = f"<start_function_declaration>{name}\n"
    declaration += f"{description}\n"

    if parameters and parameters.get('properties'):
        declaration += "Parameters:\n"
        props = parameters['properties']
        required = parameters.get('required', [])

        for param_name, param_info in props.items():
            if param_info is not None:  # Skip None parameters
                param_type = param_info.get('type', 'STRING').lower()
                param_desc = param_info.get('description', '')
                required_marker = " (required)" if param_name in required else ""
                declaration += f"  {param_name} ({param_type}): {param_desc}{required_marker}\n"

    declaration += "<end_function_declaration>"
    return declaration

def format_function_call(tool_call):
    """Convert tool call to FunctionGemma call format"""
    func = tool_call['function']
    name = func['name']
    arguments = func['arguments']

    # Build function call
    call = f"call:{name}{{"

    params = []
    for param_name, param_value in arguments.items():
        if param_value is not None:  # Skip None values
            # Convert datetime objects to strings
            if hasattr(param_value, 'isoformat'):
                param_value = param_value.isoformat()

            # Escape string values
            params.append(f"{param_name}:<escape>{param_value}<escape>")

    call += ", ".join(params)
    call += "}"

    return call

def format_functiongemma_prompt(example):
    """
    Convert Mobile Actions example to FunctionGemma format.
    """

    # Extract tools and messages
    tools = example['tools']
    messages = example['messages']

    # Parse messages
    developer_msg = None
    user_msg = None
    assistant_msg = None

    for msg in messages:
        if msg['role'] == 'developer':
            developer_msg = msg
        elif msg['role'] == 'user':
            user_msg = msg
        elif msg['role'] == 'assistant':
            assistant_msg = msg

    # Build developer turn with function declarations
    developer_turn = "You are an expert function calling AI assistant. "

    # Add timestamp from developer message if present
    if developer_msg and developer_msg['content']:
        # Extract just the date/time info
        content_lines = developer_msg['content'].strip().split('\n')
        if content_lines:
            developer_turn += content_lines[0] + "\n"  # Add timestamp line
    else:
        developer_turn += f"Current date: {datetime.now().strftime('%Y-%m-%d')}.\n"

    developer_turn += "You have access to the following functions:\n\n"

    # Add function declarations
    for tool in tools:
        declaration = format_function_declaration(tool)
        developer_turn += declaration + "\n\n"

    # Build user turn
    user_turn = user_msg['content'] if user_msg and user_msg['content'] else ""

    # Build model turn with function calls
    model_turn = ""

    if assistant_msg and assistant_msg.get('tool_calls'):
        for tool_call in assistant_msg['tool_calls']:
            model_turn += "<start_function_call>\n"
            model_turn += format_function_call(tool_call) + "\n"
            model_turn += "<end_function_call>\n"
    else:
        # Fallback if no tool calls
        model_turn = "<start_function_call>\ncall:unknown{}\n<end_function_call>\n"

    # Assemble full prompt
    full_prompt = (
        f"<start_of_turn>developer\n{developer_turn}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_turn}<end_of_turn>\n"
        f"<start_of_turn>model\n{model_turn}<end_of_turn>"
    )

    return {"text": full_prompt}


def main():
    print("=" * 60)
    print("FunctionGemma Dataset Preparation")
    print("=" * 60)

    # Load Mobile Actions dataset
    print("\n📥 Loading Google Mobile Actions dataset...")
    dataset = load_dataset("google/mobile-actions", split="train")
    print(f"✅ Loaded {len(dataset)} training examples")
    print(f"📋 Columns: {dataset.column_names}")

    # Format dataset
    print("\n🔧 Formatting dataset for FunctionGemma...")
    formatted_dataset = dataset.map(
        format_functiongemma_prompt,
        remove_columns=dataset.column_names
    )

    # Print samples
    print("\n📄 Sample formatted examples:")
    print("=" * 60)
    print("EXAMPLE 1:")
    print("-" * 60)
    print(formatted_dataset[0]["text"][:1000] + "...\n")
    print("-" * 60)
    print("\nEXAMPLE 2:")
    print("-" * 60)
    print(formatted_dataset[1]["text"][:1000] + "...\n")
    print("-" * 60)

    # Save to disk
    output_path = "/home/inanna/dev/gemma/data/mobile_actions_formatted"
    print(f"\n💾 Saving formatted dataset to {output_path}...")
    formatted_dataset.save_to_disk(output_path)
    print(f"✅ Dataset saved successfully!")

    # Create train/eval split
    print("\n✂️ Creating train/eval split (90/10)...")
    train_test = formatted_dataset.train_test_split(test_size=0.1, seed=42)

    train_path = "/home/inanna/dev/gemma/data/mobile_actions_train"
    eval_path = "/home/inanna/dev/gemma/data/mobile_actions_eval"

    train_test["train"].save_to_disk(train_path)
    train_test["test"].save_to_disk(eval_path)

    print(f"✅ Training set:   {len(train_test['train'])} examples → {train_path}")
    print(f"✅ Evaluation set: {len(train_test['test'])} examples → {eval_path}")

    # Analyze dataset
    print("\n" + "=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)

    # Sample a few examples to check
    sample_size = min(100, len(formatted_dataset))
    avg_length = sum(len(formatted_dataset[i]["text"]) for i in range(sample_size)) / sample_size
    print(f"Average prompt length: {avg_length:.0f} characters")

    # Check for common functions
    print(f"\nDataset contains real user queries like:")
    print(f"  - '{dataset[0]['messages'][1]['content'][:60]}...'")
    print(f"  - '{dataset[1]['messages'][1]['content'][:60]}...'")

    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review the samples above to verify format")
    print(f"2. Run training script: python train_functiongemma.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
