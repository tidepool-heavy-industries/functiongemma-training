#!/usr/bin/env python3
"""
Compare Base Model vs Fine-Tuned Model
Tests both models on the same queries to measure improvement
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import json

def load_model(model_path):
    """Load model and tokenizer in FP32 (required for Gemma3)"""
    print(f"\n📥 Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load in FP32 - Gemma3 requires this
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"✅ Model loaded in FP32")
    print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for deterministic function calling
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response


def extract_function_call(response):
    """Extract just the function call from the response"""
    # Find the model's response (after last <start_of_turn>model)
    if "<start_of_turn>model" in response:
        model_response = response.split("<start_of_turn>model")[-1]

        # Extract function call
        if "<start_function_call>" in model_response and "<end_function_call>" in model_response:
            start = model_response.find("<start_function_call>") + len("<start_function_call>")
            end = model_response.find("<end_function_call>")
            return model_response[start:end].strip()

    return None


def main():
    print("=" * 60)
    print("Base Model vs Fine-Tuned Model Comparison")
    print("=" * 60)

    # Load test dataset
    print("\n📥 Loading test dataset...")
    test_data_path = "/home/inanna/dev/gemma/data/mobile_actions_curated_test"
    test_dataset = load_from_disk(test_data_path)
    print(f"✅ {len(test_dataset)} test examples loaded")

    # Test on first 5 examples (faster for quick validation)
    num_tests = 5
    print(f"\n🧪 Testing on {num_tests} examples")

    # Load base model
    print("\n" + "=" * 60)
    print("LOADING BASE MODEL")
    print("=" * 60)
    base_model, base_tokenizer = load_model("google/functiongemma-270m-it")

    # Load fine-tuned model
    print("\n" + "=" * 60)
    print("LOADING FINE-TUNED MODEL")
    print("=" * 60)
    finetuned_path = "/home/inanna/dev/gemma/merged_model_fp32"
    finetuned_model, finetuned_tokenizer = load_model(finetuned_path)

    # Run comparison
    print("\n" + "=" * 60)
    print("RUNNING COMPARISONS")
    print("=" * 60)

    results = []

    for i in range(min(num_tests, len(test_dataset))):
        example = test_dataset[i]
        full_text = example['text']

        # Extract ground truth (expected model output)
        # Split at the last <start_of_turn>model to separate prompt from expected output
        parts = full_text.split('<start_of_turn>model\n')
        if len(parts) >= 2:
            prompt_only = '<start_of_turn>model\n'.join(parts[:-1]) + '<start_of_turn>model\n'
            ground_truth = parts[-1].replace('<end_of_turn>', '').strip()
        else:
            prompt_only = full_text
            ground_truth = "N/A"

        print(f"\n{'=' * 60}")
        print(f"TEST {i+1}/{num_tests}")
        print(f"{'=' * 60}")

        # Extract user query for display
        if "<start_of_turn>user\n" in prompt_only:
            user_parts = prompt_only.split("<start_of_turn>user\n")
            if len(user_parts) > 1:
                user_query = user_parts[-1].split("<end_of_turn>")[0].strip()
                # Truncate long queries
                if len(user_query) > 100:
                    user_query = user_query[:100] + "..."
                print(f"User Query: {user_query}")
        else:
            user_query = "[Could not extract]"
            print(f"User Query: {user_query}")

        # Truncate ground truth for display
        gt_display = ground_truth[:150] + "..." if len(ground_truth) > 150 else ground_truth
        print(f"\nExpected Output:\n  {gt_display}")

        # Base model prediction
        print(f"\n🔵 Base Model:")
        base_response = generate_response(base_model, base_tokenizer, prompt_only, max_new_tokens=150)
        base_call = extract_function_call(base_response)
        if base_call:
            base_display = base_call[:150] + "..." if len(base_call) > 150 else base_call
            print(f"  {base_display}")
        else:
            print(f"  [No function call generated]")

        # Fine-tuned model prediction
        print(f"\n🟢 Fine-Tuned Model:")
        finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt_only, max_new_tokens=150)
        finetuned_call = extract_function_call(finetuned_response)
        if finetuned_call:
            ft_display = finetuned_call[:150] + "..." if len(finetuned_call) > 150 else finetuned_call
            print(f"  {ft_display}")
        else:
            print(f"  [No function call generated]")

        # Check if function call syntax is present (looser than exact match)
        base_has_call = base_call is not None and len(base_call) > 0
        finetuned_has_call = finetuned_call is not None and len(finetuned_call) > 0

        print(f"\n✓ Base generated function call: {base_has_call}")
        print(f"✓ Fine-tuned generated function call: {finetuned_has_call}")

        results.append({
            'query': user_query,
            'ground_truth': ground_truth,
            'base_call': base_call,
            'finetuned_call': finetuned_call,
            'base_has_call': base_has_call,
            'finetuned_has_call': finetuned_has_call
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    base_success = sum(1 for r in results if r['base_has_call'])
    finetuned_success = sum(1 for r in results if r['finetuned_has_call'])

    base_rate = (base_success / len(results)) * 100
    finetuned_rate = (finetuned_success / len(results)) * 100

    print(f"\n🔵 Base Model: {base_success}/{len(results)} generated function calls ({base_rate:.1f}%)")
    print(f"🟢 Fine-Tuned Model: {finetuned_success}/{len(results)} generated function calls ({finetuned_rate:.1f}%)")
    print(f"\n📈 Improvement: {finetuned_rate - base_rate:+.1f}%")

    # Save results
    results_path = "/home/inanna/dev/gemma/comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Full results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("NOTE: This is a quick validation on 5 examples.")
    print("For comprehensive metrics, run on all 100 test examples.")
    print("=" * 60)


if __name__ == "__main__":
    main()
