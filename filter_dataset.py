#!/usr/bin/env python3
"""
Filter Mobile Actions dataset for highest quality examples
Target: 800-1000 training examples + test set for comparison
"""

from datasets import load_dataset, Dataset
import random

def score_example_quality(example):
    """
    Score example quality based on:
    - User query length (not too short, not too long)
    - Number of function parameters (more complex = better learning signal)
    - Query clarity
    """
    messages = example['messages']
    tools = example['tools']

    # Extract user query
    user_msg = next((m for m in messages if m['role'] == 'user'), None)
    if not user_msg or not user_msg.get('content'):
        return 0

    user_query = user_msg['content']

    # Extract assistant tool calls
    assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
    if not assistant_msg or not assistant_msg.get('tool_calls'):
        return 0

    tool_calls = assistant_msg['tool_calls']

    # Scoring criteria
    score = 0

    # 1. Query length (prefer 30-150 characters)
    query_len = len(user_query)
    if 30 <= query_len <= 150:
        score += 3
    elif 20 <= query_len <= 200:
        score += 2
    elif query_len > 10:
        score += 1

    # 2. Number of function parameters (more = better)
    total_params = 0
    for tc in tool_calls:
        args = tc['function']['arguments']
        # Count non-None parameters
        total_params += sum(1 for v in args.values() if v is not None)

    if total_params >= 3:
        score += 3
    elif total_params == 2:
        score += 2
    elif total_params == 1:
        score += 1

    # 3. Query complexity (contains dates, names, specific details)
    complexity_markers = [
        'at', 'to', 'for', 'about', 'with', 'on', 'in',  # Prepositions
        'please', 'can you', 'could you',  # Politeness
        ':', '@', ',',  # Structured data
    ]
    complexity_score = sum(1 for marker in complexity_markers if marker.lower() in user_query.lower())
    score += min(complexity_score, 3)

    # 4. Penalize very short queries
    if query_len < 20:
        score -= 2

    # 5. Bonus for multiple tool calls
    if len(tool_calls) > 1:
        score += 2

    # 6. Diversity bonus for specific functions
    interesting_functions = ['create_calendar_event', 'send_email', 'create_contact', 'show_map']
    for tc in tool_calls:
        if tc['function']['name'] in interesting_functions:
            score += 1
            break

    return max(score, 0)


def main():
    print("=" * 60)
    print("Dataset Quality Filtering")
    print("=" * 60)

    # Load dataset
    print("\n📥 Loading dataset...")
    dataset = load_dataset("google/mobile-actions", split="train")
    print(f"✅ Loaded {len(dataset)} examples")

    # Score all examples
    print("\n📊 Scoring examples for quality...")
    scored_examples = []

    for i, example in enumerate(dataset):
        score = score_example_quality(example)
        scored_examples.append({
            'index': i,
            'score': score,
            'example': example
        })

        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(dataset)} examples...")

    # Sort by score
    scored_examples.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n✅ Scoring complete")
    print(f"   Score range: {scored_examples[-1]['score']} to {scored_examples[0]['score']}")

    # Show distribution
    high_quality = sum(1 for x in scored_examples if x['score'] >= 8)
    medium_quality = sum(1 for x in scored_examples if 5 <= x['score'] < 8)
    low_quality = sum(1 for x in scored_examples if x['score'] < 5)

    print(f"\n📈 Quality Distribution:")
    print(f"   High quality (score ≥8):   {high_quality:,} examples")
    print(f"   Medium quality (5-7):      {medium_quality:,} examples")
    print(f"   Low quality (<5):          {low_quality:,} examples")

    # Select top 1000 for training + 100 for testing
    target_train = 900
    target_test = 100

    print(f"\n✂️ Selecting examples:")
    print(f"   Training set:   {target_train} examples (top quality)")
    print(f"   Test set:       {target_test} examples (for base vs fine-tuned comparison)")

    # Get top examples
    top_examples = scored_examples[:target_train + target_test]

    # Shuffle to mix quality levels slightly
    random.seed(42)
    random.shuffle(top_examples)

    # Split train/test
    train_examples = [x['example'] for x in top_examples[:target_train]]
    test_examples = [x['example'] for x in top_examples[target_train:]]

    # Show samples
    print(f"\n📄 Sample high-quality examples:")
    print("-" * 60)
    for i, scored in enumerate(scored_examples[:3]):
        ex = scored['example']
        user_msg = next((m for m in ex['messages'] if m['role'] == 'user'), None)
        if user_msg:
            print(f"{i+1}. (Score: {scored['score']}) {user_msg['content'][:80]}...")
    print("-" * 60)

    # Save using our existing format script
    print(f"\n🔧 Formatting selected examples...")

    # Import our formatting function
    import sys
    sys.path.insert(0, '/home/inanna/dev/gemma')
    from prepare_dataset import format_functiongemma_prompt

    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)

    # Format them
    train_formatted = train_dataset.map(
        format_functiongemma_prompt,
        remove_columns=train_dataset.column_names
    )
    test_formatted = test_dataset.map(
        format_functiongemma_prompt,
        remove_columns=test_dataset.column_names
    )

    # Save to disk
    train_path = "/home/inanna/dev/gemma/data/mobile_actions_curated_train"
    test_path = "/home/inanna/dev/gemma/data/mobile_actions_curated_test"

    train_formatted.save_to_disk(train_path)
    test_formatted.save_to_disk(test_path)

    print(f"✅ Training set:   {len(train_formatted)} examples → {train_path}")
    print(f"✅ Test set:       {len(test_formatted)} examples → {test_path}")

    # Save raw test queries for comparison script
    test_queries_path = "/home/inanna/dev/gemma/data/test_queries.txt"
    with open(test_queries_path, 'w') as f:
        for ex in test_examples[:20]:  # Save first 20 for quick comparison
            user_msg = next((m for m in ex['messages'] if m['role'] == 'user'), None)
            if user_msg and user_msg.get('content'):
                f.write(user_msg['content'] + '\n')

    print(f"✅ Test queries:   20 examples → {test_queries_path}")

    # Calculate training time estimate
    # ~4 seconds per step, batch size 1, gradient accumulation 8
    # steps = train_size / effective_batch_size
    steps_per_epoch = train_formatted.count // 8
    total_steps = steps_per_epoch * 3  # 3 epochs
    estimated_minutes = (total_steps * 4) / 60

    print(f"\n⏱️ Estimated training time:")
    print(f"   Steps per epoch: ~{steps_per_epoch}")
    print(f"   Total steps (3 epochs): ~{total_steps}")
    print(f"   Estimated time: ~{estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)")

    print("\n" + "=" * 60)
    print("✅ Curated dataset ready!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Update train script to use: {train_path}")
    print(f"2. Run training: python train_functiongemma.py")
    print(f"3. Compare models: python compare_models.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
