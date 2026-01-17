#!/usr/bin/env python3
"""
Fine-tune FunctionGemma for LSP symbol selection
Based on train_functiongemma_safe.py, adapted for select_symbols function
"""

import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
from datetime import datetime

# Configuration
BASE_MODEL = "google/functiongemma-270m-it"
TRAINING_DATA = "./training-runs/lsp/2026-01-17/training-shuffled.jsonl"
OUTPUT_DIR = f"./training-runs/lsp/2026-01-17/checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FINAL_LORA_DIR = "./training-runs/lsp/2026-01-17/checkpoints/final_lora"

# Training hyperparameters (same as successful mobile_actions training)
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10  # Increased from 3 - LSP task is harder, needs more training
MAX_SEQ_LENGTH = 2048

print("=" * 80)
print("FunctionGemma LSP Symbol Selection Fine-Tuning")
print("=" * 80)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FINAL_LORA_DIR), exist_ok=True)

print(f"\n📁 Configuration:")
print(f"   Base Model: {BASE_MODEL}")
print(f"   Training Data: {TRAINING_DATA}")
print(f"   Output: {OUTPUT_DIR}")
print(f"   LoRA Output: {FINAL_LORA_DIR}")

# Load training data
print(f"\n📥 Loading training data from {TRAINING_DATA}...")
with open(TRAINING_DATA, 'r') as f:
    data = [json.loads(line) for line in f]

print(f"✅ Loaded {len(data)} examples")

# Quick stats
negative_count = sum(1 for ex in data if 'selected:<escape><escape>' in ex['text'])
positive_count = len(data) - negative_count
print(f"   Positive examples: {positive_count} ({positive_count/len(data)*100:.1f}%)")
print(f"   Negative examples: {negative_count} ({negative_count/len(data)*100:.1f}%)")

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)

# Load model and tokenizer
print(f"\n📥 Loading base model: {BASE_MODEL}")
print("   Using FP32 (required for Gemma3)")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, fix_mistral_regex=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,  # CRITICAL: Gemma3 requires FP32
    device_map="auto",
    trust_remote_code=True
)

print("✅ Model loaded")
print(f"   Model dtype: {model.dtype}")
print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Prepare model for training
print("\n🔧 Preparing model for training...")
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Alpha
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenization function
def tokenize_function(examples):
    """Tokenize the text field"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False
    )

print("\n🔧 Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

print(f"✅ Tokenized {len(tokenized_dataset)} examples")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    fp16=False,  # CRITICAL: Must be False for Gemma3
    bf16=False,
    gradient_checkpointing=False,  # Disabled to avoid TorchDynamo issues
    optim="adamw_torch",
    warmup_steps=50,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=[],  # Disabled tensorboard (not installed)
    save_strategy="steps",
    push_to_hub=False,
    remove_unused_columns=True,
)

print("\n📊 Training Configuration:")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch Size: {PER_DEVICE_BATCH_SIZE}")
print(f"   Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"   Effective Batch Size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"   FP16: {training_args.fp16}")
print(f"   Gradient Checkpointing: {training_args.gradient_checkpointing}")

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
print("\n" + "=" * 80)
print("🚀 Starting Training")
print("=" * 80)

train_result = trainer.train()

print("\n" + "=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print(f"   Final Loss: {train_result.training_loss:.4f}")
print(f"   Training Time: {train_result.metrics['train_runtime']:.2f}s")
print(f"   Samples/Second: {train_result.metrics['train_samples_per_second']:.2f}")

# Save final LoRA adapters
print(f"\n💾 Saving final LoRA adapters to: {FINAL_LORA_DIR}")
model.save_pretrained(FINAL_LORA_DIR)
tokenizer.save_pretrained(FINAL_LORA_DIR)

print("\n" + "=" * 80)
print("🎉 Training Complete!")
print("=" * 80)
print(f"\nNext steps:")
print(f"1. Merge adapters: python merge_lsp_model.py")
print(f"2. Test the model: python test_lsp_model.py")
print("=" * 80)
