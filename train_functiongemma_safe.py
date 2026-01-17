#!/usr/bin/env python3
"""
FunctionGemma 270M Fine-Tuning Script (Safe Mode)
Uses standard Transformers/PEFT without Unsloth optimizations
For GTX 1660 SUPER (6GB VRAM, FP16-only)
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import os

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "google/functiongemma-270m-it"
MAX_SEQ_LENGTH = 2048

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training
OUTPUT_DIR = "/home/inanna/dev/gemma/outputs"
CHECKPOINT_DIR = "/home/inanna/dev/gemma/checkpoints"
TRAIN_DATA_PATH = "/home/inanna/dev/gemma/data/mobile_actions_curated_train"
EVAL_DATA_PATH = "/home/inanna/dev/gemma/data/mobile_actions_curated_test"

PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100

# ============================================================================
# Setup
# ============================================================================

print("=" * 60)
print("FunctionGemma 270M Fine-Tuning (Safe Mode)")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# Load Model with 4-bit Quantization
# ============================================================================

print("\n📥 Loading model in 4-bit...")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"✅ Model loaded")

# ============================================================================
# Prepare for Training
# ============================================================================

print("\n🔧 Preparing model for training...")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# Add LoRA adapters
model = get_peft_model(model, lora_config)

print(f"✅ LoRA adapters added")
model.print_trainable_parameters()

# ============================================================================
# Load Dataset
# ============================================================================

print(f"\n📚 Loading datasets...")
train_dataset = load_from_disk(TRAIN_DATA_PATH)
eval_dataset = load_from_disk(EVAL_DATA_PATH)

print(f"✅ Training: {len(train_dataset)} examples")
print(f"✅ Evaluation: {len(eval_dataset)} examples")

# ============================================================================
# Tokenize Dataset
# ============================================================================

print(f"\n🔧 Tokenizing datasets...")

def tokenize_function(examples):
    """Tokenize the text field"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,  # Dynamic padding in collator
        return_tensors=None,
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    num_proc=4,
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    num_proc=4,
)

print(f"✅ Tokenization complete")

# ============================================================================
# Training Arguments
# ============================================================================

print(f"\n⚙️ Configuring training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Batch size
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # Learning rate
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,

    # Training duration
    num_train_epochs=NUM_EPOCHS,

    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=MAX_GRAD_NORM,

    # Precision - NO gradient checkpointing to avoid TorchDynamo issues
    fp16=True,
    bf16=False,
    gradient_checkpointing=False,  # DISABLED to avoid compilation errors

    # Logging
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=LOGGING_STEPS,

    # Saving
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,

    # Evaluation
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    load_best_model_at_end=True,

    # Misc
    report_to="none",
    seed=42,
)

print(f"✅ Configuration complete")

# ============================================================================
# Data Collator
# ============================================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

# ============================================================================
# Trainer
# ============================================================================

print(f"\n🚀 Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print(f"✅ Trainer initialized")

# ============================================================================
# Training
# ============================================================================

print("\n" + "=" * 60)
print("🚀 Starting Training")
print("=" * 60)
print(f"⏱️ Estimated time: ~30-45 minutes")
print(f"💾 Checkpoints: {OUTPUT_DIR}")
print(f"📊 Monitor: watch -n 1 nvidia-smi")
print("=" * 60)
print()

try:
    trainer.train()
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print("=" * 60)
except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ Training failed: {str(e)}")
    print("=" * 60)
    raise

# ============================================================================
# Save Final Model
# ============================================================================

print(f"\n💾 Saving final model...")
final_path = f"{CHECKPOINT_DIR}/final_lora"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"✅ Saved to: {final_path}")

print("\n" + "=" * 60)
print("🎉 Training Complete!")
print("=" * 60)
print(f"📍 LoRA adapters: {final_path}")
print(f"\nNext: python merge_model.py")
print("=" * 60)
