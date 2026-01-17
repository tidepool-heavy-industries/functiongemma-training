#!/usr/bin/env python3
"""
FunctionGemma 270M Fine-Tuning Script
Optimized for GTX 1660 SUPER (6GB VRAM, FP16-only, Turing CC 7.5)
Using QLoRA + Unsloth for memory-efficient training
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
import os

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_NAME = "google/functiongemma-270m-it"
MAX_SEQ_LENGTH = 2048  # FunctionGemma supports up to 32K, but limiting for VRAM

# LoRA configuration (optimized for 6GB VRAM)
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # Scaling factor (typically 2*r)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    # Optionally add more for better quality: "gate_proj", "up_proj", "down_proj"
]

# Training hyperparameters (optimized for GTX 1660 SUPER)
OUTPUT_DIR = "/home/inanna/dev/gemma/outputs"
CHECKPOINT_DIR = "/home/inanna/dev/gemma/checkpoints"
TRAIN_DATA_PATH = "/home/inanna/dev/gemma/data/mobile_actions_curated_train"
EVAL_DATA_PATH = "/home/inanna/dev/gemma/data/mobile_actions_curated_test"

PER_DEVICE_BATCH_SIZE = 1  # CRITICAL: Must be 1 for 6GB VRAM
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 1 * 8 = 8
LEARNING_RATE = 1e-5  # Conservative for FP16
NUM_EPOCHS = 3
MAX_GRAD_NORM = 1.0  # Gradient clipping (CRITICAL for FP16 stability)
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100

# ============================================================================
# GPU Check
# ============================================================================

print("=" * 60)
print("FunctionGemma 270M Fine-Tuning")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
else:
    print("⚠️  WARNING: CUDA not available. Training will be very slow!")

print("=" * 60)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# Load Model in 4-bit Quantization
# ============================================================================

print("\n📥 Loading FunctionGemma 270M in 4-bit quantization...")
print(f"   Model: {MODEL_NAME}")
print(f"   Max sequence length: {MAX_SEQ_LENGTH}")
print(f"   Quantization: 4-bit (QLoRA)")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (will use FP16 for GTX 1660 SUPER)
    load_in_4bit=True,  # CRITICAL for 6GB VRAM
    device_map="auto",
)

print(f"✅ Model loaded successfully")
print(f"   Detected dtype: {model.dtype}")

# ============================================================================
# Add LoRA Adapters
# ============================================================================

print(f"\n🔧 Adding LoRA adapters...")
print(f"   Rank: {LORA_R}")
print(f"   Alpha: {LORA_ALPHA}")
print(f"   Target modules: {LORA_TARGET_MODULES}")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
)

print("✅ LoRA adapters added")

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/all_params*100:.2f}% of total)")

# ============================================================================
# Load Dataset
# ============================================================================

print(f"\n📚 Loading datasets...")
print(f"   Training: {TRAIN_DATA_PATH}")
print(f"   Evaluation: {EVAL_DATA_PATH}")

train_dataset = load_from_disk(TRAIN_DATA_PATH)
eval_dataset = load_from_disk(EVAL_DATA_PATH)

print(f"✅ Datasets loaded")
print(f"   Training samples: {len(train_dataset):,}")
print(f"   Evaluation samples: {len(eval_dataset):,}")

# Print sample
print(f"\n📄 Sample training example:")
print("-" * 60)
print(train_dataset[0]["text"][:500] + "...")
print("-" * 60)

# ============================================================================
# Training Arguments
# ============================================================================

print(f"\n⚙️  Configuring training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Batch size strategy for 6GB VRAM
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # Learning rate and schedule
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,

    # Training duration
    num_train_epochs=NUM_EPOCHS,
    max_steps=-1,  # Train for full epochs

    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=MAX_GRAD_NORM,

    # Memory optimization
    gradient_checkpointing=True,
    fp16=True,  # FP16 training (GTX 1660 SUPER)
    bf16=False,  # NOT available on GTX 1660 SUPER

    # Logging and checkpointing
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,  # Keep only 2 checkpoints (disk space)

    # Evaluation
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Misc
    report_to="none",  # Disable WandB/TensorBoard to save memory
    seed=42,
    dataloader_num_workers=2,
)

print(f"✅ Training configuration:")
print(f"   Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Number of epochs: {NUM_EPOCHS}")
print(f"   FP16: {training_args.fp16}")
print(f"   Gradient checkpointing: {training_args.gradient_checkpointing}")

# ============================================================================
# Trainer
# ============================================================================

print(f"\n🚀 Initializing trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

print("✅ Trainer initialized")

# ============================================================================
# Training
# ============================================================================

print("\n" + "=" * 60)
print("🦥 Starting QLoRA Fine-Tuning")
print("=" * 60)
print(f"⏱️  Estimated time: 2-4 hours (depends on hardware)")
print(f"💾 Checkpoints will be saved to: {OUTPUT_DIR}")
print(f"📊 Monitor GPU usage: watch -n 1 nvidia-smi")
print("=" * 60)
print()

try:
    trainer.train()
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print("=" * 60)

except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ Training failed with error:")
    print(f"   {str(e)}")
    print("=" * 60)
    raise

# ============================================================================
# Save Final Model
# ============================================================================

print(f"\n💾 Saving final LoRA adapters...")
final_adapter_path = f"{CHECKPOINT_DIR}/final_lora"
model.save_pretrained(final_adapter_path)
tokenizer.save_pretrained(final_adapter_path)

print(f"✅ Final LoRA adapters saved to: {final_adapter_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("🎉 Training Complete!")
print("=" * 60)
print(f"📍 LoRA adapters saved to: {final_adapter_path}")
print(f"\nNext steps:")
print(f"1. Run merge script: python merge_model.py")
print(f"2. Convert to GGUF for Ollama deployment")
print("=" * 60)
