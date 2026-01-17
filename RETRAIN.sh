#!/bin/bash
# Clean retrain with corrected tokenizer and 7 epochs
# Expected runtime: ~40-45 minutes

set -e

cd /home/inanna/dev/gemma

echo "========================================="
echo "Clean Retrain - Conversation Model"
echo "========================================="
echo "Epochs: 7 (was 3)"
echo "Tokenizer: FIXED (fix_mistral_regex=True)"
echo "Expected time: 40-45 minutes"
echo "========================================="
echo ""

# Activate environment
source .venv/bin/activate
source .nix-lib-path

# Run training
echo "Starting training..."
python train_conversation.py

# Merge LoRA adapters
echo ""
echo "========================================="
echo "Merging LoRA adapters..."
echo "========================================="
python merge_conversation_model.py

# Test the model
echo ""
echo "========================================="
echo "Testing the model..."
echo "========================================="
python test_conversation_model.py

echo ""
echo "========================================="
echo "✅ Complete!"
echo "========================================="
echo "Check the test results above to see if the model learned properly."
echo "Target: Loss < 1.0, proper function call structure"
echo "========================================="
