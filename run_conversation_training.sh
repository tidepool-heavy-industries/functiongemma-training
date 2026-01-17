#!/bin/bash

echo "========================================="
echo "FunctionGemma Conversation Training"
echo "========================================="
echo ""

cd /home/inanna/dev/gemma

echo "Activating environment..."
source .venv/bin/activate
source .nix-lib-path

echo ""
echo "Starting training on 609 examples..."
echo "This will take approximately 15-30 minutes"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

python train_conversation.py
