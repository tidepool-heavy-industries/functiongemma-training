#!/usr/bin/env bash
# Activate Python venv with Nix library paths
# Usage: source activate-venv.sh

if [ ! -d .venv ]; then
    echo "❌ Virtual environment not found. Run 'nix develop' first."
    return 1
fi

if [ ! -f .nix-lib-path ]; then
    echo "❌ Nix library paths not found. Run 'nix develop' first."
    return 1
fi

source .venv/bin/activate
source .nix-lib-path

echo "✅ Python venv activated with Nix CUDA libraries"
