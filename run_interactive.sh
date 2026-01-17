#!/bin/bash
# Launcher for interactive function calling tester

cd /home/inanna/dev/gemma
source .venv/bin/activate
source .nix-lib-path
python interactive_test.py
