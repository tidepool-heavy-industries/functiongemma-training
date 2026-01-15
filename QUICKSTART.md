# FunctionGemma Quick Start

## 1. Enter the Nix Environment

First time setup:
```bash
# Allow direnv to load the environment
direnv allow
```

This will download and set up everything. **The first time will take 5-10 minutes** as Nix downloads Python, CUDA, and all dependencies.

You'll see a banner like:
```
🚀 FunctionGemma Development Environment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Platform: x86_64-linux
Acceleration: CUDA (NVIDIA)
...
```

## 2. Activate Python Virtual Environment

The Nix shell creates a `.venv` automatically. Activate it:

```bash
source .venv/bin/activate
```

## 3. Install FunctionGemma Dependencies

```bash
pip install --upgrade pip
pip install torch transformers accelerate bitsandbytes
```

## 4. Test FunctionGemma

Run the test script:

```bash
python test_functiongemma.py
```

This will:
- Check CUDA availability
- Download FunctionGemma 270M (first time only, ~500MB)
- Run a test inference with function calling
- Display the results

## 5. Expected Output

You should see:
```
🔍 Checking CUDA availability...
✅ CUDA available: NVIDIA GeForce RTX ...

📥 Loading FunctionGemma 270M...
✅ Model loaded successfully!

🧪 Testing function calling capability...
Prompt: What's the weather in Seattle?

Generating response...

📤 Model response:
[Model's function call output]

✅ FunctionGemma is working correctly!
```

## Troubleshooting

### CUDA not detected
```bash
# Check if NVIDIA driver is visible from WSL2
nvidia-smi
```

If this fails, ensure your Windows host has NVIDIA drivers installed and WSL2 is configured for GPU access.

### Model download fails
The model downloads to `$HF_HOME` which is set to `$HOME/.cache/huggingface/`. Ensure you have ~2GB free space and network access to huggingface.co.

### Out of memory
The 270M model should run on any GPU with 2GB+ VRAM. If you get OOM errors:
```python
# In the test script, change to 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)
```

## Next Steps

Once FunctionGemma is running, you can:

1. **Try custom tools**: Edit `test_functiongemma.py` to add your own function definitions
2. **Fine-tune**: The environment has everything needed for LoRA/QLoRA fine-tuning
3. **Build applications**: Use FunctionGemma as an agent that calls Python functions

## Useful Commands

```bash
# Check environment
which python       # Should be in Nix store
which nvcc        # CUDA compiler
python -c "import torch; print(torch.cuda.is_available())"

# GPU monitoring
watch -n 1 nvidia-smi

# Exit environment
deactivate        # Exit Python venv
exit              # Exit Nix shell (or Ctrl+D)
```
