# FunctionGemma Fine-Tuning Project - Current State

**Last Updated:** 2026-01-16
**Status:** ✅ COMPLETE - Model trained, tested, and deployed

## Project Overview

Fine-tuned Google's FunctionGemma 270M model for better function calling accuracy on a GTX 1660 SUPER (6GB VRAM). Successfully trained with QLoRA, converted to GGUF, and deployed to Ollama.

## Quick Start - Use the Model

### Interactive Testing (Recommended)
```bash
cd /home/inanna/dev/gemma
./run_interactive.sh
```

This launches an interactive prompt where you can test queries and see parsed function calls.

### Programmatic Use
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/home/inanna/dev/gemma/merged_model_fp32",
    torch_dtype=torch.float32,  # IMPORTANT: Must use FP32 for Gemma3
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/home/inanna/dev/gemma/merged_model_fp32")

# Define tools in JSON schema format
tools = [{
    "type": "function",
    "function": {
        "name": "your_function",
        "description": "What it does",
        "parameters": {"type": "object", "properties": {...}, "required": [...]}
    }
}]

# Query
messages = [{"role": "user", "content": "your query"}]

# CRITICAL: Use apply_chat_template with tools parameter
inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,  # <-- THIS IS REQUIRED
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])
```

## Directory Structure

```
/home/inanna/dev/gemma/
├── merged_model_fp32/                          # Fine-tuned HuggingFace model (1.1GB)
├── functiongemma-270m-finetuned-Q4_K_M.gguf   # Quantized GGUF (242MB)
├── functiongemma-270m-finetuned-fp32.gguf     # FP32 GGUF (1.1GB)
├── checkpoints/final_lora/                     # LoRA adapters
├── data/
│   ├── mobile_actions_curated_train/          # 900 training examples
│   └── mobile_actions_curated_test/           # 100 test examples
│
├── interactive_test.py                         # Interactive tester with parsed output
├── run_interactive.sh                          # Easy launcher
├── test_proper_inference.py                    # Demonstrates correct usage
├── compare_models_proper.py                    # Base vs fine-tuned comparison
├── Modelfile                                   # Ollama configuration
├── FINAL_SUMMARY.md                            # Complete documentation
├── FINDINGS.md                                 # Technical analysis
└── CLAUDE.md                                   # This file
```

## Key Scripts

### Ready to Use
- **`./run_interactive.sh`** - Interactive testing with parsed output
- **`python test_proper_inference.py`** - Test on 3 examples
- **`python compare_models_proper.py`** - Compare base vs fine-tuned (10 examples)

### Training/Development
- **`prepare_dataset.py`** - Format dataset from HuggingFace
- **`filter_dataset.py`** - Quality filter (9654 → 900 examples)
- **`train_functiongemma_safe.py`** - QLoRA training script
- **`merge_model_fp32.py`** - Merge LoRA adapters in FP32

## Model Files

| File | Size | Purpose |
|------|------|---------|
| `merged_model_fp32/` | 1.1GB | Fine-tuned model (Python/transformers) |
| `functiongemma-270m-finetuned-Q4_K_M.gguf` | 242MB | Quantized (llama.cpp/Ollama) |
| `functiongemma-270m-finetuned-fp32.gguf` | 1.1GB | FP32 GGUF (llama.cpp) |

**Ollama model:** `functiongemma-finetuned` (242MB, deployed)

## Training Results

- **Loss:** 4.09 → 0.70 (excellent convergence)
- **Duration:** ~92 minutes on GTX 1660 SUPER
- **Dataset:** 900 examples (filtered from 9,654)
- **Method:** QLoRA (rank=16, alpha=32, FP32 training)

## Performance

### Comparison (10 test examples)
- **Base model:** 100% function call success, ~80% parameter accuracy
- **Fine-tuned:** 100% function call success, ~95% parameter accuracy

### Improvements from Fine-Tuning
- ✅ More accurate function name selection
- ✅ Better parameter parsing (e.g., properly splits first/last names)
- ✅ Fewer hallucinated parameters
- ✅ Better handling of parallel function calls

## Critical Technical Details

### 1. Gemma3 Requires FP32
- **FP16 causes NaN values** - training and inference must use FP32
- Quantization to Q4_K_M is safe AFTER merging in FP32
- Why: Architecture-specific issue with Gemma3

### 2. Must Use Chat Template with Tools
**Wrong (doesn't work):**
```python
prompt = "<start_of_turn>user\nTurn on flashlight<end_of_turn>\n"
inputs = tokenizer(prompt, return_tensors="pt")
```

**Correct (works):**
```python
messages = [{"role": "user", "content": "Turn on flashlight"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,  # <-- REQUIRED
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
```

### 3. Special Token Format
Model outputs use `<escape>` tokens for strings:
```
call:create_calendar_event{datetime:<escape>2026-01-17T15:00:00<escape>,title:<escape>Meeting<escape>}
```

Parse by removing `<escape>` tokens.

### 4. Stop Tokens
Important for preventing hallucination:
- `<end_function_call>` - Stop after function call
- `<start_function_response>` - Don't let model hallucinate responses
- `<end_of_turn>` - Standard turn boundary

## Available Functions (Built-in)

The model was trained on:
1. `turn_on_flashlight` / `turn_off_flashlight`
2. `create_calendar_event(datetime, title)`
3. `create_contact(first_name, last_name, phone_number, email)`
4. `send_email(to, subject, body)`
5. `open_wifi_settings`
6. `show_map(query)`

You can define NEW functions by providing them in the `tools` parameter.

## Environment Setup

```bash
cd /home/inanna/dev/gemma

# Activate environment
source .venv/bin/activate
source .nix-lib-path

# Now Python imports will work
python your_script.py
```

**Why:** PyTorch needs the GCC stdcxx library path from nix.

## Known Limitations

### Ollama CLI Doesn't Work
```bash
$ ollama run functiongemma-finetuned "Turn on flashlight"
> I apologize, but I cannot assist... [generic response]
```

**Why:** Ollama CLI doesn't support the structured tool format FunctionGemma needs.

**Workaround:** Use Python with transformers, or Ollama API with custom formatting.

### 270M Parameter Limitations
- ❌ Complex multi-step reasoning
- ❌ Ambiguity resolution without explicit hints
- ✅ Direct query → function mapping
- ✅ Parameter extraction from natural language
- ✅ Parallel function calls

## Troubleshooting

### "libstdc++.so.6 not found"
```bash
source .nix-lib-path
```

### "Model generates padding tokens"
You're not using FP32. Load with:
```python
torch_dtype=torch.float32  # NOT float16
```

### "No function calls generated"
You forgot the `tools` parameter in `apply_chat_template()`.

### "Invalid function syntax"
Make sure your tools are in the correct JSON schema format (see Quick Start).

## Hardware Requirements

### Training
- GPU: GTX 1660 SUPER (6GB VRAM) - worked perfectly
- RAM: ~16GB
- Disk: ~75GB free
- Time: ~90 minutes

### Inference
- **FP32 model:** 2.1GB GPU memory
- **Q4_K_M:** ~500MB GPU memory
- **CPU-only:** Works but slow (~5-10 sec/query)

## Next Steps (Optional)

### Improve the Model
1. **More data:** Increase to 2000-3000 examples
2. **Domain-specific:** Fine-tune on your own functions
3. **Add reasoning:** Include "thinking" tokens in training data
4. **Larger model:** Try Qwen 2.5 1.5B for better reasoning

### Production Deployment
1. **API wrapper:** Create FastAPI endpoint around the model
2. **Function executor:** Parse output and actually call functions
3. **Multi-turn:** Add conversation state management
4. **Batch inference:** Process multiple queries efficiently

## References

- **Model:** `google/functiongemma-270m-it`
- **Dataset:** `Salesforce/mobile_actions` (filtered)
- **Framework:** Unsloth + transformers + PEFT
- **Quantization:** llama.cpp

## Important Files for Future Sessions

Must read:
- **`FINAL_SUMMARY.md`** - Comprehensive project documentation
- **`FINDINGS.md`** - Why initial testing failed + fixes
- **`interactive_test.py`** - Working example with parsing

Configuration:
- **`Modelfile`** - Ollama deployment config
- **`flake.nix`** - Nix environment setup
- **`.nix-lib-path`** - Library paths for PyTorch

## Commands Cheat Sheet

```bash
# Run interactive tester
./run_interactive.sh

# Test on examples
source .venv/bin/activate && source .nix-lib-path
python test_proper_inference.py

# Compare base vs fine-tuned
python compare_models_proper.py

# Use with Ollama (limited - CLI doesn't work well)
ollama list  # Check it's installed
# Use Python instead for actual function calling

# Convert new models to GGUF
cd /home/inanna/dev/llama.cpp
python convert_hf_to_gguf.py <model_path> --outfile output.gguf --outtype f32

# Quantize GGUF
./build/bin/llama-quantize input.gguf output.gguf Q4_K_M
```

## Git Status

```
M flake.nix           # Added GCC library path
?? CLAUDE.md          # This file
?? FINAL_SUMMARY.md   # Documentation
?? FINDINGS.md        # Technical analysis
?? *.py               # All scripts
?? merged_model_fp32/ # Fine-tuned model
?? *.gguf             # Converted models
?? Modelfile          # Ollama config
```

## Contact / Issues

For issues with:
- **FunctionGemma:** Check Google's model card on HuggingFace
- **Ollama:** https://github.com/ollama/ollama
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **This project:** Read FINAL_SUMMARY.md and FINDINGS.md

---

**Ready to use!** Run `./run_interactive.sh` to start testing. 🚀
