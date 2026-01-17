# FunctionGemma 270M Fine-Tuning & Ollama Deployment Plan
## Implementation Strategy for GTX 1660 SUPER (6GB VRAM, FP16-only)

---

## Executive Summary

This plan details the end-to-end pipeline for:
1. **Fine-tuning** FunctionGemma 270M using QLoRA on GTX 1660 SUPER (6GB VRAM, FP16-only)
2. **Converting** the fine-tuned model to GGUF format via manual llama.cpp pipeline
3. **Deploying** to Ollama for production inference

**Hardware Constraints:**
- GPU: GTX 1660 SUPER, 6GB VRAM, FP16 tensor cores only (NO BF16)
- RAM: 26GB allocated to WSL2
- Disk: 6.5GB free (98% full - requires cleanup)
- CUDA: 12.8 installed and verified working

**Critical Design Decisions:**
- **Framework**: Unsloth (only framework supporting FP16-only GPUs for Gemma)
- **Quantization**: QLoRA 4-bit (mandatory for 6GB VRAM)
- **Batch Strategy**: batch_size=1, gradient_accumulation_steps=8-16
- **Conversion**: Manual llama.cpp pipeline (more robust than Unsloth auto-export)
- **Target Format**: GGUF Q4_K_M (optimal for 6GB GPU inference)

---

## Phase 1: Environment Setup & Disk Management

### 1.1 Disk Space Assessment and Cleanup

**Current Status:** 6.5GB free (98% full) - insufficient for training

**Required Space Breakdown:**
- PyTorch + CUDA libs: ~5GB
- Unsloth + dependencies: ~2GB
- Base model download: ~1GB (FunctionGemma 270M 4-bit)
- Training dataset: ~100-500MB
- Training checkpoints: ~500MB (LoRA adapters)
- Merged model (FP16): ~1GB
- GGUF conversion workspace: ~2GB
- Final quantized model: ~300-500MB
- **Total Required: ~12-15GB**

**Action Items:**
1. Identify large files/directories to clean:
   ```bash
   du -h --max-depth=1 /home/inanna | sort -hr | head -20
   du -h --max-depth=1 /home/inanna/dev | sort -hr | head -20
   ```
2. Check for cached files:
   - `~/.cache/huggingface` (model cache)
   - `~/.cache/pip` (pip cache)
   - `/tmp` (temporary files)
   - Nix store if overgrown
3. Remove mistral.rs artifacts (no longer needed):
   - `/home/inanna/dev/mistral.rs/target` directory (Rust build artifacts - likely 5-10GB)
   - Any downloaded mistral.rs models
4. Target: Free up at least 10GB additional space

### 1.2 Python Environment Strategy

**Decision: Use Python venv (NOT Conda) within Nix shell**

**Rationale:**
- Nix flake already provides CUDA 12.8 and system dependencies
- venv is lightweight and avoids Conda's large footprint
- Better integration with Nix-provided CUDA toolkit
- Explicit control over PyTorch installation

**Implementation:**
```bash
cd /home/inanna/dev/gemma

# Create venv inside Nix shell
nix develop  # Enter the Nix environment
python3.11 -m venv .venv
source .venv/bin/activate

# Verify CUDA visibility
echo $CUDA_HOME  # Should point to Nix CUDA path
nvidia-smi       # Verify GPU detection
```

### 1.3 Core ML Stack Installation

**PyTorch Installation (CUDA 12.8 compatible):**
```bash
# Install PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected Output:**
```
CUDA available: True
GPU: NVIDIA GeForce GTX 1660 SUPER
```

### 1.4 Unsloth Installation for FP16-Only GPU

**Critical Note:** Unsloth has special builds for different GPU architectures. GTX 1660 SUPER is Turing (CC 7.5), which requires the "ampere-torch240" build.

```bash
# Install Unsloth for Turing architecture (GTX 1660 SUPER)
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Install supporting PEFT libraries
pip install --no-deps \
    packaging ninja einops \
    xformers trl peft accelerate bitsandbytes

# Install Transformers and Datasets
pip install transformers datasets
```

**Verification:**
```python
from unsloth import FastLanguageModel
import torch
print(f"Unsloth imported successfully")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU Compute Capability: {torch.cuda.get_device_capability()}")
```

**Expected:** Should print `(7, 5)` for GTX 1660 SUPER

### 1.5 Ollama Installation

```bash
# Install Ollama on WSL2
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Start Ollama service (will auto-start on subsequent boots)
ollama serve &

# Test with a small model
ollama run gemma:2b "Hello"
```

**Expected:** Ollama should detect the GPU and load models to VRAM

---

## Phase 2: Dataset Preparation

### 2.1 Dataset Selection Strategy

**Option A: Use Google's Mobile Actions Dataset (Recommended for First Training)**

**Pros:**
- Official Google dataset designed for FunctionGemma
- Pre-formatted examples
- Known good results (58% → 85% accuracy)
- ~5,000 training examples

**Cons:**
- Domain-specific (Android mobile actions)
- May not match user's target use case

**Implementation:**
```python
from datasets import load_dataset

# Load Mobile Actions dataset
dataset = load_dataset("google/mobile-actions", split="train")

# Inspect format
print(dataset[0])
```

**Option B: Create Custom Function Calling Dataset**

**Pros:**
- Tailored to specific use case
- Full control over function definitions and examples
- Can use GPT-4/Claude for synthetic generation

**Cons:**
- Requires significant upfront work
- Risk of low-quality synthetic data
- Needs careful prompt engineering

**Recommendation:** Start with Mobile Actions to validate the pipeline, then create custom dataset once workflow is proven.

### 2.2 FunctionGemma Prompt Format

FunctionGemma requires a **very specific** prompt structure with control tokens:

```
<start_of_turn>developer
You are an expert function calling AI assistant. Current date: 2025-01-16. You have access to the following functions:

<start_function_declaration>function_name1
Description of function
Parameters:
  param1 (type): description
  param2 (type): description
<end_function_declaration>

<start_function_declaration>function_name2
...
<end_function_declaration>
<end_of_turn>
<start_of_turn>user
[User query/instruction]
<end_of_turn>
<start_of_turn>model
<start_function_call>
call:function_name{param1:<escape>value1<escape>, param2:<escape>value2<escape>}
<end_function_call>
[Optional natural language response]
<end_of_turn>
```

**Critical Rules:**
1. All string values MUST use `<escape>` delimiters
2. System/developer turn must include function declarations
3. Model turn must use `call:` prefix before function name
4. Control tokens are literal strings (not special tokens in tokenizer)

### 2.3 Dataset Preprocessing Script

Create `prepare_dataset.py`:

```python
from datasets import load_dataset, Dataset
import json

def format_functiongemma_prompt(example):
    """
    Convert example to FunctionGemma format.

    Expected input format:
    {
        "functions": [...],  # List of function definitions
        "user_query": "...",
        "function_call": {"name": "...", "parameters": {...}},
        "response": "..."  # Optional
    }
    """

    # Build developer turn with function declarations
    developer_turn = "You are an expert function calling AI assistant. "
    developer_turn += "Current date: 2025-01-16. "
    developer_turn += "You have access to the following functions:\n\n"

    for func in example["functions"]:
        developer_turn += "<start_function_declaration>" + func["name"] + "\n"
        developer_turn += func["description"] + "\n"
        developer_turn += "Parameters:\n"
        for param_name, param_info in func["parameters"].items():
            developer_turn += f"  {param_name} ({param_info['type']}): {param_info['description']}\n"
        developer_turn += "<end_function_declaration>\n\n"

    # Build user turn
    user_turn = example["user_query"]

    # Build model turn
    model_turn = "<start_function_call>\n"
    func_call = example["function_call"]
    model_turn += f"call:{func_call['name']}{{"

    params = []
    for param_name, param_value in func_call["parameters"].items():
        # Escape string values
        params.append(f"{param_name}:<escape>{param_value}<escape>")

    model_turn += ", ".join(params)
    model_turn += "}\n<end_function_call>\n"

    if "response" in example and example["response"]:
        model_turn += example["response"]

    # Assemble full prompt
    full_prompt = (
        f"<start_of_turn>developer\n{developer_turn}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_turn}<end_of_turn>\n"
        f"<start_of_turn>model\n{model_turn}<end_of_turn>"
    )

    return {"text": full_prompt}

# Load and format dataset
dataset = load_dataset("google/mobile-actions", split="train")
formatted_dataset = dataset.map(format_functiongemma_prompt)

# Save to disk (in Linux filesystem for performance)
formatted_dataset.save_to_disk("/home/inanna/dev/gemma/data/mobile_actions_formatted")

print(f"Formatted {len(formatted_dataset)} examples")
print("\nSample:")
print(formatted_dataset[0]["text"])
```

### 2.4 Dataset Storage Location

**CRITICAL:** Store datasets in Linux native filesystem, NOT `/mnt/c/`

```bash
mkdir -p /home/inanna/dev/gemma/data
# All datasets go here
```

**Why:** Accessing Windows filesystem from WSL2 incurs 10-50x I/O penalty due to 9P protocol translation. This will starve the GPU during training.

---

## Phase 3: Fine-Tuning with QLoRA

### 3.1 Training Configuration for GTX 1660 SUPER

**QLoRA Hyperparameters (optimized for 6GB VRAM):**

```python
from transformers import TrainingArguments
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                                  # LoRA rank (16 or 32)
    lora_alpha=32,                         # Scaling factor (typically 2*r)
    target_modules=[                       # Which layers to adapt
        "q_proj",
        "v_proj",
        # Optionally add: "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,                     # Regularization
    bias="none",                           # Don't train bias terms
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="/home/inanna/dev/gemma/outputs",

    # Batch size strategy for 6GB VRAM
    per_device_train_batch_size=1,         # CRITICAL: Keep at 1 for 6GB VRAM
    gradient_accumulation_steps=8,         # Effective batch size = 1*8 = 8

    # Learning rate and schedule
    learning_rate=1e-5,                    # Conservative for FP16
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # Training duration
    num_train_epochs=3,                    # Start with 3, can increase
    max_steps=-1,                          # Train for full epochs

    # Optimization
    optim="adamw_torch",                   # Standard optimizer
    weight_decay=0.01,
    max_grad_norm=1.0,                     # Gradient clipping (CRITICAL for FP16)

    # Memory optimization
    gradient_checkpointing=True,           # Trades compute for memory
    fp16=True,                             # FP16 training (GTX 1660 SUPER)
    bf16=False,                            # NOT available on GTX 1660 SUPER

    # Logging and checkpointing
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,                    # Keep only 2 checkpoints (disk space)

    # Evaluation
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Misc
    report_to="none",                      # Disable WandB/TensorBoard to save memory
    seed=42,
)
```

### 3.2 Training Script

Create `train_functiongemma.py`:

```python
import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Model configuration
max_seq_length = 2048  # FunctionGemma supports up to 32K, but we limit for VRAM
model_name = "google/functiongemma-270m-it"

# Load model in 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect (will use FP16 for GTX 1660 SUPER)
    load_in_4bit=True,  # CRITICAL for 6GB VRAM
    device_map="auto",
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
)

# Load formatted dataset
dataset = load_from_disk("/home/inanna/dev/gemma/data/mobile_actions_formatted")

# Split into train/eval
train_test = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# Training arguments
training_args = TrainingArguments(
    output_dir="/home/inanna/dev/gemma/outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=42,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

# Start training
print("\n=== Starting Training ===\n")
trainer.train()

# Save final model
print("\n=== Saving Model ===\n")
model.save_pretrained("/home/inanna/dev/gemma/checkpoints/final_lora")
tokenizer.save_pretrained("/home/inanna/dev/gemma/checkpoints/final_lora")

print("Training complete!")
```

### 3.3 Training Execution

```bash
cd /home/inanna/dev/gemma
source .venv/bin/activate
python train_functiongemma.py
```

**Expected Behavior:**
- Initial model loading: ~1-2 minutes
- Dataset loading: ~10-30 seconds
- Training speed: ~50-100 tokens/sec on GTX 1660 SUPER
- VRAM usage: 3-5GB during training
- Training time: ~1-3 hours for 3 epochs (depends on dataset size)

**Monitoring:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Red Flags:**
- OOM errors → Reduce `per_device_train_batch_size` or `max_seq_length`
- NaN/Inf in loss → Reduce `learning_rate` or increase `max_grad_norm`
- Very slow training (<20 tok/s) → Check disk I/O (dataset location)

---

## Phase 4: Model Merging and Export

### 4.1 Merge LoRA Adapters into Base Model

After training, we have:
- Base model (FunctionGemma 270M in 4-bit)
- LoRA adapters (~50-200MB)

We need to merge these into a single FP16 model for conversion.

Create `merge_model.py`:

```python
import torch
from unsloth import FastLanguageModel

print("Loading model and adapters...")

# Load the base model and LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/inanna/dev/gemma/checkpoints/final_lora",  # Path to saved LoRA
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Merge adapters into base model
print("Merging LoRA adapters into base model...")
model = FastLanguageModel.for_inference(model)  # Prepare for inference

# Save merged model in FP16
output_dir = "/home/inanna/dev/gemma/merged_model_fp16"
print(f"Saving merged model to {output_dir}...")

model.save_pretrained_merged(
    output_dir,
    tokenizer,
    save_method="merged_16bit",  # Save as FP16
)

print("Model merged and saved successfully!")
print(f"Location: {output_dir}")
```

**Execution:**
```bash
python merge_model.py
```

**Expected Output:**
- Merged model directory: `/home/inanna/dev/gemma/merged_model_fp16/`
- Size: ~1GB (FP16)
- Contains: `model.safetensors`, `config.json`, `tokenizer.json`, etc.

### 4.2 Test Merged Model (Optional but Recommended)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/inanna/dev/gemma/merged_model_fp16"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Test with a function calling prompt
prompt = """<start_of_turn>developer
You are an expert function calling AI assistant.
<start_function_declaration>get_weather
Get weather for a location
Parameters:
  location (string): City name
<end_function_declaration>
<end_of_turn>
<start_of_turn>user
What's the weather in San Francisco?
<end_of_turn>
<start_of_turn>model
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

**Expected Output:** Should generate a proper function call like:
```
<start_function_call>
call:get_weather{location:<escape>San Francisco<escape>}
<end_function_call>
```

---

## Phase 5: GGUF Conversion via llama.cpp

### 5.1 Install llama.cpp

```bash
cd /home/inanna/dev

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support
make LLAMA_CUDA=1 -j$(nproc)

# Verify build
./llama-cli --version
```

**Expected:** Should compile successfully and show version.

### 5.2 Convert HuggingFace Model to GGUF

```bash
cd /home/inanna/dev/llama.cpp

# Install Python dependencies for conversion script
pip install -r requirements.txt

# Convert merged model to GGUF (FP16)
python convert_hf_to_gguf.py \
    /home/inanna/dev/gemma/merged_model_fp16 \
    --outfile /home/inanna/dev/gemma/functiongemma-270m-finetuned-f16.gguf \
    --outtype f16

# Check file size
ls -lh /home/inanna/dev/gemma/functiongemma-270m-finetuned-f16.gguf
```

**Expected:**
- File size: ~550-600MB (FP16 GGUF)
- No errors during conversion

### 5.3 Quantize to Q4_K_M

```bash
cd /home/inanna/dev/llama.cpp

# Quantize to 4-bit (Q4_K_M - balanced quality/size)
./llama-quantize \
    /home/inanna/dev/gemma/functiongemma-270m-finetuned-f16.gguf \
    /home/inanna/dev/gemma/functiongemma-270m-finetuned-q4_k_m.gguf \
    Q4_K_M

# Verify quantized model
ls -lh /home/inanna/dev/gemma/functiongemma-270m-finetuned-q4_k_m.gguf
```

**Expected:**
- File size: ~150-200MB (Q4_K_M quantized)
- Quantization should complete in <1 minute

### 5.4 Test GGUF Model with llama.cpp (Optional)

```bash
./llama-cli \
    -m /home/inanna/dev/gemma/functiongemma-270m-finetuned-q4_k_m.gguf \
    -p "<start_of_turn>user\nHello, what can you do?<end_of_turn>\n<start_of_turn>model\n" \
    -n 100 \
    --gpu-layers 999  # Offload all layers to GPU
```

**Expected:** Should generate coherent text on GPU with ~50-70 tok/s.

---

## Phase 6: Ollama Deployment

### 6.1 Create Modelfile

The Modelfile defines how Ollama loads and interacts with the model. **The TEMPLATE section is CRITICAL** - it must match the training format exactly.

Create `/home/inanna/dev/gemma/Modelfile`:

```dockerfile
# Modelfile for FunctionGemma 270M Fine-tuned

FROM /home/inanna/dev/gemma/functiongemma-270m-finetuned-q4_k_m.gguf

# System prompt template - MUST match training format
TEMPLATE """{{ if .System }}<start_of_turn>developer
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

# Stop tokens - CRITICAL for proper generation termination
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_function_call>"
PARAMETER stop "<end_function_call>"

# Generation parameters
PARAMETER temperature 0.3        # Lower for function calling (more deterministic)
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Context window
PARAMETER num_ctx 4096           # Can go up to 32768 for FunctionGemma

# System message
SYSTEM """You are an expert function calling AI assistant. Current date: 2025-01-16."""
```

### 6.2 Ingest Model into Ollama

```bash
cd /home/inanna/dev/gemma

# Create the model in Ollama
ollama create functiongemma-finetuned -f Modelfile

# Verify model was created
ollama list | grep functiongemma
```

**Expected Output:**
```
NAME                          ID              SIZE      MODIFIED
functiongemma-finetuned       abc123def456    180 MB    2 minutes ago
```

### 6.3 Test Ollama Model

**Test 1: Basic Interaction**
```bash
ollama run functiongemma-finetuned "Hello! Can you help me with function calling?"
```

**Test 2: Function Calling**
```bash
ollama run functiongemma-finetuned "What's the weather like in Seattle?" --system "You have access to the following functions:

<start_function_declaration>get_weather
Get current weather for a location
Parameters:
  location (string): City name
<end_function_declaration>"
```

**Expected Output:** Should generate a proper function call:
```
<start_function_call>
call:get_weather{location:<escape>Seattle<escape>}
<end_function_call>
```

### 6.4 Performance Verification

```bash
# Monitor GPU usage during inference
watch -n 0.5 nvidia-smi

# Test with longer prompt to stress test
ollama run functiongemma-finetuned "$(cat test_prompt.txt)"
```

**Expected Performance:**
- **Prompt processing:** ~500-1000 tok/s
- **Generation:** 50-65 tok/s on Q4_K_M
- **VRAM usage:** ~2.4GB
- **GPU utilization:** 70-95%

**If performance is poor:**
- Check if layers are offloaded to CPU (indicates VRAM shortage)
- Try more aggressive quantization (Q4_K_S or Q3_K_M)
- Verify GPU is being used: `ollama ps` during inference

---

## Phase 7: Documentation and Validation

### 7.1 Create Comprehensive Documentation

Files to create:

1. **`TRAINING_GUIDE.md`** - Complete training walkthrough
2. **`DEPLOYMENT_GUIDE.md`** - Ollama setup and usage
3. **`TROUBLESHOOTING.md`** - Common issues and fixes
4. **`FUNCTION_CALLING_GUIDE.md`** - How to use FunctionGemma for various tasks
5. **`HARDWARE_NOTES.md`** - GTX 1660 SUPER specific considerations

### 7.2 Benchmark Fine-tuned Model

Create `benchmark.py`:

```python
import json
from datasets import load_dataset

# Load test set
test_data = load_dataset("google/mobile-actions", split="test")

# Run inference on each example
correct = 0
total = 0

for example in test_data:
    # Format prompt
    prompt = format_test_prompt(example)

    # Run through Ollama
    result = call_ollama(prompt)

    # Parse function call
    predicted_function = parse_function_call(result)
    expected_function = example["function_call"]

    # Check accuracy
    if predicted_function == expected_function:
        correct += 1
    total += 1

accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}%")
```

**Target Accuracy:** 80-90% on Mobile Actions test set (baseline was 58%)

### 7.3 Create Example Applications

**Example 1: Function Calling API Server**
```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
OLLAMA_API = "http://localhost:11434/api/generate"

@app.route("/function-call", methods=["POST"])
def function_call():
    data = request.json

    # Format prompt with function definitions and user query
    prompt = build_functiongemma_prompt(
        functions=data["functions"],
        user_query=data["query"]
    )

    # Call Ollama
    response = requests.post(OLLAMA_API, json={
        "model": "functiongemma-finetuned",
        "prompt": prompt,
        "stream": False
    })

    # Parse function call
    function_call = parse_function_call(response.json()["response"])

    return jsonify(function_call)

if __name__ == "__main__":
    app.run(port=5000)
```

**Example 2: CLI Tool for Quick Function Calling**

---

## Phase 8: Optimization and Iteration

### 8.1 If Training Results Are Poor

**Diagnostics:**
1. Check training loss curve - did it converge?
2. Inspect generated outputs - what patterns are wrong?
3. Validate dataset format - are control tokens correct?
4. Check template matching - does Modelfile match training?

**Remedies:**
- **Increase training epochs** (3 → 5)
- **Expand LoRA targets** (add k_proj, o_proj, gate_proj)
- **Increase LoRA rank** (16 → 32 or 64)
- **Adjust learning rate** (try 5e-6 or 2e-5)
- **Clean dataset** (remove low-quality examples)
- **Add more training data** (synthetic generation)

### 8.2 If Inference Performance Is Poor

**VRAM Limited:**
- Try Q3_K_M quantization (smaller but lower quality)
- Reduce num_ctx in Modelfile (4096 → 2048)
- Use Q4_K_S instead of Q4_K_M

**Speed Limited:**
- Verify all layers on GPU: `ollama ps` should show full VRAM usage
- Check WSL2 memory allocation in `.wslconfig`
- Update Ollama to latest version

### 8.3 Advanced Optimizations

**Context Extension (RoPE Scaling):**
If you need longer context (>4096 tokens), use RoPE scaling during training:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=8192,  # Extend context
    rope_scaling=2.0,     # Scale factor
    ...
)
```

**Multi-GPU Support (if upgrading hardware):**
- Ollama supports multi-GPU automatically
- Training can use `CUDA_VISIBLE_DEVICES=0,1`

---

## Success Criteria

✅ **Environment Setup:**
- PyTorch detects GTX 1660 SUPER
- Unsloth installs and loads models in 4-bit
- Ollama runs and detects GPU

✅ **Training:**
- Training completes without OOM errors
- Final loss < 1.0 (typically 0.3-0.7 for function calling)
- VRAM usage stays under 6GB

✅ **Conversion:**
- LoRA adapters merge successfully
- GGUF conversion completes without errors
- Quantized model size ~150-200MB

✅ **Deployment:**
- Ollama loads model and uses GPU (check with nvidia-smi)
- Inference speed: 50+ tok/s
- Model generates valid function calls with correct format

✅ **Quality:**
- Function calling accuracy: >80% on test set
- No hallucinations in function parameters
- Proper stop token behavior (doesn't ramble)

---

## Risk Mitigation

### Risk 1: OOM During Training
**Probability:** Medium
**Impact:** High (blocks training)
**Mitigation:**
- Start with per_device_batch_size=1
- Reduce max_seq_length to 1024 if needed
- Enable gradient_checkpointing
- Close all other applications

### Risk 2: Poor Model Quality After Training
**Probability:** Medium
**Impact:** Medium (requires retraining)
**Mitigation:**
- Validate dataset format thoroughly before training
- Start with proven dataset (Mobile Actions)
- Monitor loss during training
- Save checkpoints frequently

### Risk 3: Template Mismatch in Ollama
**Probability:** Medium
**Impact:** High (model unusable)
**Mitigation:**
- Document exact training template
- Test merged model before GGUF conversion
- Verify Modelfile template matches training exactly

### Risk 4: Disk Space Exhaustion
**Probability:** High (currently 98% full)
**Impact:** Critical (blocks all operations)
**Mitigation:**
- Clean up mistral.rs artifacts FIRST (priority)
- Monitor disk usage throughout pipeline
- Delete intermediate files after each phase

### Risk 5: FP16 Numerical Instability
**Probability:** Low-Medium
**Impact:** Medium (training divergence)
**Mitigation:**
- Use conservative learning rate (1e-5)
- Enable gradient clipping (max_grad_norm=1.0)
- Monitor for NaN/Inf in loss
- Unsloth handles most FP16 issues automatically

---

## Timeline Estimate (Conservative)

| Phase | Duration | Notes |
|-------|----------|-------|
| Disk cleanup | 15-30 min | Manual inspection needed |
| Environment setup | 30-60 min | PyTorch install is slow |
| Dataset preparation | 30-60 min | Depends on data source |
| Fine-tuning | 2-4 hours | Depends on dataset size |
| Merging & conversion | 15-30 min | Mostly automated |
| GGUF quantization | 5-10 min | Fast |
| Ollama deployment | 10-20 min | Testing takes time |
| Documentation | 1-2 hours | Thorough docs |
| **Total** | **5-9 hours** | Excludes troubleshooting |

**Note:** Training is the longest phase and can run unattended.

---

## Next Steps

Upon approval of this plan:

1. **Execute Phase 1:** Clean disk and set up environment
2. **Checkpoint:** Verify GPU detection and Unsloth installation
3. **Execute Phase 2:** Prepare dataset
4. **Checkpoint:** Inspect formatted examples
5. **Execute Phase 3:** Run training (can run overnight)
6. **Checkpoint:** Evaluate training metrics
7. **Execute Phases 4-6:** Convert and deploy to Ollama
8. **Final validation:** Test function calling accuracy

Would you like me to proceed with implementation?
