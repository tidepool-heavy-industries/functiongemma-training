# Fine-Tuning Results & Findings

## Summary

Successfully completed QLoRA fine-tuning on FunctionGemma 270M, but discovered critical issues with function call generation during testing.

## Training Results

✅ **Training Metrics:**
- Loss: 4.09 → 0.70 (excellent convergence)
- Duration: ~92 minutes
- Examples: 900 training + 100 test
- No NaN/Inf issues during training

✅ **Model Merge:**
- Successfully merged LoRA adapters in FP32
- Model size: 1.1GB (FP32)
- Verified: No NaN values, generates text

## Testing Results

❌ **Function Call Generation Issues:**

Both the **base** model and **fine-tuned** model fail to generate proper function call syntax.

### Expected Output:
```
<start_function_call>
call:turn_on_flashlight{}
<end_function_call>
<start_function_call>
call:create_calendar_event{datetime:<escape>2026-11-20T10:00:00<escape>, title:<escape>Review Q4 Strategy Draft<escape>}
<end_function_call>
```

### Actual Output (Fine-tuned):
```
<start_function_call>create_calendar_event
<end_of_turn>
```

### Actual Output (Base):
```
<start_function_call>turn_on_flashlight<escape>turn_off_flashlight<escape><start_function_call>send_email
```

### Comparison Test (5 examples):
- Base model: 1/5 generated any function call syntax (20%)
- Fine-tuned model: 0/5 generated function call syntax (0%)

## Root Cause Analysis

### 1. Chat Template vs Raw Text Mismatch

FunctionGemma was trained using a **chat template** that automatically inserts `call:` prefix:
```jinja
{{-  '<start_function_call>call:' + function['name'] + '{' -}}
```

The model expects structured JSON input with `tool_calls` fields, not raw text tokens.

### 2. Training Data Format

Our training data used raw text with manual `<start_function_call>call:...` tokens, which is correct for text-based fine-tuning, but:
- The base model may not have learned this format well originally
- Fine-tuning amplified the pattern the base model already had (which was incorrect)

### 3. Base Model Limitations

The base `google/functiongemma-270m-it` model itself doesn't generate proper `call:` syntax when prompted with raw text (tested independently).

## Options Moving Forward

### Option 1: Use Chat Template for Inference
**Pros:**
- Leverages the model's original training format
- More likely to generate correct syntax
- Standard approach for FunctionGemma

**Cons:**
- Requires converting prompts to JSON message format
- More complex inference pipeline
- May not work well with Ollama's text-based interface

**Action:** Modify comparison script to use `tokenizer.apply_chat_template()` with proper `tool_calls` structure.

### Option 2: Retrain with More Examples & Longer Duration
**Pros:**
- Could learn the text-based format better
- Might overcome base model's limitations

**Cons:**
- May not fix fundamental architecture mismatch
- Requires more time/compute (~3-4 hours)
- No guarantee of success

**Action:** Filter dataset to ~2000-3000 examples, train for 5-10 epochs.

### Option 3: Switch to Different Base Model
**Pros:**
- Models like `mistralai/Mistral-7B-Instruct-v0.2` or `NousResearch/Hermes-2-Pro-Llama-3-8B` are better for function calling
- Larger context windows
- Better instruction following

**Cons:**
- Requires 4-bit quantization (7B+ models)
- Slower inference
- Different dataset format

**Action:** Research and switch to proven function-calling model.

### Option 4: Deploy as-is for Comparison
**Pros:**
- Complete the original goal of deploying to Ollama
- Can still compare base vs fine-tuned behavior
- Learn about the deployment process

**Cons:**
- Model doesn't work correctly
- Not useful for actual function calling

**Action:** Convert to GGUF, deploy to Ollama, document limitations.

## Recommendation

**Start with Option 1** (chat template inference), then decide:
- If it works → Deploy to Ollama with chat template wrapper
- If it doesn't work → Consider Option 3 (switch to better base model like Hermes-2-Pro)

Option 2 (retrain longer) is least likely to succeed given the base model's limitations.

## Files for Reference

- **Training data:** `data/mobile_actions_curated_train/`
- **Merged model:** `merged_model_fp32/` (1.1GB, FP32)
- **Test results:** `comparison_results.json`
- **Chat template:** `chat_template.txt`
- **Debugging scripts:**
  - `debug_generation.py` - Shows generation behavior
  - `test_without_stops.py` - Tests without stop tokens
  - `test_base_generation.py` - Tests base model
  - `compare_models.py` - Side-by-side comparison

## Next Steps (Pending Your Decision)

1. Try chat template-based inference?
2. Deploy as-is to Ollama for learning purposes?
3. Switch to different base model?
4. Investigate other approaches?
