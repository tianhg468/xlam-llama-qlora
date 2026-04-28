# Performance Optimizations Guide

This guide covers various optimizations to improve training speed, reduce memory usage, and speed up inference.

## Table of Contents
1. [Data Preparation Optimizations](#data-preparation-optimizations)
2. [Training Optimizations](#training-optimizations)
3. [Inference Optimizations](#inference-optimizations)
4. [Memory Optimizations](#memory-optimizations)
5. [Monitoring & Debugging](#monitoring--debugging)

---

## Data Preparation Optimizations

### 1. Use Optimized Data Prep Script

The optimized data prep script includes quality filtering, caching, and statistics:

```bash
python src/data_prep_optimized.py
```

**Benefits:**
- **Quality Filtering**: Removes malformed or extreme examples
- **Caching**: Saves formatted data to avoid reprocessing
- **Length Filtering**: Removes examples exceeding max sequence length
- **Statistics**: Provides dataset insights

**What gets filtered:**
- Missing required fields (query, tools, answers)
- Invalid JSON
- Queries too short (<10 chars) or too long (>2000 chars)
- Too many tools (>50) or answers (>10)
- Examples exceeding max sequence length

### 2. Dataset Statistics

After running optimized data prep, check `data/dataset_statistics.json`:

```json
{
  "total_examples": 54000,
  "query_length": {
    "min": 15,
    "max": 450,
    "mean": 127.3
  },
  "num_tools": {
    "min": 1,
    "max": 12,
    "mean": 3.8
  }
}
```

Use these stats to:
- Tune `max_seq_length` in config
- Understand data distribution
- Identify outliers

### 3. Cached Processing

On subsequent runs, formatted data loads from cache (`data/.formatted_cache.json`):
- First run: ~5-10 minutes
- Cached runs: ~10 seconds

Delete cache to reprocess:
```bash
rm data/.formatted_cache.json
```

---

## Training Optimizations

### 1. Flash Attention 2

Install Flash Attention 2 for 2-3x faster training:

```bash
pip install flash-attn --no-build-isolation
```

The training script automatically uses it if available, falling back to standard attention.

**Speedup**: ~2-3x faster on A100/H100

### 2. Gradient Checkpointing

Already enabled in config:
```yaml
training:
  gradient_checkpointing: true
```

**Benefit**: ~40% memory reduction, ~15% slower training

### 3. Dataloader Optimizations

Add to your training script:

```python
from src.utils.training_utils import setup_dataloader_optimizations

# After creating SFTConfig
sft_config = setup_dataloader_optimizations(sft_config)
```

This sets:
- `dataloader_num_workers: 4` (parallel data loading)
- `dataloader_pin_memory: True` (faster GPU transfer)

**Speedup**: ~10-20% faster data loading

### 4. Optimal Batch Size Finder

Find the largest batch size that fits in memory:

```python
from src.utils.training_utils import find_optimal_batch_size

optimal_bs = find_optimal_batch_size(
    model=model,
    tokenizer=tokenizer,
    sample_text=train_dataset[0]["text"],
    max_seq_length=2048,
)
print(f"Use batch_size: {optimal_bs}")
```

Update your config with the optimal batch size.

### 5. Early Stopping

Stop training when validation loss plateaus:

```python
from src.utils.training_utils import EarlyStoppingCallback

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    callbacks=[EarlyStoppingCallback(patience=3, min_delta=0.001)],
    ...
)
```

**Benefit**: Saves time if model converges early

### 6. Gradient Monitoring

Monitor gradient norms to detect training issues:

```python
from src.utils.training_utils import GradientMonitorCallback, LossLoggingCallback

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    callbacks=[
        GradientMonitorCallback(),
        LossLoggingCallback(),
    ],
    ...
)
```

Logs to W&B:
- `gradient_norm`: Total gradient norm
- `loss_mean_10`: Moving average of last 10 losses

### 7. Mixed Precision Training

Already enabled with bf16:
```yaml
training:
  bf16: true  # Use on A100/H100
  fp16: false # Use on older GPUs (V100, T4)
```

**Speedup**: ~2x faster, ~50% memory reduction

---

## Inference Optimizations

### 1. Merge LoRA Adapter

Merge LoRA weights into base model for faster inference:

```bash
python src/merge_adapter.py
```

This creates a single merged model in `outputs/merged_model/`.

**Benefits:**
- No need for PEFT at inference
- ~20-30% faster inference
- Simpler deployment

**Usage:**
```python
from transformers import AutoModelForCausalLM

# Before (slower)
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "outputs/llama31-8b-xlam-lora")

# After (faster)
model = AutoModelForCausalLM.from_pretrained("outputs/merged_model")
```

### 2. Batched Inference

Process multiple examples at once:

```python
from src.utils.batched_inference import generate_batch

responses = generate_batch(
    model=model,
    tokenizer=tokenizer,
    prompts=["prompt1", "prompt2", "prompt3"],
    batch_size=4,  # Process 4 at a time
    max_new_tokens=512,
)
```

**Speedup**: ~2-4x faster for evaluation

### 3. KV Cache

Enable KV caching for autoregressive generation:

```python
from src.utils.batched_inference import generate_batch_with_cache

responses = generate_batch_with_cache(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    use_cache=True,  # Enable KV cache
)
```

**Speedup**: ~1.5-2x faster generation

### 4. Optimal Inference Batch Size

Find optimal batch size for inference:

```python
from src.utils.batched_inference import estimate_batch_size_for_inference

optimal_bs = estimate_batch_size_for_inference(
    model=model,
    tokenizer=tokenizer,
    sample_prompt=test_data[0]["prompt"],
    max_new_tokens=512,
)
```

---

## Memory Optimizations

### 1. Sequence Length Tuning

Shorter sequences = less memory:

```yaml
training:
  max_seq_length: 1024  # Reduce from 2048 if needed
```

**Memory saved**: ~50% reduction (2048 → 1024)

### 2. Gradient Accumulation

Increase gradient accumulation instead of batch size:

```yaml
training:
  per_device_train_batch_size: 1  # Small batch size
  gradient_accumulation_steps: 16  # Accumulate gradients
  # Effective batch size = 1 * 16 = 16
```

**Memory saved**: ~75% (batch_size 4 → 1)

### 3. Quantization

Use 4-bit quantization (already enabled):
```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
```

**Memory saved**: ~75% reduction vs FP16

### 4. CPU Offloading

For very limited VRAM, offload to CPU:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "20GB", "cpu": "32GB"},  # Limit GPU to 20GB, use 32GB CPU
)
```

**Trade-off**: Much slower but fits on smaller GPUs

---

## Monitoring & Debugging

### 1. Training Summary

Before training, print a summary:

```python
from src.utils.training_utils import print_training_summary

print_training_summary(
    config=config,
    num_train_examples=len(train_dataset),
    num_val_examples=len(val_dataset),
)
```

Output:
```
TRAINING SUMMARY
================
Dataset:
  Train examples: 54,000
  Validation examples: 3,000

Batch Configuration:
  Per-device batch size: 4
  Gradient accumulation steps: 4
  Effective batch size: 16

Training Steps:
  Total steps: 10,125
  Eval every: 100 steps
  Save every: 200 steps

Estimates:
  Estimated training time: ~2.8 hours
```

### 2. W&B Monitoring

The project already logs to W&B. Monitor:
- Training/validation loss
- Learning rate schedule
- Gradient norms (with GradientMonitorCallback)
- GPU memory usage
- Training speed (samples/sec)

### 3. Data Quality Checks

Before training, inspect dataset stats:

```bash
cat data/dataset_statistics.json
```

Check for:
- Extremely long/short examples
- Imbalanced distributions
- Missing data

### 4. Model Inspection

Check trainable parameters:

```python
model.print_trainable_parameters()
```

Output:
```
trainable params: 83,886,080 || all params: 8,118,013,952 || trainable%: 1.03%
```

Expected: ~1-2% trainable for LoRA

---

## Quick Reference: Optimization Checklist

### Before Training:
- [ ] Run optimized data prep: `python src/data_prep_optimized.py`
- [ ] Install Flash Attention 2 (if available)
- [ ] Review dataset statistics
- [ ] Find optimal batch size
- [ ] Print training summary

### During Training:
- [ ] Monitor W&B dashboard
- [ ] Check gradient norms
- [ ] Watch for OOM errors
- [ ] Verify loss is decreasing

### After Training:
- [ ] Merge adapter: `python src/merge_adapter.py`
- [ ] Test inference speed
- [ ] Run evaluations with batched inference

### For Evaluation:
- [ ] Use batched inference
- [ ] Enable KV cache
- [ ] Estimate optimal batch size

---

## Performance Benchmarks

Typical improvements on A100 40GB:

| Optimization | Speedup | Memory Saved |
|-------------|---------|--------------|
| Flash Attention 2 | 2-3x | - |
| bf16 Mixed Precision | 2x | 50% |
| Gradient Checkpointing | -15% | 40% |
| Batched Inference | 3-4x | - |
| Merged Model | 1.3x | - |
| Dataloader workers | 1.15x | - |
| KV Cache | 1.5-2x | - |

**Combined**: Can achieve 5-10x faster training and 5-8x faster inference!

---

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `max_seq_length`
2. Reduce `per_device_train_batch_size`
3. Increase `gradient_accumulation_steps`
4. Enable gradient checkpointing
5. Use CPU offloading

### Slow Training
1. Install Flash Attention 2
2. Use bf16 (if supported)
3. Increase `dataloader_num_workers`
4. Use optimal batch size
5. Reduce logging frequency

### Slow Inference
1. Merge LoRA adapter
2. Use batched inference
3. Enable KV cache
4. Quantize to 4-bit
5. Use optimal inference batch size

---

## Additional Resources

- [HuggingFace Performance Guide](https://huggingface.co/docs/transformers/performance)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [BitsAndBytes Quantization](https://huggingface.co/docs/bitsandbytes)
