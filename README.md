# xLAM Llama 3.1 qLoRA Fine-tuning

Fine-tune `meta-llama/Llama-3.1-8B-Instruct` on `Salesforce/xlam-function-calling-60k` using qLoRA to produce a small specialist model that outperforms the base model at function calling.

## Overview

This project demonstrates production-quality LLM fine-tuning for function calling using:

- **Base Model**: Llama 3.1 8B Instruct (gated on HuggingFace)
- **Dataset**: xLAM Function Calling 60k (gated, CC-BY-4.0 license)
- **Method**: qLoRA (4-bit NF4 quantization + LoRA adapters)
- **Framework**: HuggingFace Transformers, PEFT, TRL, bitsandbytes

The trained model achieves improved performance on function-calling tasks while requiring only ~24GB VRAM.

## Hardware Requirements

- **Recommended**: Single NVIDIA GPU with 24GB VRAM (RTX 3090/4090 or A100)
- **Also supported**: A100 40GB/80GB
- **Minimum VRAM**: ~22GB (with gradient checkpointing and batch size optimizations)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: ~50GB free space for models, datasets, and outputs

## Project Structure

```
xlam-llama-qlora/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── Makefile                       # Build targets
├── .gitignore
├── configs/
│   └── config.yaml                # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── data_prep.py               # Dataset formatting
│   ├── train.py                   # qLoRA training
│   ├── eval.py                    # Model evaluation
│   └── inference.py               # Single-query inference
├── data/                          # Generated (gitignored)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── outputs/                       # Generated (gitignored)
    ├── llama31-8b-xlam-lora/      # Saved LoRA adapter
    └── eval_results.json          # Evaluation metrics
```

## Setup

### 1. Prerequisites

Ensure you have Python 3.10+ and CUDA 11.8+ installed.

### 2. Accept HuggingFace Licenses

Both the base model and dataset are gated and require license acceptance:

1. **Llama 3.1**: Visit [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and accept the license
2. **xLAM Dataset**: Visit [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) and accept the license

### 3. Login to HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your HuggingFace token when prompted.

### 4. Install Dependencies

```bash
cd xlam-llama-qlora
pip install -r requirements.txt
```

### 5. Setup Weights & Biases

The project uses W&B for training metrics and visualization:

```bash
wandb login
```

Enter your API key from https://wandb.ai/authorize (sign up for free if needed).

To disable W&B, edit `configs/config.yaml` and change `report_to: "wandb"` to `report_to: "none"`.

### 6. (Optional) Install Flash Attention 2

For faster training, install Flash Attention 2:

```bash
pip install flash-attn --no-build-isolation
```

If installation fails, the code will automatically fall back to standard attention.

## Usage

### Quick Start

Run the full pipeline (data preparation, training, evaluation):

```bash
make all
```

### Step-by-Step

#### 1. Prepare Dataset

Download and format the xLAM dataset:

```bash
make data
```

**Output**: Creates `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` and prints a formatted example.

**Expected Runtime**: 5-10 minutes

#### 2. Train Model

Fine-tune Llama 3.1 with qLoRA:

```bash
make train
```

**Output**: Saves LoRA adapter to `outputs/llama31-8b-xlam-lora/`

**Expected Runtime**:
- 3 epochs on 54k examples (~8-12 hours on RTX 3090)
- For quick smoke test, set `max_steps: 200` in `configs/config.yaml`

**Monitoring**:
- View real-time metrics at the W&B dashboard URL printed at training start
- Console logs show loss, learning rate, and evaluation metrics every 10-100 steps
- Access your run anytime at https://wandb.ai/your-username/xlam-llama-qlora

#### 3. Evaluate Model

Compare base model vs fine-tuned model on held-out test set:

```bash
make eval
```

**Output**: Saves metrics to `outputs/eval_results.json` and prints comparison table.

**Metrics**:
- `json_validity_rate`: Fraction of valid JSON outputs
- `tool_name_accuracy`: Correct tool selection
- `arg_key_accuracy`: Correct argument keys
- `arg_value_exact_accuracy`: Exact argument match

**Expected Runtime**: 10-20 minutes for 500 test examples

#### 4. Run Inference

Test the fine-tuned model on a single query:

```bash
make infer QUERY="get the weather in Tokyo"
```

**Output**: Prints formatted tool call JSON.

**Custom Tools**: Pass tools via `--tools` flag:

```bash
python -m src.inference --query "book a flight" --tools tools.json
```

## Configuration

All hyperparameters are in `configs/config.yaml`. Key settings:

### LoRA
- `r: 16`, `alpha: 32`, `dropout: 0.05`
- Target modules: All attention + MLP layers

### Training
- 3 epochs, learning rate `2e-4`, cosine scheduler
- Effective batch size: 16 (4 per device × 4 gradient accumulation)
- Optimizer: `paged_adamw_8bit`
- Max sequence length: 2048 tokens

### Quantization
- 4-bit NF4 with double quantization
- BF16 compute dtype

### Quick Smoke Test

To test the pipeline quickly, edit `configs/config.yaml`:

```yaml
training:
  max_steps: 200  # Override num_train_epochs
```

This runs 200 optimization steps (~15-30 minutes).

## Expected Results

After training, the fine-tuned model should outperform the base model on all metrics:

| Metric | Base Model | LoRA Model | Improvement |
|--------|-----------|-----------|-------------|
| JSON Validity | ~70% | ~95%+ | +25%+ |
| Tool Name Accuracy | ~50% | ~85%+ | +35%+ |
| Arg Key Accuracy | ~40% | ~75%+ | +35%+ |
| Exact Match | ~25% | ~60%+ | +35%+ |

*(Actual results vary based on random seed and hardware)*

## Troubleshooting

### 1. Out of Memory (OOM)

**Symptoms**: CUDA OOM error during training

**Solutions**:
- Reduce `per_device_train_batch_size` to 2 or 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable gradient checkpointing (already enabled by default)
- Reduce `max_seq_length` to 1024

### 2. BitsAndBytes Errors

**Symptoms**: `bitsandbytes` import errors or CUDA library not found

**Solutions**:
```bash
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir
```

Ensure CUDA toolkit is properly installed.

### 3. HuggingFace 401 Unauthorized

**Symptoms**: HTTP 401 when loading Llama 3.1

**Solutions**:
1. Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Login: `huggingface-cli login`
3. Verify token has read access

### 4. Loss Goes to Zero / NaN

**Symptoms**: Training loss immediately drops to 0.0 or becomes NaN

**Likely Cause**: Response template mismatch between `data_prep.py` and `train.py`

**Verification**:
1. Run `make data` and check printed example
2. Verify the response template in the output matches `RESPONSE_TEMPLATE` in both scripts
3. Ensure `<|start_header_id|>assistant<|end_header_id|>\n\n` is identical in both files

### 5. Slow Training

**Symptoms**: Training taking longer than expected

**Solutions**:
- Install Flash Attention 2: `pip install flash-attn --no-build-isolation`
- Verify GPU utilization: `nvidia-smi`
- Check data loading isn't bottlenecked (should see GPU util > 90%)

### 6. Malformed JSON Outputs

**Symptoms**: Model generates invalid JSON during evaluation

**Cause**: Insufficient training or early stopping

**Solutions**:
- Train for full 3 epochs
- Check training loss converged (should be < 0.5)
- Increase LoRA `r` value for more capacity

### 7. Dataset Download Fails

**Symptoms**: `load_dataset` fails to download xLAM

**Solutions**:
1. Accept dataset license at https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
2. Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`
3. Retry with `HF_DATASETS_OFFLINE=0`

## Technical Details

### Data Formatting

Each example is formatted using Llama 3.1's chat template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

[System prompt with tools]<|eot_id|><|start_header_id|>user<|end_header_id|>

[User query]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[Tool calls as JSON]<|eot_id|>
```

Loss is computed **only** on the assistant's response using `DataCollatorForCompletionOnlyLM`.

### Training Strategy

- **Quantization**: 4-bit NF4 reduces memory by ~4× with minimal quality loss
- **LoRA**: Trains only ~1% of parameters (adapters), enabling efficient fine-tuning
- **Gradient Checkpointing**: Trades compute for memory (20% slower, 40% less VRAM)
- **Paged Optimizer**: Uses CPU RAM for optimizer states, reducing GPU memory

### Evaluation Methodology

- **Greedy Decoding**: `temperature=0.0`, `do_sample=False` for reproducibility
- **Set Comparison**: Tool names compared as sets (order-invariant)
- **Type Normalization**: Numeric types normalized before comparison
- **Graceful Degradation**: Invalid JSON doesn't crash evaluation

## Limitations

- **Single GPU Only**: No multi-GPU support (use DeepSpeed/FSDP for that)
- **No Model Merging**: Saves adapter only (use `merge_and_unload()` for deployment)
- **English Only**: Dataset is English-only

## License

This code is provided as-is for educational and research purposes. Please respect:

- **Llama 3.1**: Meta's license (https://ai.meta.com/llama/license/)
- **xLAM Dataset**: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)

## Citation

If you use this code, please cite:

```bibtex
@software{xlam_llama_qlora,
  title = {xLAM Llama 3.1 qLoRA Fine-tuning},
  author = {Tian Huang},
  year = {2024},
  url = {https://github.com/tianhg468/xlam-llama-qlora}
}
```

## Acknowledgments

- Meta AI for Llama 3.1
- Salesforce Research for xLAM dataset
- HuggingFace for transformers, PEFT, TRL
- Tim Dettmers for bitsandbytes

## Contact

For issues or questions, please open a GitHub issue.
