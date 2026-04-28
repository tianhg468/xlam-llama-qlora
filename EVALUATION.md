# Evaluation Guide

This project includes three types of evaluation to comprehensively assess your fine-tuned model:

1. **Custom xLAM Evaluation** - Compare base vs LoRA on tool calling tasks
2. **LM Evaluation Harness** - Standard benchmarks (MMLU, etc.)
3. **Frontier Model Comparison** - Compare against GPT-4, Claude, Gemini

## 1. Custom xLAM Evaluation

Evaluates your base model and LoRA-finetuned model on the xLAM test set using custom tool calling metrics.

### Run:
```bash
python src/eval.py
```

### Metrics:
- **JSON Validity Rate**: Percentage of valid JSON responses
- **Tool Name Accuracy**: Correct tool names selected
- **Arg Key Accuracy**: Correct argument keys provided
- **Arg Value Exact**: Exact match on argument values

### Output:
- Results saved to `outputs/eval_results.json`
- Comparison table printed to console

### Configuration:
Edit `configs/config.yaml`:
```yaml
evaluation:
  num_test_samples: 500  # Number of test examples
  max_new_tokens: 512
  temperature: 0.0
  do_sample: false
```

## 2. LM Evaluation Harness

Uses EleutherAI's standard evaluation harness to benchmark on academic datasets like MMLU.

### Installation:
```bash
pip install lm-eval>=0.4.0
```

### Run:
```bash
python src/eval_lm_harness.py
```

### Tasks Available:
- `mmlu` - Massive Multitask Language Understanding (full)
- `mmlu_abstract_algebra` - MMLU subset
- `arc_easy`, `arc_challenge` - AI2 Reasoning Challenge
- `hellaswag` - Commonsense reasoning
- `winogrande` - Winograd Schema Challenge
- Many more (see [lm-evaluation-harness tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks))

### Configuration:
Edit `configs/config.yaml`:
```yaml
lm_harness:
  tasks:
    - "mmlu"  # Can add multiple tasks
    - "arc_easy"
  num_fewshot: 5
  batch_size: 1
  device: "cuda"
```

### Output:
- Base model results: `outputs/lm_harness_results/base_model/results.json`
- LoRA model results: `outputs/lm_harness_results/lora_model/results.json`
- Combined results: `outputs/lm_harness_results/combined_results.json`

## 3. Frontier Model Comparison

Compare your model against frontier models (GPT-4, Claude, Gemini) on the same xLAM test set.

### Installation:
```bash
pip install openai anthropic google-generativeai
```

### Setup API Keys:
Set environment variables for the models you want to test:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

Or create a `.env` file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### Run:
```bash
python src/eval_frontier.py
```

### Supported Models:
- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-5-sonnet`, `claude-3-opus`
- **Google**: `gemini-1.5-pro`

### Configuration:
Edit `configs/config.yaml`:
```yaml
frontier_models:
  num_test_samples: 100  # Keep low to control API costs
  enabled:
    - "gpt-4o"
    - "gpt-4-turbo"
    - "claude-3-5-sonnet"
    # - "gemini-1.5-pro"  # Uncomment to enable
```

### Output:
- Detailed results: `outputs/frontier_comparison/comparison_results.json`
- Comparison table printed to console showing all models side-by-side

### Cost Considerations:
API calls can add up quickly. Start with `num_test_samples: 100` or less. At 100 examples:
- GPT-4-Turbo: ~$0.50-1.00
- Claude-3.5-Sonnet: ~$0.30-0.60
- Gemini-1.5-Pro: ~$0.10-0.30

## Quick Start: Run All Evaluations

For a complete evaluation workflow:

```bash
# 1. Run custom xLAM evaluation
python src/eval.py

# 2. Run LM evaluation harness (MMLU)
python src/eval_lm_harness.py

# 3. Run frontier model comparison (requires API keys)
python src/eval_frontier.py
```

## Interpreting Results

### Custom xLAM Metrics
- Focus on **Tool Name Accuracy** and **Arg Value Exact** as key indicators
- JSON Validity shows if the model learned the output format
- Compare improvements from Base → LoRA

### LM Harness MMLU
- MMLU tests general knowledge across 57 subjects
- Compare Base vs LoRA to see if tool calling training affected general capabilities
- Typical scores: 40-60% for 8B models

### Frontier Comparison
- Shows how your specialized LoRA compares to general-purpose frontier models
- Your model may outperform on tool calling despite being 8B vs frontier 175B+ models
- JSON Validity often higher for fine-tuned models

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `num_test_samples`
- Use smaller test set

### API Errors
- Check API keys are set correctly
- Verify API quota/credits
- Reduce `num_test_samples` for cost control
- Disable expensive models in config

### Import Errors
```bash
pip install -r requirements.txt
```

## Example Results Structure

```
outputs/
├── eval_results.json                      # Custom xLAM eval
├── lm_harness_results/
│   ├── base_model/results.json           # Base MMLU scores
│   ├── lora_model/results.json           # LoRA MMLU scores
│   └── combined_results.json             # Combined
└── frontier_comparison/
    └── comparison_results.json            # vs GPT-4, Claude, etc.
```
