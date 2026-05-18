"""
Evaluation script comparing base model vs LoRA-finetuned model on xLAM test set.
Computes JSON validity, tool name accuracy, arg key accuracy, and exact match metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config() -> Dict:
    """Load configuration from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_test_data(data_dir: Path, num_samples: int = None) -> List[Dict]:
    """Load test dataset from JSONL."""
    test_path = data_dir / "test.jsonl"
    examples = []
    with open(test_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
            if num_samples and len(examples) >= num_samples:
                break
    return examples


def setup_quantization_config(config: Dict) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig for 4-bit NF4 quantization."""
    quant_cfg = config["quantization"]
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map[quant_cfg["bnb_4bit_compute_dtype"]]

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )


def load_models(config: Dict, adapter_path: Path):
    """
    Load both base model and LoRA-adapted model.

    IMPORTANT: Must load base model twice to avoid adapter contamination.
    If we load adapter on top of base_model, they share the same underlying weights.

    Returns:
        Tuple of (base_model, lora_model, tokenizer)
    """
    bnb_config = setup_quantization_config(config)

    print("Loading base model (for base evaluation)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )

    print("Loading second base model instance (for LoRA evaluation)...")
    base_model_for_lora = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )

    print("Loading LoRA adapter on top of second base model...")
    lora_model = PeftModel.from_pretrained(
        base_model_for_lora,
        str(adapter_path),
        is_trainable=False,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return base_model, lora_model, tokenizer


def generate_response(model, tokenizer, prompt: str, config: Dict) -> str:
    """Generate model response using greedy decoding."""
    eval_cfg = config["evaluation"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # For Llama 3.1, <|eot_id|> is already the EOS token (ID: 128009)
    # Explicitly override ALL generation config to prevent model defaults from interfering
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=eval_cfg["max_new_tokens"],
            min_new_tokens=1,
            do_sample=False,  # Force greedy decoding
            num_beams=1,  # No beam search
            temperature=None,  # Explicitly disable
            top_p=None,  # Explicitly disable
            top_k=None,  # Explicitly disable
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,  # Single EOS token (128009)
            repetition_penalty=1.2,  # Penalty to discourage repetition
            no_repeat_ngram_size=3,  # Never repeat same 3-token sequence
        )

    # Decode only the generated tokens (skip input prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Fallback: If response contains repetitive closing brackets, truncate at first valid JSON
    # This handles cases where EOS token doesn't stop generation properly
    if response.count("}}]") > 2:  # More than 2 occurrences suggests repetition
        # Find the first complete JSON array
        import re
        match = re.search(r'\[.*?\]\s*(?=\}|$)', response)
        if match:
            response = match.group(0)

    return response


def parse_tool_calls(response: str) -> Tuple[bool, List[Dict]]:
    """
    Parse model response as JSON tool calls.

    Returns:
        (is_valid, parsed_calls)
        - is_valid: True if response is valid JSON
        - parsed_calls: List of tool call dicts, or empty list if invalid
    """
    # Try to extract JSON from response (model might add extra text)
    response = response.strip()

    # Try direct parse first
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return True, parsed
        elif isinstance(parsed, dict):
            return True, [parsed]  # single tool call
        else:
            return False, []
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in response
    start_idx = response.find("[")
    end_idx = response.rfind("]")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            json_str = response[start_idx:end_idx+1]
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return True, parsed
        except json.JSONDecodeError:
            pass

    return False, []


def compute_metrics(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        predictions: List of dicts with 'valid' (bool) and 'calls' (List[Dict])
        ground_truths: List of dicts with 'calls' (List[Dict])

    Returns:
        Dict with metric scores
    """
    assert len(predictions) == len(ground_truths)

    json_valid_count = 0
    tool_name_correct = 0
    arg_key_correct = 0
    arg_value_exact = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        # JSON validity
        if pred["valid"]:
            json_valid_count += 1

        # Skip other metrics if invalid JSON
        if not pred["valid"]:
            continue

        pred_calls = pred["calls"]
        gt_calls = gt["calls"]

        # Defensive: ensure calls are lists of dicts
        if not isinstance(pred_calls, list) or not isinstance(gt_calls, list):
            continue

        # Filter out non-dict items (defensive programming)
        pred_calls = [call for call in pred_calls if isinstance(call, dict)]
        gt_calls = [call for call in gt_calls if isinstance(call, dict)]

        if not pred_calls or not gt_calls:
            continue

        # Tool name accuracy (set comparison, order-invariant)
        pred_tool_names = set(call.get("name", "") for call in pred_calls)
        gt_tool_names = set(call.get("name", "") for call in gt_calls)

        if pred_tool_names == gt_tool_names:
            tool_name_correct += 1

            # Arg key accuracy (check if tool names match first)
            # For each tool, check if arg keys match
            # Create mappings for comparison
            pred_tools_map = {call.get("name", ""): call.get("arguments", {}) for call in pred_calls}
            gt_tools_map = {call.get("name", ""): call.get("arguments", {}) for call in gt_calls}

            all_keys_match = True
            all_values_match = True

            for tool_name in gt_tool_names:
                pred_args = pred_tools_map.get(tool_name, {})
                gt_args = gt_tools_map.get(tool_name, {})

                # Check keys
                if set(pred_args.keys()) != set(gt_args.keys()):
                    all_keys_match = False
                    all_values_match = False
                    break

                # Check values (with type normalization)
                for key in gt_args.keys():
                    pred_val = pred_args.get(key)
                    gt_val = gt_args[key]

                    # Normalize numeric types
                    if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
                        if abs(pred_val - gt_val) > 1e-6:
                            all_values_match = False
                    elif pred_val != gt_val:
                        all_values_match = False

            if all_keys_match:
                arg_key_correct += 1

            if all_values_match:
                arg_value_exact += 1

    return {
        "json_validity_rate": json_valid_count / total if total > 0 else 0.0,
        "tool_name_accuracy": tool_name_correct / total if total > 0 else 0.0,
        "arg_key_accuracy": arg_key_correct / total if total > 0 else 0.0,
        "arg_value_exact_accuracy": arg_value_exact / total if total > 0 else 0.0,
        "total_examples": total,
    }


def evaluate_model(model, tokenizer, test_data: List[Dict], config: Dict, model_name: str, checkpoint_dir: Path = None) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate a single model on test data with incremental checkpointing.

    Returns:
        (metrics_dict, predictions_list)
    """
    print(f"\nEvaluating {model_name}...")

    predictions = []
    ground_truths = []

    # Try to load checkpoint if it exists
    checkpoint_file = None
    start_idx = 0
    if checkpoint_dir:
        checkpoint_file = checkpoint_dir / f"{model_name.replace(' ', '_')}_checkpoint.json"
        if checkpoint_file.exists():
            print(f"  Found checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                predictions = checkpoint_data["predictions"]
                ground_truths = checkpoint_data["ground_truths"]
                start_idx = len(predictions)
                print(f"  Resuming from example {start_idx}/{len(test_data)}")

    for idx, example in enumerate(test_data):
        # Skip already processed examples
        if idx < start_idx:
            continue

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(test_data)} examples")

        prompt = example["prompt"]
        target = example["target"]

        # Parse ground truth
        # target is like: '[{"name":"tool","arguments":{...}}]<|eot_id|>'
        # Remove <|eot_id|> and parse
        target_json_str = target.replace("<|eot_id|>", "").strip()
        try:
            gt_calls = json.loads(target_json_str)
            if not isinstance(gt_calls, list):
                gt_calls = [gt_calls]
        except json.JSONDecodeError:
            gt_calls = []

        ground_truths.append({"calls": gt_calls})

        # Generate prediction
        response = generate_response(model, tokenizer, prompt, config)
        is_valid, pred_calls = parse_tool_calls(response)

        predictions.append({
            "valid": is_valid,
            "calls": pred_calls,
            "raw_response": response,
        })

        # Save checkpoint every 50 examples
        if checkpoint_file and (idx + 1) % 50 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "predictions": predictions,
                    "ground_truths": ground_truths,
                    "progress": idx + 1,
                    "total": len(test_data)
                }, f)

    # Remove checkpoint file when complete
    if checkpoint_file and checkpoint_file.exists():
        checkpoint_file.unlink()

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)
    print(f"\n{model_name} Results:")
    print(f"  JSON Validity: {metrics['json_validity_rate']:.2%}")
    print(f"  Tool Name Accuracy: {metrics['tool_name_accuracy']:.2%}")
    print(f"  Arg Key Accuracy: {metrics['arg_key_accuracy']:.2%}")
    print(f"  Arg Value Exact: {metrics['arg_value_exact_accuracy']:.2%}")

    return metrics, predictions


def main():
    """Run evaluation on both base and LoRA models."""
    config = load_config()
    project_root = Path(__file__).parent.parent

    # Try to load data from Google Drive first, fallback to local
    drive_data_dir = Path("/content/drive/MyDrive/xlam-llama-qlora/data")
    local_data_dir = project_root / "data"

    if (drive_data_dir / "test.jsonl").exists():
        data_dir = drive_data_dir
        print(f"✅ Loading data from Google Drive: {data_dir}")
    elif (local_data_dir / "test.jsonl").exists():
        data_dir = local_data_dir
        print(f"⚠️  Loading data from local: {data_dir}")
    else:
        raise FileNotFoundError(
            f"Test data not found in Drive ({drive_data_dir}) or local ({local_data_dir}). "
            f"Please run data preparation first: python -m src.data_prep_optimized"
        )

    # Adapter path - use absolute path if provided, otherwise relative to project
    adapter_path_config = config["training"]["output_dir"]
    if Path(adapter_path_config).is_absolute():
        adapter_path = Path(adapter_path_config)
    else:
        adapter_path = project_root / adapter_path_config

    print(f"✅ Loading adapter from: {adapter_path}")

    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Setup checkpoint directory (prefer Drive for persistence)
    drive_checkpoint_dir = Path("/content/drive/MyDrive/xlam-llama-qlora/outputs/eval_checkpoints")
    local_checkpoint_dir = output_dir / "eval_checkpoints"

    if drive_checkpoint_dir.parent.parent.exists():
        checkpoint_dir = drive_checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Checkpoints will be saved to Google Drive: {checkpoint_dir}")
    else:
        checkpoint_dir = local_checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  Checkpoints will be saved locally: {checkpoint_dir}")

    print("="*80)
    print("EVALUATION - Base vs LoRA on xLAM Test Set")
    print("="*80)
    print("💾 Progress auto-saved every 50 examples")

    # Load test data
    num_samples = config["evaluation"]["num_test_samples"]
    print(f"\nLoading up to {num_samples} test examples...")
    test_data = load_test_data(data_dir, num_samples)
    print(f"Loaded {len(test_data)} test examples")

    # Load models
    base_model, lora_model, tokenizer = load_models(config, adapter_path)

    # Evaluate base model
    base_metrics, base_predictions = evaluate_model(
        base_model,
        tokenizer,
        test_data,
        config,
        "Base Model",
        checkpoint_dir=checkpoint_dir
    )

    # Evaluate LoRA model
    lora_metrics, lora_predictions = evaluate_model(
        lora_model,
        tokenizer,
        test_data,
        config,
        "LoRA Model",
        checkpoint_dir=checkpoint_dir
    )

    # Save results
    results = {
        "base_model": {
            "model_name": config["base_model"],
            "metrics": base_metrics,
        },
        "lora_model": {
            "model_name": f"{config['base_model']} + LoRA",
            "adapter_path": str(adapter_path),
            "metrics": lora_metrics,
        },
        "config": {
            "num_test_samples": len(test_data),
            "evaluation_config": config["evaluation"],
        }
    }

    # Save results to local outputs directory
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save to Google Drive if available (to persist across Colab restarts)
    drive_output_dir = Path("/content/drive/MyDrive/xlam-llama-qlora/outputs")
    if drive_output_dir.parent.exists():
        drive_output_dir.mkdir(parents=True, exist_ok=True)
        drive_results_path = drive_output_dir / "eval_results.json"
        with open(drive_results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Results saved to Google Drive: {drive_results_path}")
        print(f"   (Also saved locally: {results_path})")
    else:
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {results_path}")

    # Print comparison
    print("\nCOMPARISON:")
    print(f"{'Metric':<30} {'Base Model':<15} {'LoRA Model':<15} {'Improvement':<15}")
    print("-"*80)
    for metric_name in ["json_validity_rate", "tool_name_accuracy", "arg_key_accuracy", "arg_value_exact_accuracy"]:
        base_val = base_metrics[metric_name]
        lora_val = lora_metrics[metric_name]
        improvement = lora_val - base_val
        print(f"{metric_name:<30} {base_val:>14.2%} {lora_val:>14.2%} {improvement:>+14.2%}")


if __name__ == "__main__":
    main()
