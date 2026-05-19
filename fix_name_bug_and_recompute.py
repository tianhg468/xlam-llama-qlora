"""
Fix -name bug in predictions and recompute metrics.
This helps us see what the "true" model performance is if we fix the formatting issue.
"""
import json
from pathlib import Path


def fix_predictions(predictions):
    """Fix -name keys in predictions by renaming to name."""
    fixed_count = 0
    fixed_predictions = []

    for pred in predictions:
        pred_copy = pred.copy()

        if pred.get("valid") and isinstance(pred.get("calls"), list):
            fixed_calls = []
            for call in pred["calls"]:
                if isinstance(call, dict):
                    fixed_call = call.copy()
                    # Fix -name -> name
                    if "-name" in fixed_call:
                        fixed_call["name"] = fixed_call.pop("-name")
                        fixed_count += 1
                    fixed_calls.append(fixed_call)
                else:
                    fixed_calls.append(call)
            pred_copy["calls"] = fixed_calls

        fixed_predictions.append(pred_copy)

    return fixed_predictions, fixed_count


def compute_metrics(predictions, ground_truths):
    """Compute evaluation metrics."""
    total = len(predictions)
    json_valid_count = 0
    tool_name_correct = 0
    arg_key_correct = 0
    arg_value_exact_correct = 0
    valid_pairs = 0

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

        # Filter out non-dict items
        pred_calls = [call for call in pred_calls if isinstance(call, dict)]
        gt_calls = [call for call in gt_calls if isinstance(call, dict)]

        if not pred_calls or not gt_calls:
            continue

        valid_pairs += 1

        # Tool name accuracy
        pred_tool_names = set(call.get("name", "") for call in pred_calls)
        gt_tool_names = set(call.get("name", "") for call in gt_calls)
        if pred_tool_names == gt_tool_names:
            tool_name_correct += 1

        # Arg key accuracy
        pred_arg_keys = set(
            tuple(sorted(call.get("arguments", {}).keys()))
            for call in pred_calls
        )
        gt_arg_keys = set(
            tuple(sorted(call.get("arguments", {}).keys()))
            for call in gt_calls
        )
        if pred_arg_keys == gt_arg_keys:
            arg_key_correct += 1

        # Arg value exact match
        if pred_calls == gt_calls:
            arg_value_exact_correct += 1

    return {
        "json_validity_rate": json_valid_count / total if total > 0 else 0,
        "tool_name_accuracy": tool_name_correct / valid_pairs if valid_pairs > 0 else 0,
        "arg_key_accuracy": arg_key_correct / valid_pairs if valid_pairs > 0 else 0,
        "arg_value_exact_accuracy": arg_value_exact_correct / valid_pairs if valid_pairs > 0 else 0,
    }


def main():
    """Fix -name bug in checkpoint or results file and recompute metrics."""
    # Try checkpoint first
    checkpoint_file = Path("outputs/eval_checkpoints/LoRA_Model_checkpoint.json")
    results_file = Path("outputs/eval_lora_results.json")

    data_source = None
    predictions = None
    ground_truths = None

    if checkpoint_file.exists():
        data_source = "checkpoint"
        print(f"Loading from checkpoint: {checkpoint_file}")
        with open(checkpoint_file) as f:
            data = json.load(f)
        predictions = data["predictions"]
        ground_truths = data["ground_truths"]
    elif results_file.exists():
        data_source = "results"
        print(f"Loading from results file: {results_file}")
        with open(results_file) as f:
            data = json.load(f)
        predictions = data["lora_model"].get("predictions", [])
        if not predictions:
            print("❌ Results file doesn't contain predictions")
            print("   Please re-run evaluation with updated code")
            return
        # Need to load test data to get ground truths
        print("⚠️  Results file doesn't have ground truths, loading from test data...")
        test_file = Path("data/test.jsonl")
        if not test_file.exists():
            print(f"❌ Test data not found: {test_file}")
            return

        ground_truths = []
        with open(test_file) as f:
            for i, line in enumerate(f):
                if i >= len(predictions):
                    break
                example = json.loads(line)
                target_json_str = example["target"].replace("<|eot_id|>", "").strip()
                try:
                    gt_calls = json.loads(target_json_str)
                    if not isinstance(gt_calls, list):
                        gt_calls = [gt_calls]
                except json.JSONDecodeError:
                    gt_calls = []
                ground_truths.append({"calls": gt_calls})
    else:
        print("❌ No checkpoint or results file found")
        print(f"   Checkpoint: {checkpoint_file}")
        print(f"   Results: {results_file}")
        print("   Please run evaluation first")
        return

    print(f"Loaded {len(predictions)} predictions")

    # Original metrics
    print("\n" + "="*80)
    print("ORIGINAL METRICS (with -name bug)")
    print("="*80)
    original_metrics = compute_metrics(predictions, ground_truths)
    for metric, value in original_metrics.items():
        print(f"{metric}: {value:.2%}")

    # Fix predictions
    print("\n" + "="*80)
    print("FIXING -name BUG")
    print("="*80)
    fixed_predictions, fixed_count = fix_predictions(predictions)
    print(f"Fixed {fixed_count} instances of '-name' -> 'name'")

    # Recompute metrics
    print("\n" + "="*80)
    print("FIXED METRICS (after correcting -name bug)")
    print("="*80)
    fixed_metrics = compute_metrics(fixed_predictions, ground_truths)
    for metric, value in fixed_metrics.items():
        print(f"{metric}: {value:.2%}")

    # Show improvement
    print("\n" + "="*80)
    print("IMPROVEMENT")
    print("="*80)
    for metric in original_metrics:
        improvement = fixed_metrics[metric] - original_metrics[metric]
        print(f"{metric}: {improvement:+.2%}")

    # Save fixed results
    if data_source == "results":
        fixed_results_file = Path("outputs/eval_lora_results_fixed.json")
        with open(results_file) as f:
            results_data = json.load(f)
        results_data["lora_model"]["predictions"] = fixed_predictions
        results_data["lora_model"]["metrics_fixed"] = fixed_metrics
        results_data["lora_model"]["metrics_original"] = original_metrics
        results_data["lora_model"]["fixes_applied"] = {
            "name_bug_fixes": fixed_count
        }
        with open(fixed_results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\n✅ Fixed results saved to: {fixed_results_file}")


if __name__ == "__main__":
    main()
