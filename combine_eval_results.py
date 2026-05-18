"""
Combine separate base and LoRA evaluation results into a single comparison.
Reads eval_base_results.json and eval_lora_results.json, combines them.
"""
import json
from pathlib import Path


def main():
    """Combine evaluation results from separate runs."""
    output_dir = Path("outputs")

    base_results_path = output_dir / "eval_base_results.json"
    lora_results_path = output_dir / "eval_lora_results.json"

    # Check if both files exist
    if not base_results_path.exists():
        print(f"❌ Base results not found: {base_results_path}")
        print("   Run: python eval_base_only.py")
        return

    if not lora_results_path.exists():
        print(f"❌ LoRA results not found: {lora_results_path}")
        print("   Run: python eval_lora_only.py")
        return

    # Load both results
    with open(base_results_path) as f:
        base_data = json.load(f)

    with open(lora_results_path) as f:
        lora_data = json.load(f)

    # Combine into standard eval format
    combined = {
        "base_model": base_data["base_model"],
        "lora_model": lora_data["lora_model"],
        "config": base_data["config"],  # Use config from either (should be same)
    }

    # Save combined results
    combined_path = output_dir / "eval_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    print("="*80)
    print("COMBINED EVALUATION RESULTS")
    print("="*80)
    print(f"✅ Combined results saved to: {combined_path}")

    # Print comparison table
    base_metrics = base_data["base_model"]["metrics"]
    lora_metrics = lora_data["lora_model"]["metrics"]

    print("\nCOMPARISON:")
    print(f"{'Metric':<30} {'Base Model':<15} {'LoRA Model':<15} {'Improvement':<15}")
    print("-"*80)

    for metric in ["json_validity_rate", "tool_name_accuracy", "arg_key_accuracy", "arg_value_exact_accuracy"]:
        base_val = base_metrics[metric]
        lora_val = lora_metrics[metric]
        improvement = lora_val - base_val
        print(f"{metric:<30} {base_val:>14.2%} {lora_val:>14.2%} {improvement:>+14.2%}")

    print("="*80)


if __name__ == "__main__":
    main()
