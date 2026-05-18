"""
Evaluate ONLY the base model on xLAM test set.
Saves results to outputs/eval_base_results.json
"""
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval import (
    load_config,
    load_test_data,
    setup_quantization_config,
    evaluate_model,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def main():
    """Evaluate base model only."""
    config = load_config()
    project_root = Path(__file__).parent

    # Try to load data from Google Drive first, fallback to local
    drive_data_dir = Path("/content/drive/MyDrive/xlam-llama-qlora/data")
    local_data_dir = project_root / "data"

    if (drive_data_dir / "test.jsonl").exists():
        data_dir = drive_data_dir
        print(f"✅ Loading data from Google Drive: {data_dir}")
    elif (local_data_dir / "test.jsonl").exists():
        data_dir = local_data_dir
        print(f"✅ Loading data from local: {data_dir}")
    else:
        raise FileNotFoundError(
            f"Test data not found in Drive ({drive_data_dir}) or local ({local_data_dir}). "
            f"Please run data preparation first: python -m src.data_prep_optimized"
        )

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
        print(f"✅ Checkpoints will be saved locally: {checkpoint_dir}")

    print("="*80)
    print("EVALUATING BASE MODEL ONLY")
    print("="*80)
    print("💾 Progress auto-saved every 50 examples")

    # Load test data
    num_samples = config["evaluation"]["num_test_samples"]
    print(f"\nLoading up to {num_samples} test examples...")
    test_data = load_test_data(data_dir, num_samples)
    print(f"Loaded {len(test_data)} test examples")

    # Load base model only
    print("\nLoading base model...")
    bnb_config = setup_quantization_config(config)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate base model
    base_metrics, base_predictions = evaluate_model(
        base_model,
        tokenizer,
        test_data,
        config,
        "Base Model",
        checkpoint_dir=checkpoint_dir
    )

    # Save results
    results = {
        "base_model": {
            "model_name": config["base_model"],
            "metrics": base_metrics,
        },
        "config": {
            "num_test_samples": len(test_data),
            "evaluation_config": config["evaluation"],
        }
    }

    results_path = output_dir / "eval_base_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save to Google Drive if available
    drive_output_dir = Path("/content/drive/MyDrive/xlam-llama-qlora/outputs")
    if drive_output_dir.parent.exists():
        drive_output_dir.mkdir(parents=True, exist_ok=True)
        drive_results_path = drive_output_dir / "eval_base_results.json"
        with open(drive_results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*80}")
        print("BASE MODEL EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Results saved to Google Drive: {drive_results_path}")
        print(f"   (Also saved locally: {results_path})")
    else:
        print(f"\n{'='*80}")
        print("BASE MODEL EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {results_path}")

    # Print results
    print("\nBase Model Results:")
    print(f"  JSON Validity: {base_metrics['json_validity_rate']:.2%}")
    print(f"  Tool Name Accuracy: {base_metrics['tool_name_accuracy']:.2%}")
    print(f"  Arg Key Accuracy: {base_metrics['arg_key_accuracy']:.2%}")
    print(f"  Arg Value Exact: {base_metrics['arg_value_exact_accuracy']:.2%}")


if __name__ == "__main__":
    main()
