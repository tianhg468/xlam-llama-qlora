"""
Merge LoRA adapter weights into base model for faster inference.

This creates a single merged model without needing PEFT at inference time,
which is faster and uses less memory.
"""

import argparse
from pathlib import Path
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config() -> dict:
    """Load configuration from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_and_save(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    device_map: str = "auto",
    max_memory: dict = None,
):
    """
    Merge LoRA adapter into base model and save.

    Args:
        base_model_name: Base model name or path
        adapter_path: Path to LoRA adapter
        output_path: Where to save merged model
        device_map: Device map for loading
        max_memory: Memory limits per device
    """
    print("="*80)
    print("MERGING LORA ADAPTER INTO BASE MODEL")
    print("="*80)

    # Load base model in full precision (required for merging)
    print(f"\nLoading base model: {base_model_name}")
    print("Note: Loading in full precision (not quantized) for merging...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # Use FP16 to save memory
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=False,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device_map,
    )

    print("\nMerging adapter weights into base model...")
    # This merges LoRA weights into the base model
    model = model.merge_and_unload()

    print(f"\nSaving merged model to: {output_path}")
    model.save_pretrained(output_path)

    # Also save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    print("\n" + "="*80)
    print("MERGE COMPLETE!")
    print("="*80)
    print(f"\nMerged model saved to: {output_path}")
    print("\nYou can now use this model directly without PEFT:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    print("\nThis is faster than loading base + adapter separately!")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (default: from config)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save merged model (default: outputs/merged_model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (default: from config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()
    project_root = Path(__file__).parent.parent

    # Set paths from args or config
    base_model = args.base_model or config["base_model"]
    adapter_path = args.adapter_path or str(project_root / config["training"]["output_dir"])
    output_path = args.output_path or str(project_root / "outputs" / "merged_model")

    # Check if adapter exists
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        print(f"Error: Adapter not found at {adapter_path}")
        print("Train the model first using: python -m src.train")
        return

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Merge and save
    merge_and_save(
        base_model_name=base_model,
        adapter_path=adapter_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
