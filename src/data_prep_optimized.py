"""
Optimized data preparation with quality filtering, caching, and statistics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import yaml
from datasets import load_dataset
from tqdm import tqdm

# Import token retrieval function (compatible with newer huggingface_hub)
try:
    from huggingface_hub import get_token
except ImportError:
    # Fallback for older versions
    from huggingface_hub import HfFolder
    get_token = HfFolder.get_token

# Import from data_prep to reuse format_example
import sys
sys.path.append(str(Path(__file__).parent))
from data_prep import format_example, RESPONSE_TEMPLATE
from utils.data_quality import validate_example, compute_statistics, filter_by_length


def load_config() -> Dict:
    """Load configuration from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Download xLAM dataset with quality filtering, statistics, and caching."""
    config = load_config()

    # Create data directory - save to Google Drive to persist across restarts
    # Try Drive first, fallback to local if Drive not available
    drive_data_dir = Path("/content/drive/MyDrive/xlam-llama-qlora/data")
    local_data_dir = Path(__file__).parent.parent / "data"

    if drive_data_dir.parent.exists():
        data_dir = drive_data_dir
        print(f"✅ Using Google Drive for data: {data_dir}")
        print("   Data will persist across Colab restarts!")
    else:
        data_dir = local_data_dir
        print(f"⚠️  Using local storage for data: {data_dir}")
        print("   WARNING: Data will be lost if Colab restarts!")

    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if cached data exists
    cache_file = data_dir / ".formatted_cache.json"
    if cache_file.exists():
        print(f"Found cached formatted data at {cache_file}")
        print("Loading from cache...")
        with open(cache_file, "r") as f:
            formatted_examples = json.load(f)
        print(f"Loaded {len(formatted_examples)} examples from cache")
    else:
        print(f"Loading dataset: {config['dataset_name']}")
        # Get HuggingFace token for gated datasets
        token = get_token()

        if token is None:
            print("⚠️  Warning: No HuggingFace token found!")
            print("Please login first: from huggingface_hub import login; login()")
            raise RuntimeError(
                "HuggingFace authentication required for gated dataset. "
                "Run: from huggingface_hub import login; login()"
            )

        print(f"✅ Token found: {token[:10]}...")

        # Load dataset - authentication should be automatic from login()
        # But we'll explicitly pass token=True to use the logged-in credentials
        dataset = load_dataset(config["dataset_name"], split="train", token=True)

        print(f"Total examples: {len(dataset)}")

        # Validate and format examples
        print("\nValidating and formatting examples...")
        formatted_examples = []
        validation_stats = {}

        for idx, example in enumerate(tqdm(dataset, desc="Processing")):
            # Validate example quality
            is_valid, reason = validate_example(example)

            if not is_valid:
                validation_stats[reason] = validation_stats.get(reason, 0) + 1
                continue

            # Format example
            try:
                formatted = format_example(example)
                formatted_examples.append(formatted)
            except Exception as e:
                validation_stats["formatting_error"] = validation_stats.get("formatting_error", 0) + 1
                continue

        print(f"\nSuccessfully formatted {len(formatted_examples)} examples")
        print(f"Filtered out {len(dataset) - len(formatted_examples)} examples")

        if validation_stats:
            print("\nFiltering reasons:")
            for reason, count in sorted(validation_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count}")

        # Cache formatted data
        print(f"\nCaching formatted data to {cache_file}")
        with open(cache_file, "w") as f:
            json.dump(formatted_examples, f)

    # Compute statistics BEFORE length filtering
    print("\n" + "="*80)
    print("DATASET STATISTICS (before length filtering)")
    print("="*80)
    stats = compute_statistics(formatted_examples)
    print(json.dumps(stats, indent=2))

    # Filter by length based on max_seq_length
    max_seq_length = config["training"]["max_seq_length"]
    # Approximate: 1 token ≈ 4 characters for English text
    # Use conservative estimate: max_seq_length * 3 characters
    max_char_length = max_seq_length * 3
    print(f"\nFiltering examples by length (max {max_char_length} characters ≈ {max_seq_length} tokens)...")
    formatted_examples = filter_by_length(formatted_examples, max_char_length)

    # Compute statistics AFTER length filtering
    print("\n" + "="*80)
    print("DATASET STATISTICS (after length filtering)")
    print("="*80)
    stats = compute_statistics(formatted_examples)
    print(json.dumps(stats, indent=2))

    # Save statistics to file
    stats_file = data_dir / "dataset_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_file}")

    # Split train/val/test
    from sklearn.model_selection import train_test_split

    train_ratio = config["train_split"]
    val_ratio = config["val_split"]
    test_ratio = config["test_split"]
    seed = config["seed"]

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Splits must sum to 1.0"

    # First split: train vs (val+test)
    train_data, temp_data = train_test_split(
        formatted_examples,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_ratio_adjusted,
        random_state=seed,
        shuffle=True
    )

    print(f"\n" + "="*80)
    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print("="*80)

    # Save as JSONL
    def save_jsonl(data: List[Dict], path: Path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} examples to {path}")

    save_jsonl(train_data, data_dir / "train.jsonl")
    save_jsonl(val_data, data_dir / "val.jsonl")
    save_jsonl(test_data, data_dir / "test.jsonl")

    # Print one formatted example for verification
    print("\n" + "="*80)
    print("SAMPLE FORMATTED EXAMPLE (first training example):")
    print("="*80)
    print(train_data[0]["text"][:1000] + "..." if len(train_data[0]["text"]) > 1000 else train_data[0]["text"])
    print("="*80)
    print(f"\nPrompt length: {len(train_data[0]['prompt'])} chars")
    print(f"Target length: {len(train_data[0]['target'])} chars")
    print(f"Total length: {len(train_data[0]['text'])} chars")
    print("\nResponse template used for collator:")
    print(repr(RESPONSE_TEMPLATE))
    print("="*80)

    print("\n✅ Data preparation complete!")
    print(f"   - Formatted data cached at: {cache_file}")
    print(f"   - Statistics saved at: {stats_file}")
    print(f"   - Splits saved at: {data_dir}/")


if __name__ == "__main__":
    main()
