"""
Data preparation for xLAM function-calling dataset.
Downloads, formats, splits, and saves as JSONL.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import yaml
from datasets import load_dataset

# Import token retrieval function (compatible with newer huggingface_hub)
try:
    from huggingface_hub import get_token
except ImportError:
    # Fallback for older versions
    from huggingface_hub import HfFolder
    get_token = HfFolder.get_token


# Exact Llama 3.1 response template for completion-only loss
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def load_config() -> Dict:
    """Load configuration from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_example(example: Dict) -> Dict:
    """
    Format a single xLAM example into Llama 3.1 chat template.

    Args:
        example: Dict with 'query', 'tools', 'answers' fields (tools/answers are JSON strings)

    Returns:
        Dict with 'text', 'prompt', 'target' fields
    """
    query = example["query"]

    # Parse JSON strings
    try:
        tools = json.loads(example["tools"])
        answers = json.loads(example["answers"])
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON in example: {e}")

    # Format tools for readability (indent=2)
    tools_json = json.dumps(tools, indent=2)

    # Format answers compactly (no spaces)
    answers_json = json.dumps(answers, separators=(",", ":"))

    # Build the full prompt following exact Llama 3.1 template
    system_message = f"""You are a helpful AI assistant with access to a set of tools. When the user makes a request, decide which tool(s) to call and respond ONLY with a JSON list of tool calls in this format:

[{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}]

Available tools:
{tools_json}"""

    # Construct full text with special tokens
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{query}<|eot_id|>{RESPONSE_TEMPLATE}"
    )

    target = f"{answers_json}<|eot_id|>"
    text = prompt + target

    return {
        "text": text,
        "prompt": prompt,
        "target": target,
    }


def main():
    """Download xLAM dataset, format, split, and save as JSONL."""
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

    # Format all examples
    print("Formatting examples...")
    formatted_examples = []
    for idx, example in enumerate(dataset):
        try:
            formatted = format_example(example)
            formatted_examples.append(formatted)
        except Exception as e:
            print(f"Warning: Skipping example {idx} due to error: {e}")
            continue

    print(f"Successfully formatted {len(formatted_examples)} examples")

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

    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

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
    print(train_data[0]["text"])
    print("="*80)
    print(f"\nPrompt length: {len(train_data[0]['prompt'])} chars")
    print(f"Target length: {len(train_data[0]['target'])} chars")
    print(f"Total length: {len(train_data[0]['text'])} chars")
    print("\nResponse template used for collator:")
    print(repr(RESPONSE_TEMPLATE))
    print("="*80)


if __name__ == "__main__":
    main()
