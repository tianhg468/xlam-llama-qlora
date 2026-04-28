"""
Evaluation using lm-evaluation-harness from EleutherAI.
Compares base model vs LoRA-finetuned model on standard benchmarks like MMLU.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml


def load_config() -> Dict:
    """Load configuration from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_lm_eval(
    model_path: str,
    tasks: List[str],
    num_fewshot: int = 5,
    batch_size: int = 1,
    output_path: str = None,
    device: str = "cuda",
    model_args: str = None,
) -> Dict:
    """
    Run lm-evaluation-harness on a model.

    Args:
        model_path: Path to model or model name on HuggingFace
        tasks: List of tasks to evaluate (e.g., ['mmlu', 'arc_easy'])
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        output_path: Path to save results JSON
        device: Device to use (cuda, cpu)
        model_args: Additional model arguments

    Returns:
        Dict with evaluation results
    """
    tasks_str = ",".join(tasks)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args or f"pretrained={model_path}",
        "--tasks", tasks_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--device", device,
    ]

    if output_path:
        cmd.extend(["--output_path", output_path])

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        print(result.stdout)

        # Parse results from output file if it exists
        if output_path:
            results_file = Path(output_path) / "results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    return json.load(f)

        return {"stdout": result.stdout, "stderr": result.stderr}

    except subprocess.CalledProcessError as e:
        print(f"Error running lm_eval: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return {"error": str(e), "stdout": e.stdout, "stderr": e.stderr}


def run_lm_eval_on_lora(
    base_model: str,
    lora_adapter_path: str,
    tasks: List[str],
    num_fewshot: int = 5,
    batch_size: int = 1,
    output_path: str = None,
    device: str = "cuda",
    load_in_4bit: bool = True,
) -> Dict:
    """
    Run lm-evaluation-harness on a LoRA model.

    Args:
        base_model: Base model name or path
        lora_adapter_path: Path to LoRA adapter
        tasks: List of tasks to evaluate
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        output_path: Path to save results JSON
        device: Device to use
        load_in_4bit: Whether to load in 4-bit quantization

    Returns:
        Dict with evaluation results
    """
    # Build model args for PEFT model
    model_args_parts = [
        f"pretrained={base_model}",
        f"peft={lora_adapter_path}",
    ]

    if load_in_4bit:
        model_args_parts.extend([
            "load_in_4bit=True",
            "bnb_4bit_compute_dtype=bfloat16",
            "bnb_4bit_quant_type=nf4",
            "bnb_4bit_use_double_quant=True",
        ])

    model_args = ",".join(model_args_parts)

    return run_lm_eval(
        model_path=base_model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        output_path=output_path,
        device=device,
        model_args=model_args,
    )


def main():
    """Run lm-evaluation-harness on both base and LoRA models."""
    config = load_config()
    project_root = Path(__file__).parent.parent
    adapter_path = project_root / config["training"]["output_dir"]
    output_dir = project_root / "outputs" / "lm_harness_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get harness config
    harness_config = config.get("lm_harness", {})
    tasks = harness_config.get("tasks", ["mmlu"])
    num_fewshot = harness_config.get("num_fewshot", 5)
    batch_size = harness_config.get("batch_size", 1)
    device = harness_config.get("device", "cuda")

    print("="*80)
    print("LM-EVALUATION-HARNESS: Base vs LoRA Model Comparison")
    print("="*80)
    print(f"Base model: {config['base_model']}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Tasks: {tasks}")
    print(f"Few-shot: {num_fewshot}")
    print(f"Batch size: {batch_size}")

    # Evaluate base model
    print("\n" + "="*80)
    print("Evaluating BASE MODEL")
    print("="*80)

    base_output = output_dir / "base_model"
    base_results = run_lm_eval(
        model_path=config["base_model"],
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        output_path=str(base_output),
        device=device,
        model_args=f"pretrained={config['base_model']},load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True",
    )

    # Evaluate LoRA model
    print("\n" + "="*80)
    print("Evaluating LORA MODEL")
    print("="*80)

    lora_output = output_dir / "lora_model"
    lora_results = run_lm_eval_on_lora(
        base_model=config["base_model"],
        lora_adapter_path=str(adapter_path),
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        output_path=str(lora_output),
        device=device,
        load_in_4bit=True,
    )

    # Save combined results
    combined_results = {
        "config": {
            "base_model": config["base_model"],
            "lora_adapter_path": str(adapter_path),
            "tasks": tasks,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
        },
        "base_model_results": base_results,
        "lora_model_results": lora_results,
    }

    combined_path = output_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined_results, f, indent=2)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"  Base model results: {base_output}")
    print(f"  LoRA model results: {lora_output}")
    print(f"  Combined results: {combined_path}")

    # Print comparison if results available
    print("\nTo view detailed results:")
    print(f"  Base: cat {base_output}/results.json")
    print(f"  LoRA: cat {lora_output}/results.json")


if __name__ == "__main__":
    main()
