"""
qLoRA training script for Llama 3.1 on xLAM function-calling dataset.
"""

import json
import os
from pathlib import Path
from typing import Dict

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


# Must match data_prep.py exactly
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def load_config() -> Dict:
    """Load configuration from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_quantization_config(config: Dict) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig for 4-bit NF4 quantization."""
    quant_cfg = config["quantization"]

    # Map string dtype to torch dtype
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


def setup_lora_config(config: Dict) -> LoraConfig:
    """Create LoRA configuration."""
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )


def load_formatted_dataset(data_dir: Path):
    """Load pre-formatted JSONL datasets."""
    train_path = str(data_dir / "train.jsonl")
    val_path = str(data_dir / "val.jsonl")

    dataset_dict = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "validation": val_path,
        }
    )

    return dataset_dict["train"], dataset_dict["validation"]


def main():
    """Run qLoRA training."""
    config = load_config()

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / config["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("qLoRA TRAINING - Llama 3.1 on xLAM Function Calling")
    print("="*80)

    # Load datasets
    print("\nLoading formatted datasets...")
    train_dataset, val_dataset = load_formatted_dataset(data_dir)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    # Setup quantization
    print("\nSetting up 4-bit quantization...")
    bnb_config = setup_quantization_config(config)

    # Determine attention implementation
    attn_implementation = config.get("attn_implementation", "flash_attention_2")
    try:
        # Try flash attention 2 first
        print(f"Attempting to use attention implementation: {attn_implementation}")
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=False,
        )
        print(f"Successfully loaded model with {attn_implementation}")
    except Exception as e:
        print(f"Flash Attention 2 not available ({e}), falling back to eager")
        attn_implementation = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=False,
        )

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=False,
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    tokenizer.padding_side = "right"

    # Prepare model for k-bit training
    print("\nPreparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Disable cache to avoid warnings
    model.config.use_cache = False

    # Apply LoRA
    print("\nApplying LoRA adapters...")
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Setup data collator for completion-only loss
    print("\nSetting up DataCollatorForCompletionOnlyLM...")
    # Encode response template without special tokens to get token IDs
    response_template_ids = tokenizer.encode(
        RESPONSE_TEMPLATE,
        add_special_tokens=False
    )
    print(f"Response template: {repr(RESPONSE_TEMPLATE)}")
    print(f"Response template token IDs: {response_template_ids}")

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # Setup SFTConfig
    print("\nConfiguring SFTTrainer...")
    training_cfg = config["training"]

    sft_config = SFTConfig(
        # Paths
        output_dir=str(output_dir),

        # Training
        num_train_epochs=training_cfg["num_train_epochs"],
        max_steps=training_cfg["max_steps"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=training_cfg["gradient_checkpointing"],

        # Optimizer
        optim=training_cfg["optim"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        max_grad_norm=training_cfg["max_grad_norm"],

        # Scheduler
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],

        # Precision
        bf16=training_cfg["bf16"],
        fp16=training_cfg["fp16"],

        # Logging & Evaluation
        logging_steps=training_cfg["logging_steps"],
        eval_strategy=training_cfg["eval_strategy"],
        eval_steps=training_cfg["eval_steps"],
        save_strategy=training_cfg["save_strategy"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],

        # Misc
        report_to=training_cfg["report_to"],
        seed=training_cfg["seed"],

        # SFT specific
        max_seq_length=training_cfg["max_seq_length"],
        dataset_text_field="text",  # our formatted examples have a "text" field
        packing=False,  # don't pack multiple examples into one sequence
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    trainer.train()

    # Save final adapter
    print("\n" + "="*80)
    print("SAVING ADAPTER")
    print("="*80)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"\nAdapter saved to: {output_dir}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
