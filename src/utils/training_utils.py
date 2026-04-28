"""
Training utilities for optimizations and monitoring.
"""

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Optional
import wandb


class LossLoggingCallback(TrainerCallback):
    """Callback to log detailed loss statistics."""

    def __init__(self):
        self.losses = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict] = None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])

            # Log additional statistics
            if len(self.losses) >= 10:
                recent_losses = self.losses[-10:]
                logs["loss_mean_10"] = sum(recent_losses) / len(recent_losses)
                logs["loss_min_10"] = min(recent_losses)
                logs["loss_max_10"] = max(recent_losses)


class GradientMonitorCallback(TrainerCallback):
    """Callback to monitor gradient norms."""

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        if model is not None and state.global_step % args.logging_steps == 0:
            total_norm = 0.0
            num_params = 0

            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    num_params += 1

            if num_params > 0:
                total_norm = total_norm ** 0.5

                # Log to wandb if available
                if wandb.run is not None:
                    wandb.log({
                        "gradient_norm": total_norm,
                        "step": state.global_step,
                    })


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.wait = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict, **kwargs):
        eval_loss = metrics.get("eval_loss")

        if eval_loss is None:
            return

        if self.best_loss is None:
            self.best_loss = eval_loss
        elif eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping triggered after {self.wait} evaluations without improvement")
                control.should_training_stop = True


def find_optimal_batch_size(model, tokenizer, sample_text: str, max_seq_length: int, device: str = "cuda") -> int:
    """
    Find the largest batch size that fits in memory using binary search.

    Args:
        model: The model to test
        tokenizer: Tokenizer
        sample_text: Sample text to tokenize
        max_seq_length: Maximum sequence length
        device: Device to test on

    Returns:
        Optimal batch size
    """
    def try_batch_size(batch_size: int) -> bool:
        """Test if a batch size fits in memory."""
        try:
            # Tokenize sample
            inputs = tokenizer(
                [sample_text] * batch_size,
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            # Backward pass simulation (allocates gradients)
            if outputs.logits.requires_grad:
                loss = outputs.logits.sum()
                loss.backward()

            # Clear cache
            del inputs, outputs
            if device == "cuda":
                torch.cuda.empty_cache()

            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                if device == "cuda":
                    torch.cuda.empty_cache()
                return False
            raise e

    # Binary search for optimal batch size
    left, right = 1, 128
    optimal = 1

    print("Finding optimal batch size...")
    while left <= right:
        mid = (left + right) // 2
        print(f"  Testing batch size: {mid}...", end=" ")

        if try_batch_size(mid):
            print("✓")
            optimal = mid
            left = mid + 1
        else:
            print("✗ (OOM)")
            right = mid - 1

    print(f"Optimal batch size: {optimal}")
    return optimal


def print_training_summary(config: Dict, num_train_examples: int, num_val_examples: int):
    """Print a summary of training configuration."""
    training_cfg = config["training"]

    # Calculate effective batch size
    effective_batch_size = (
        training_cfg["per_device_train_batch_size"] *
        training_cfg["gradient_accumulation_steps"]
    )

    # Estimate training steps
    if training_cfg["max_steps"] > 0:
        total_steps = training_cfg["max_steps"]
    else:
        steps_per_epoch = num_train_examples // effective_batch_size
        total_steps = steps_per_epoch * training_cfg["num_train_epochs"]

    # Estimate time (rough approximation)
    # Assume ~1 second per step for 8B model on A100
    estimated_hours = total_steps / 3600

    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Dataset:")
    print(f"  Train examples: {num_train_examples:,}")
    print(f"  Validation examples: {num_val_examples:,}")
    print(f"\nBatch Configuration:")
    print(f"  Per-device batch size: {training_cfg['per_device_train_batch_size']}")
    print(f"  Gradient accumulation steps: {training_cfg['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"\nTraining Steps:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Eval every: {training_cfg['eval_steps']} steps")
    print(f"  Save every: {training_cfg['save_steps']} steps")
    print(f"\nLearning Rate:")
    print(f"  Initial LR: {training_cfg['learning_rate']}")
    print(f"  Scheduler: {training_cfg['lr_scheduler_type']}")
    print(f"  Warmup ratio: {training_cfg['warmup_ratio']}")
    print(f"\nEstimates:")
    print(f"  Estimated training time: ~{estimated_hours:.1f} hours")
    print(f"  (Actual time varies based on GPU and data complexity)")
    print("="*80 + "\n")


def setup_dataloader_optimizations(training_args):
    """
    Configure dataloader optimizations.

    Args:
        training_args: TrainingArguments object

    Returns:
        Modified training_args with dataloader optimizations
    """
    # Set dataloader num_workers
    # Use 4 workers for better throughput (adjust based on system)
    training_args.dataloader_num_workers = 4

    # Enable pin_memory for faster GPU transfer
    training_args.dataloader_pin_memory = True

    # Prefetch factor (how many batches to prefetch per worker)
    # This is not directly supported in TrainingArguments, but good to know

    return training_args
