"""
Batched inference utilities for faster evaluation.
"""

import torch
from typing import List, Dict
from transformers import AutoTokenizer


def generate_batch(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    batch_size: int = 4,
) -> List[str]:
    """
    Generate responses for a batch of prompts.

    Args:
        model: The model to use
        tokenizer: Tokenizer
        prompts: List of prompts
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        batch_size: Batch size for generation

    Returns:
        List of generated responses
    """
    all_responses = []

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (skip input prompt)
        for j, output in enumerate(outputs):
            generated_ids = output[inputs["input_ids"][j].shape[0]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_responses.append(response)

    return all_responses


def generate_batch_with_cache(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    batch_size: int = 4,
    use_cache: bool = True,
) -> List[str]:
    """
    Generate responses with KV cache optimization.

    Args:
        model: The model to use
        tokenizer: Tokenizer
        prompts: List of prompts
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        batch_size: Batch size for generation
        use_cache: Whether to use KV cache

    Returns:
        List of generated responses
    """
    # Enable caching for faster generation
    original_use_cache = model.config.use_cache
    model.config.use_cache = use_cache

    responses = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        batch_size=batch_size,
    )

    # Restore original setting
    model.config.use_cache = original_use_cache

    return responses


def estimate_batch_size_for_inference(
    model,
    tokenizer: AutoTokenizer,
    sample_prompt: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> int:
    """
    Estimate optimal batch size for inference.

    Args:
        model: The model
        tokenizer: Tokenizer
        sample_prompt: Sample prompt to test
        max_new_tokens: Max tokens to generate
        device: Device to test on

    Returns:
        Estimated optimal batch size
    """
    def try_batch_size(batch_size: int) -> bool:
        """Test if a batch size works."""
        try:
            prompts = [sample_prompt] * batch_size
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

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

    # Binary search
    left, right = 1, 64
    optimal = 1

    print("Estimating optimal inference batch size...")
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

    print(f"Optimal inference batch size: {optimal}")
    return optimal
