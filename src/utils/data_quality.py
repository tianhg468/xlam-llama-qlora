"""
Data quality filtering and statistics for xLAM dataset.
"""

import json
from typing import Dict, List, Tuple


def validate_example(example: Dict) -> Tuple[bool, str]:
    """
    Validate a single xLAM example for quality.

    Returns:
        (is_valid, reason)
    """
    # Check required fields
    if "query" not in example or not example["query"]:
        return False, "missing_query"

    if "tools" not in example or not example["tools"]:
        return False, "missing_tools"

    if "answers" not in example or not example["answers"]:
        return False, "missing_answers"

    # Validate JSON parsing
    try:
        tools = json.loads(example["tools"])
        answers = json.loads(example["answers"])
    except json.JSONDecodeError:
        return False, "invalid_json"

    # Validate data types
    if not isinstance(tools, list):
        return False, "tools_not_list"

    if not isinstance(answers, list):
        return False, "answers_not_list"

    # Validate tools structure
    for tool in tools:
        if not isinstance(tool, dict):
            return False, "invalid_tool_format"
        if "name" not in tool:
            return False, "tool_missing_name"

    # Validate answers structure
    for answer in answers:
        if not isinstance(answer, dict):
            return False, "invalid_answer_format"
        if "name" not in answer:
            return False, "answer_missing_name"
        if "arguments" not in answer:
            return False, "answer_missing_arguments"

    # Check query length (avoid extremely short or long queries)
    query_len = len(example["query"])
    if query_len < 10:
        return False, "query_too_short"
    if query_len > 2000:
        return False, "query_too_long"

    # Check number of tools (avoid edge cases)
    if len(tools) == 0:
        return False, "no_tools"
    if len(tools) > 50:
        return False, "too_many_tools"

    # Check number of answers
    if len(answers) == 0:
        return False, "no_answers"
    if len(answers) > 10:
        return False, "too_many_answers"

    return True, "valid"


def compute_statistics(examples: List[Dict]) -> Dict:
    """Compute statistics on formatted examples."""
    stats = {
        "total_examples": len(examples),
        "query_lengths": [],
        "prompt_lengths": [],
        "target_lengths": [],
        "text_lengths": [],
        "num_tools": [],
        "num_answers": [],
    }

    for example in examples:
        stats["query_lengths"].append(len(example.get("query", "")))
        stats["prompt_lengths"].append(len(example.get("prompt", "")))
        stats["target_lengths"].append(len(example.get("target", "")))
        stats["text_lengths"].append(len(example.get("text", "")))

        # Parse to get counts
        try:
            if "tools" in example:
                tools = json.loads(example["tools"])
                stats["num_tools"].append(len(tools))
            if "answers" in example or "target" in example:
                # Extract from target
                target = example.get("target", "")
                target = target.replace("<|eot_id|>", "").strip()
                if target:
                    answers = json.loads(target)
                    if isinstance(answers, list):
                        stats["num_answers"].append(len(answers))
                    else:
                        stats["num_answers"].append(1)
        except:
            pass

    # Compute summary statistics
    summary = {
        "total_examples": stats["total_examples"],
        "query_length": {
            "min": min(stats["query_lengths"]) if stats["query_lengths"] else 0,
            "max": max(stats["query_lengths"]) if stats["query_lengths"] else 0,
            "mean": sum(stats["query_lengths"]) / len(stats["query_lengths"]) if stats["query_lengths"] else 0,
        },
        "prompt_length": {
            "min": min(stats["prompt_lengths"]) if stats["prompt_lengths"] else 0,
            "max": max(stats["prompt_lengths"]) if stats["prompt_lengths"] else 0,
            "mean": sum(stats["prompt_lengths"]) / len(stats["prompt_lengths"]) if stats["prompt_lengths"] else 0,
        },
        "target_length": {
            "min": min(stats["target_lengths"]) if stats["target_lengths"] else 0,
            "max": max(stats["target_lengths"]) if stats["target_lengths"] else 0,
            "mean": sum(stats["target_lengths"]) / len(stats["target_lengths"]) if stats["target_lengths"] else 0,
        },
        "text_length": {
            "min": min(stats["text_lengths"]) if stats["text_lengths"] else 0,
            "max": max(stats["text_lengths"]) if stats["text_lengths"] else 0,
            "mean": sum(stats["text_lengths"]) / len(stats["text_lengths"]) if stats["text_lengths"] else 0,
        },
        "num_tools": {
            "min": min(stats["num_tools"]) if stats["num_tools"] else 0,
            "max": max(stats["num_tools"]) if stats["num_tools"] else 0,
            "mean": sum(stats["num_tools"]) / len(stats["num_tools"]) if stats["num_tools"] else 0,
        },
        "num_answers": {
            "min": min(stats["num_answers"]) if stats["num_answers"] else 0,
            "max": max(stats["num_answers"]) if stats["num_answers"] else 0,
            "mean": sum(stats["num_answers"]) / len(stats["num_answers"]) if stats["num_answers"] else 0,
        },
    }

    return summary


def filter_by_length(examples: List[Dict], max_length: int = 2048) -> List[Dict]:
    """
    Filter examples by text length.

    Args:
        examples: List of formatted examples
        max_length: Maximum character length for text field

    Returns:
        Filtered list of examples
    """
    filtered = []
    removed_count = 0

    for example in examples:
        text_length = len(example.get("text", ""))
        if text_length <= max_length:
            filtered.append(example)
        else:
            removed_count += 1

    if removed_count > 0:
        print(f"Removed {removed_count} examples exceeding max length {max_length}")

    return filtered
