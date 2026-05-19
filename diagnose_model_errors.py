"""
Comprehensive diagnostic of model errors to understand training failure.
"""
import json
from pathlib import Path
from difflib import SequenceMatcher


def similarity(a, b):
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def analyze_checkpoint(checkpoint_file):
    """Analyze errors in checkpoint predictions."""
    with open(checkpoint_file) as f:
        data = json.load(f)

    predictions = data["predictions"]
    ground_truths = data["ground_truths"]

    print("="*80)
    print("COMPREHENSIVE ERROR ANALYSIS")
    print("="*80)

    # Error categories
    invalid_json_count = 0
    space_name_bug_count = 0
    tool_name_typo_count = 0
    tool_name_hallucination_count = 0
    naming_convention_mismatch_count = 0
    arg_type_mismatch_count = 0
    arg_value_wrong_count = 0

    tool_name_errors = []
    arg_errors = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths), 1):
        # Invalid JSON
        if not pred.get("valid"):
            invalid_json_count += 1
            continue

        pred_calls = pred.get("calls", [])
        gt_calls = gt.get("calls", [])

        if not pred_calls or not gt_calls:
            continue

        # Analyze each call
        for pred_call in pred_calls:
            if not isinstance(pred_call, dict):
                continue

            # Check for space-before-name bug
            if " name" in pred_call or "-name" in pred_call:
                space_name_bug_count += 1

            pred_tool = pred_call.get("name", "")
            pred_args = pred_call.get("arguments", {})

            # Find best matching ground truth call
            best_match = None
            best_similarity = 0
            for gt_call in gt_calls:
                if isinstance(gt_call, dict):
                    gt_tool = gt_call.get("name", "")
                    sim = similarity(pred_tool, gt_tool)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = gt_call

            if best_match:
                gt_tool = best_match.get("name", "")
                gt_args = best_match.get("arguments", {})

                # Tool name analysis
                if pred_tool != gt_tool:
                    error_info = {
                        "example": i,
                        "predicted": pred_tool,
                        "expected": gt_tool,
                        "similarity": best_similarity
                    }

                    # Categorize error
                    if best_similarity > 0.8:
                        # Likely a typo
                        tool_name_typo_count += 1
                        error_info["type"] = "typo"
                    elif best_similarity > 0.5:
                        # Partial match - hallucination
                        tool_name_hallucination_count += 1
                        error_info["type"] = "hallucination"
                    else:
                        # Completely wrong
                        tool_name_hallucination_count += 1
                        error_info["type"] = "complete_hallucination"

                    # Check naming convention
                    if "_" in gt_tool and pred_tool and "_" not in pred_tool:
                        naming_convention_mismatch_count += 1
                        error_info["naming_issue"] = "snake_case -> camelCase"
                    elif "_" not in gt_tool and pred_tool and "_" in pred_tool:
                        naming_convention_mismatch_count += 1
                        error_info["naming_issue"] = "camelCase -> snake_case"

                    tool_name_errors.append(error_info)

                # Argument analysis (only if tool name matches)
                if pred_tool == gt_tool:
                    for key in gt_args:
                        gt_val = gt_args[key]
                        pred_val = pred_args.get(key)

                        if pred_val is None:
                            arg_errors.append({
                                "example": i,
                                "tool": pred_tool,
                                "arg": key,
                                "issue": "missing_arg"
                            })
                        elif type(pred_val) != type(gt_val):
                            arg_type_mismatch_count += 1
                            arg_errors.append({
                                "example": i,
                                "tool": pred_tool,
                                "arg": key,
                                "issue": "type_mismatch",
                                "predicted": f"{pred_val} ({type(pred_val).__name__})",
                                "expected": f"{gt_val} ({type(gt_val).__name__})"
                            })
                        elif pred_val != gt_val:
                            arg_value_wrong_count += 1
                            arg_errors.append({
                                "example": i,
                                "tool": pred_tool,
                                "arg": key,
                                "issue": "wrong_value",
                                "predicted": pred_val,
                                "expected": gt_val
                            })

    # Print summary
    total = len(predictions)
    print(f"\nTotal examples: {total}")
    print(f"\n{'='*80}")
    print("ERROR SUMMARY")
    print(f"{'='*80}")
    print(f"Invalid JSON: {invalid_json_count}/{total} ({invalid_json_count/total*100:.1f}%)")
    print(f"Space-before-name bug: {space_name_bug_count}")
    print(f"Tool name typos: {tool_name_typo_count}")
    print(f"Tool name hallucinations: {tool_name_hallucination_count}")
    print(f"Naming convention mismatches: {naming_convention_mismatch_count}")
    print(f"Argument type mismatches: {arg_type_mismatch_count}")
    print(f"Argument value errors: {arg_value_wrong_count}")

    # Detailed errors
    if tool_name_errors:
        print(f"\n{'='*80}")
        print("TOOL NAME ERRORS (first 10)")
        print(f"{'='*80}")
        for err in tool_name_errors[:10]:
            print(f"\nExample {err['example']}: {err.get('type', 'unknown')}")
            print(f"  Predicted: {err['predicted']}")
            print(f"  Expected:  {err['expected']}")
            print(f"  Similarity: {err['similarity']:.2f}")
            if 'naming_issue' in err:
                print(f"  Naming: {err['naming_issue']}")

    if arg_errors:
        print(f"\n{'='*80}")
        print("ARGUMENT ERRORS (first 10)")
        print(f"{'='*80}")
        for err in arg_errors[:10]:
            print(f"\nExample {err['example']} - {err['tool']}.{err['arg']}")
            print(f"  Issue: {err['issue']}")
            if 'predicted' in err:
                print(f"  Predicted: {err['predicted']}")
                print(f"  Expected:  {err['expected']}")

    # Conclusion
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")

    if invalid_json_count > total * 0.3:
        print("❌ CRITICAL: >30% invalid JSON - model didn't learn task structure")

    if tool_name_hallucination_count > 3:
        print("❌ CRITICAL: Model is hallucinating tool names - training failed")

    if naming_convention_mismatch_count > 2:
        print("⚠️  WARNING: Naming convention confusion (snake_case vs camelCase)")

    if space_name_bug_count > 2:
        print("⚠️  WARNING: Persistent space-before-name formatting bug")

    print("\n💡 RECOMMENDATION:")
    if invalid_json_count > total * 0.3 or tool_name_hallucination_count > 3:
        print("   The model training FAILED. Possible causes:")
        print("   1. Training data was corrupted during preparation")
        print("   2. Learning rate too high (model diverged)")
        print("   3. Not enough training steps (model didn't converge)")
        print("   4. LoRA rank too low (insufficient capacity)")
        print("")
        print("   NEXT STEPS:")
        print("   1. Check wandb training loss - did it actually decrease?")
        print("   2. Verify training data format is correct")
        print("   3. Try inference with base model to compare")
        print("   4. Consider re-training with adjusted hyperparameters")


def main():
    checkpoint_file = Path("outputs/eval_checkpoints/LoRA_Model_checkpoint.json")

    if not checkpoint_file.exists():
        print(f"❌ Checkpoint not found: {checkpoint_file}")
        return

    analyze_checkpoint(checkpoint_file)


if __name__ == "__main__":
    main()
