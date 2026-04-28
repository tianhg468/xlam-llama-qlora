"""
Data analysis tools for inspecting dataset quality and distribution.
"""

import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_text_lengths(data: List[Dict], title: str = "Text Length Distribution"):
    """Analyze and plot text length distribution."""
    text_lengths = [len(ex.get("text", "")) for ex in data]

    stats = {
        "count": len(text_lengths),
        "min": min(text_lengths),
        "max": max(text_lengths),
        "mean": sum(text_lengths) / len(text_lengths),
        "median": sorted(text_lengths)[len(text_lengths) // 2],
        "percentile_95": sorted(text_lengths)[int(len(text_lengths) * 0.95)],
        "percentile_99": sorted(text_lengths)[int(len(text_lengths) * 0.99)],
    }

    print(f"\n{title}")
    print("=" * 60)
    print(f"Count: {stats['count']:,}")
    print(f"Min: {stats['min']:,} chars")
    print(f"Max: {stats['max']:,} chars")
    print(f"Mean: {stats['mean']:.1f} chars")
    print(f"Median: {stats['median']:,} chars")
    print(f"95th percentile: {stats['percentile_95']:,} chars")
    print(f"99th percentile: {stats['percentile_99']:,} chars")

    # Plot histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(text_lengths, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel("Text Length (characters)")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.0f}")
        plt.axvline(stats['median'], color='g', linestyle='--', label=f"Median: {stats['median']}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = Path("outputs") / f"{title.lower().replace(' ', '_')}.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create plot: {e}")

    return stats


def analyze_tool_distribution(data: List[Dict], title: str = "Tool Distribution"):
    """Analyze tool usage distribution."""
    tool_counts = {}

    for ex in data:
        try:
            # Parse target to get answer
            target = ex.get("target", "")
            target = target.replace("<|eot_id|>", "").strip()
            if target:
                answers = json.loads(target)
                if isinstance(answers, list):
                    for answer in answers:
                        tool_name = answer.get("name", "unknown")
                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                else:
                    tool_name = answers.get("name", "unknown")
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        except:
            continue

    # Sort by frequency
    sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{title}")
    print("=" * 60)
    print(f"Unique tools: {len(sorted_tools)}")
    print(f"\nTop 20 most frequent tools:")
    for i, (tool, count) in enumerate(sorted_tools[:20], 1):
        pct = 100 * count / len(data)
        print(f"{i:2d}. {tool[:40]:<40} {count:>6,} ({pct:>5.2f}%)")

    return tool_counts


def analyze_answer_counts(data: List[Dict], title: str = "Answer Count Distribution"):
    """Analyze distribution of number of answers per example."""
    answer_counts = []

    for ex in data:
        try:
            target = ex.get("target", "")
            target = target.replace("<|eot_id|>", "").strip()
            if target:
                answers = json.loads(target)
                if isinstance(answers, list):
                    answer_counts.append(len(answers))
                else:
                    answer_counts.append(1)
        except:
            pass

    if not answer_counts:
        print(f"No valid answer counts found in {title}")
        return

    from collections import Counter
    count_dist = Counter(answer_counts)

    print(f"\n{title}")
    print("=" * 60)
    print(f"Mean answers per example: {sum(answer_counts) / len(answer_counts):.2f}")
    print(f"\nDistribution:")
    for count in sorted(count_dist.keys()):
        freq = count_dist[count]
        pct = 100 * freq / len(answer_counts)
        print(f"  {count} answer(s): {freq:>6,} examples ({pct:>5.2f}%)")

    return count_dist


def main():
    """Analyze all dataset splits."""
    data_dir = Path(__file__).parent.parent.parent / "data"

    # Check if data exists
    if not (data_dir / "train.jsonl").exists():
        print("Error: Data not found. Run data_prep.py first.")
        return

    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)

    # Load data
    print("\nLoading datasets...")
    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data = load_jsonl(data_dir / "val.jsonl")
    test_data = load_jsonl(data_dir / "test.jsonl")
    print(f"Train: {len(train_data):,} examples")
    print(f"Val: {len(val_data):,} examples")
    print(f"Test: {len(test_data):,} examples")

    # Analyze text lengths
    analyze_text_lengths(train_data, "Train Set - Text Length Distribution")
    analyze_text_lengths(val_data, "Val Set - Text Length Distribution")
    analyze_text_lengths(test_data, "Test Set - Text Length Distribution")

    # Analyze tool distribution
    analyze_tool_distribution(train_data, "Train Set - Tool Distribution")

    # Analyze answer counts
    analyze_answer_counts(train_data, "Train Set - Answer Count Distribution")

    # Load and display statistics if available
    stats_file = data_dir / "dataset_statistics.json"
    if stats_file.exists():
        print("\n" + "="*80)
        print("DATASET STATISTICS (from optimized data prep)")
        print("="*80)
        with open(stats_file, "r") as f:
            stats = json.load(f)
        print(json.dumps(stats, indent=2))

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
