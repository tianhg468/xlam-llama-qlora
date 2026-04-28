#!/bin/bash

# Run all evaluations for comprehensive model assessment
# Usage: ./run_all_evals.sh

set -e  # Exit on error

echo "=================================="
echo "COMPREHENSIVE EVALUATION PIPELINE"
echo "=================================="
echo ""

# Check if in correct directory
if [ ! -f "src/eval.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# 1. Custom xLAM Evaluation
echo "Step 1/3: Running custom xLAM evaluation..."
echo "------------------------------------------"
python src/eval.py
echo ""
echo "✓ Custom xLAM evaluation complete"
echo ""

# 2. LM Evaluation Harness
echo "Step 2/3: Running LM Evaluation Harness (MMLU)..."
echo "---------------------------------------------------"
if command -v lm_eval &> /dev/null; then
    python src/eval_lm_harness.py
    echo ""
    echo "✓ LM Harness evaluation complete"
    echo ""
else
    echo "⚠ Warning: lm_eval not found. Skipping LM Harness evaluation."
    echo "  Install with: pip install lm-eval>=0.4.0"
    echo ""
fi

# 3. Frontier Model Comparison (optional - requires API keys)
echo "Step 3/3: Running frontier model comparison..."
echo "-----------------------------------------------"

# Check if any API keys are set
if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ]; then
    echo "API keys detected. Running frontier comparison..."
    python src/eval_frontier.py
    echo ""
    echo "✓ Frontier model comparison complete"
    echo ""
else
    echo "⚠ No API keys detected. Skipping frontier model comparison."
    echo "  To enable, set environment variables:"
    echo "    export OPENAI_API_KEY=sk-..."
    echo "    export ANTHROPIC_API_KEY=sk-ant-..."
    echo "    export GOOGLE_API_KEY=..."
    echo ""
fi

# Summary
echo ""
echo "=================================="
echo "EVALUATION PIPELINE COMPLETE"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  1. outputs/eval_results.json"
echo "  2. outputs/lm_harness_results/"
echo "  3. outputs/frontier_comparison/"
echo ""
echo "See EVALUATION.md for details on interpreting results."
