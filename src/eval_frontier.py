"""
Evaluation comparing your LoRA model against frontier models.
Tests on the same xLAM test set and compares tool calling performance.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def load_config() -> Dict:
    """Load configuration from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_test_data(data_dir: Path, num_samples: int = None) -> List[Dict]:
    """Load test dataset from JSONL."""
    test_path = data_dir / "test.jsonl"
    examples = []
    with open(test_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
            if num_samples and len(examples) >= num_samples:
                break
    return examples


def parse_tool_calls(response: str) -> Tuple[bool, List[Dict]]:
    """
    Parse model response as JSON tool calls.

    Returns:
        (is_valid, parsed_calls)
    """
    response = response.strip()

    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return True, parsed
        elif isinstance(parsed, dict):
            return True, [parsed]
        else:
            return False, []
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in response
    start_idx = response.find("[")
    end_idx = response.rfind("]")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            json_str = response[start_idx:end_idx+1]
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return True, parsed
        except json.JSONDecodeError:
            pass

    return False, []


def compute_metrics(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """Compute evaluation metrics."""
    assert len(predictions) == len(ground_truths)

    json_valid_count = 0
    tool_name_correct = 0
    arg_key_correct = 0
    arg_value_exact = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        if pred["valid"]:
            json_valid_count += 1

        if not pred["valid"]:
            continue

        pred_calls = pred["calls"]
        gt_calls = gt["calls"]

        pred_tool_names = set(call.get("name", "") for call in pred_calls)
        gt_tool_names = set(call.get("name", "") for call in gt_calls)

        if pred_tool_names == gt_tool_names:
            tool_name_correct += 1

            pred_tools_map = {call.get("name", ""): call.get("arguments", {}) for call in pred_calls}
            gt_tools_map = {call.get("name", ""): call.get("arguments", {}) for call in gt_calls}

            all_keys_match = True
            all_values_match = True

            for tool_name in gt_tool_names:
                pred_args = pred_tools_map.get(tool_name, {})
                gt_args = gt_tools_map.get(tool_name, {})

                if set(pred_args.keys()) != set(gt_args.keys()):
                    all_keys_match = False
                    all_values_match = False
                    break

                for key in gt_args.keys():
                    pred_val = pred_args.get(key)
                    gt_val = gt_args[key]

                    if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
                        if abs(pred_val - gt_val) > 1e-6:
                            all_values_match = False
                    elif pred_val != gt_val:
                        all_values_match = False

            if all_keys_match:
                arg_key_correct += 1

            if all_values_match:
                arg_value_exact += 1

    return {
        "json_validity_rate": json_valid_count / total if total > 0 else 0.0,
        "tool_name_accuracy": tool_name_correct / total if total > 0 else 0.0,
        "arg_key_accuracy": arg_key_correct / total if total > 0 else 0.0,
        "arg_value_exact_accuracy": arg_value_exact / total if total > 0 else 0.0,
        "total_examples": total,
    }


class FrontierModelClient:
    """Base class for frontier model API clients."""

    def generate(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        """Generate a response from the model."""
        raise NotImplementedError


class OpenAIClient(FrontierModelClient):
    """OpenAI API client for GPT models."""

    def __init__(self, model: str = "gpt-4-turbo", api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        """Generate response using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]

        try:
            if tools:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )

                # Extract tool calls
                if response.choices[0].message.tool_calls:
                    tool_calls = []
                    for tc in response.choices[0].message.tool_calls:
                        tool_calls.append({
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments)
                        })
                    return json.dumps(tool_calls)
                else:
                    return response.choices[0].message.content or ""
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                return response.choices[0].message.content or ""

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""


class AnthropicClient(FrontierModelClient):
    """Anthropic API client for Claude models."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        """Generate response using Anthropic API."""
        try:
            if tools:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    tools=tools,
                )

                # Extract tool calls
                tool_calls = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({
                            "name": block.name,
                            "arguments": block.input
                        })

                if tool_calls:
                    return json.dumps(tool_calls)
                else:
                    # Return text content if no tool calls
                    text_content = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            text_content += block.text
                    return text_content
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text if response.content else ""

        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return ""


class GeminiClient(FrontierModelClient):
    """Google Gemini API client."""

    def __init__(self, model: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")

        self.model = model
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)

    def generate(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        """Generate response using Gemini API."""
        try:
            if tools:
                # Convert tools to Gemini format
                import google.generativeai as genai

                gemini_tools = []
                for tool in tools:
                    gemini_tools.append(genai.protos.Tool(
                        function_declarations=[
                            genai.protos.FunctionDeclaration(
                                name=tool["function"]["name"],
                                description=tool["function"]["description"],
                                parameters=tool["function"]["parameters"]
                            )
                        ]
                    ))

                response = self.client.generate_content(
                    prompt,
                    tools=gemini_tools,
                )

                # Extract tool calls
                if response.candidates[0].content.parts:
                    tool_calls = []
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "function_call"):
                            fc = part.function_call
                            tool_calls.append({
                                "name": fc.name,
                                "arguments": dict(fc.args)
                            })

                    if tool_calls:
                        return json.dumps(tool_calls)

                return response.text if hasattr(response, "text") else ""
            else:
                response = self.client.generate_content(prompt)
                return response.text if hasattr(response, "text") else ""

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return ""


def evaluate_frontier_model(
    client: FrontierModelClient,
    test_data: List[Dict],
    model_name: str,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate a frontier model on test data.

    Returns:
        (metrics_dict, predictions_list)
    """
    print(f"\nEvaluating {model_name}...")

    predictions = []
    ground_truths = []

    for idx, example in enumerate(test_data):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(test_data)} examples")

        prompt = example["prompt"]
        target = example["target"]

        # Parse ground truth
        target_json_str = target.replace("<|eot_id|>", "").strip()
        try:
            gt_calls = json.loads(target_json_str)
            if not isinstance(gt_calls, list):
                gt_calls = [gt_calls]
        except json.JSONDecodeError:
            gt_calls = []

        ground_truths.append({"calls": gt_calls})

        # Generate prediction
        # Note: Frontier models may need tool definitions extracted from prompt
        response = client.generate(prompt)
        is_valid, pred_calls = parse_tool_calls(response)

        predictions.append({
            "valid": is_valid,
            "calls": pred_calls,
            "raw_response": response,
        })

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)
    print(f"\n{model_name} Results:")
    print(f"  JSON Validity: {metrics['json_validity_rate']:.2%}")
    print(f"  Tool Name Accuracy: {metrics['tool_name_accuracy']:.2%}")
    print(f"  Arg Key Accuracy: {metrics['arg_key_accuracy']:.2%}")
    print(f"  Arg Value Exact: {metrics['arg_value_exact_accuracy']:.2%}")

    return metrics, predictions


def main():
    """Run evaluation comparing your model with frontier models."""
    config = load_config()
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "frontier_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    frontier_config = config.get("frontier_models", {})
    enabled_models = frontier_config.get("enabled", [])
    num_samples = frontier_config.get("num_test_samples", 100)

    print("="*80)
    print("FRONTIER MODEL COMPARISON")
    print("="*80)
    print(f"Evaluating on {num_samples} test examples")
    print(f"Enabled models: {enabled_models}")

    # Load test data
    print(f"\nLoading test data...")
    test_data = load_test_data(data_dir, num_samples)
    print(f"Loaded {len(test_data)} test examples")

    # Initialize model clients
    clients = {}

    if "gpt-4-turbo" in enabled_models:
        try:
            clients["GPT-4-Turbo"] = OpenAIClient(model="gpt-4-turbo")
        except Exception as e:
            print(f"Warning: Could not initialize GPT-4-Turbo: {e}")

    if "gpt-4o" in enabled_models:
        try:
            clients["GPT-4o"] = OpenAIClient(model="gpt-4o")
        except Exception as e:
            print(f"Warning: Could not initialize GPT-4o: {e}")

    if "gpt-3.5-turbo" in enabled_models:
        try:
            clients["GPT-3.5-Turbo"] = OpenAIClient(model="gpt-3.5-turbo")
        except Exception as e:
            print(f"Warning: Could not initialize GPT-3.5-Turbo: {e}")

    if "claude-3-5-sonnet" in enabled_models:
        try:
            clients["Claude-3.5-Sonnet"] = AnthropicClient(model="claude-3-5-sonnet-20241022")
        except Exception as e:
            print(f"Warning: Could not initialize Claude-3.5-Sonnet: {e}")

    if "claude-3-opus" in enabled_models:
        try:
            clients["Claude-3-Opus"] = AnthropicClient(model="claude-3-opus-20240229")
        except Exception as e:
            print(f"Warning: Could not initialize Claude-3-Opus: {e}")

    if "gemini-1.5-pro" in enabled_models:
        try:
            clients["Gemini-1.5-Pro"] = GeminiClient(model="gemini-1.5-pro")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini-1.5-Pro: {e}")

    if not clients:
        print("\nError: No frontier models could be initialized.")
        print("Please check your API keys and enabled models in config.yaml")
        return

    # Evaluate each model
    all_results = {}

    for model_name, client in clients.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")

        metrics, predictions = evaluate_frontier_model(
            client,
            test_data,
            model_name,
        )

        all_results[model_name] = {
            "metrics": metrics,
            "predictions": predictions,
        }

    # Load your model's results if available
    your_results_path = project_root / "outputs" / "eval_results.json"
    if your_results_path.exists():
        with open(your_results_path, "r") as f:
            your_results = json.load(f)
            all_results["Your LoRA Model"] = {
                "metrics": your_results["lora_model"]["metrics"],
            }

    # Save combined results
    results = {
        "config": {
            "num_test_samples": len(test_data),
            "models_evaluated": list(all_results.keys()),
        },
        "results": all_results,
    }

    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)

    # Extract all model names and metrics
    model_names = list(all_results.keys())
    metric_names = [
        "json_validity_rate",
        "tool_name_accuracy",
        "arg_key_accuracy",
        "arg_value_exact_accuracy"
    ]
    metric_display = [
        "JSON Validity",
        "Tool Name Acc",
        "Arg Key Acc",
        "Arg Value Exact"
    ]

    # Print header
    print(f"{'Model':<25}", end="")
    for metric in metric_display:
        print(f"{metric:>15}", end="")
    print()
    print("-"*100)

    # Print each model's results
    for model_name in model_names:
        print(f"{model_name:<25}", end="")
        metrics = all_results[model_name]["metrics"]
        for metric_name in metric_names:
            value = metrics.get(metric_name, 0.0)
            print(f"{value:>14.2%}", end="")
        print()

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
