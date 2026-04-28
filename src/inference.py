"""
Single-query inference script using LoRA-finetuned model.
For sanity-checking and demonstration purposes.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def load_config() -> Dict:
    """Load configuration from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_quantization_config(config: Dict) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig for 4-bit NF4 quantization."""
    quant_cfg = config["quantization"]
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


def format_prompt(query: str, tools: list) -> str:
    """
    Format a query and tools into the Llama 3.1 prompt format.

    Args:
        query: User query string
        tools: List of tool definitions (dicts with 'name', 'description', 'parameters')

    Returns:
        Formatted prompt string
    """
    tools_json = json.dumps(tools, indent=2)

    system_message = f"""You are a helpful AI assistant with access to a set of tools. When the user makes a request, decide which tool(s) to call and respond ONLY with a JSON list of tool calls in this format:

[{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}]

Available tools:
{tools_json}"""

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{query}<|eot_id|>{RESPONSE_TEMPLATE}"
    )

    return prompt


def load_model_and_tokenizer(config: Dict, adapter_path: Path):
    """Load base model with LoRA adapter."""
    print("Loading base model with 4-bit quantization...")
    bnb_config = setup_quantization_config(config)

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_tool_calls(model, tokenizer, prompt: str, config: Dict) -> str:
    """Generate tool calls for the given prompt."""
    inf_cfg = config["inference"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inf_cfg["max_new_tokens"],
            temperature=inf_cfg["temperature"] if inf_cfg["temperature"] > 0 else None,
            do_sample=inf_cfg["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def parse_tool_calls(response: str):
    """Parse tool calls from model response."""
    response = response.strip()

    # Try direct parse
    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array
    start_idx = response.find("[")
    end_idx = response.rfind("]")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            json_str = response[start_idx:end_idx+1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None


def main():
    """Run inference on a single query."""
    parser = argparse.ArgumentParser(description="Run inference with LoRA-finetuned model")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="JSON string or file path containing tool definitions"
    )
    args = parser.parse_args()

    config = load_config()
    project_root = Path(__file__).parent.parent
    adapter_path = project_root / config["training"]["output_dir"]

    # Load model
    model, tokenizer = load_model_and_tokenizer(config, adapter_path)

    # Parse tools
    if args.tools:
        # Check if it's a file path
        tools_path = Path(args.tools)
        if tools_path.exists():
            with open(tools_path, "r") as f:
                tools = json.load(f)
        else:
            # Assume it's a JSON string
            tools = json.loads(args.tools)
    else:
        # Default example tools
        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]

    # Format prompt
    prompt = format_prompt(args.query, tools)

    print("\n" + "="*80)
    print("INFERENCE")
    print("="*80)
    print(f"Query: {args.query}")
    print(f"Tools: {len(tools)} available")

    # Generate
    print("\nGenerating response...")
    response = generate_tool_calls(model, tokenizer, prompt, config)

    print("\n" + "="*80)
    print("RAW RESPONSE:")
    print("="*80)
    print(response)

    # Parse
    parsed = parse_tool_calls(response)

    print("\n" + "="*80)
    print("PARSED TOOL CALLS:")
    print("="*80)

    if parsed is not None:
        print(json.dumps(parsed, indent=2))
    else:
        print("Failed to parse tool calls from response")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
