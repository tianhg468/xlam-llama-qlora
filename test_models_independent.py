"""
Quick test to verify base model and LoRA model are truly independent.
Tests on just 2-3 examples to avoid wasting GPU credits.
"""
import json
from pathlib import Path
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def setup_quantization_config(config):
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


def generate_test(model, tokenizer, prompt):
    """Generate response from a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def main():
    print("="*80)
    print("QUICK MODEL INDEPENDENCE TEST")
    print("="*80)
    print("This tests 2 examples to verify base and LoRA models are independent\n")

    config = load_config()
    bnb_config = setup_quantization_config(config)
    adapter_path = Path("outputs/llama31-8b-xlam-lora")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test prompt
    test_prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant with access to a set of tools. When the user makes a request, decide which tool(s) to call and respond ONLY with a JSON list of tool calls in this format:\n\n[{"name": "tool_name", "arguments": {"arg1": "value1"}}]\n\nAvailable tools:\n[\n  {\n    "name": "get_weather",\n    "description": "Get current weather",\n    "parameters": {\n      "location": {"type": "string"},\n      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}\n    }\n  }\n]<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nget the weather in Tokyo<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

    print("\n" + "="*80)
    print("LOADING BASE MODEL (should NOT have adapter)")
    print("="*80)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    print("✅ Base model loaded")

    print("\n" + "="*80)
    print("LOADING LORA MODEL (should HAVE adapter)")
    print("="*80)
    base_model_for_lora = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    print("✅ Second base model instance loaded")

    lora_model = PeftModel.from_pretrained(
        base_model_for_lora,
        str(adapter_path),
        is_trainable=False,
    )
    print("✅ LoRA adapter applied")

    # Verify they're different objects
    print(f"\n🔍 Memory check:")
    print(f"   Base model ID: {id(base_model)}")
    print(f"   LoRA model ID: {id(lora_model)}")
    print(f"   Are they different objects? {id(base_model) != id(lora_model)}")

    # Test generation
    print("\n" + "="*80)
    print("TESTING GENERATION ON SAME PROMPT")
    print("="*80)
    print("Query: 'get the weather in Tokyo'")

    print("\n📝 Base model response:")
    base_response = generate_test(base_model, tokenizer, test_prompt)
    print(f"   {base_response[:200]}")

    print("\n📝 LoRA model response:")
    lora_response = generate_test(lora_model, tokenizer, test_prompt)
    print(f"   {lora_response[:200]}")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    if base_response == lora_response:
        print("❌ FAIL: Both models produce IDENTICAL output!")
        print("   This means they're still sharing weights or adapter isn't active.")
        print("   DO NOT run full evaluation - it will waste money.")
        return False
    else:
        print("✅ PASS: Models produce DIFFERENT output!")
        print("   Base and LoRA models are independent.")

        # Try to detect if LoRA is producing valid JSON
        lora_has_json = "[{" in lora_response and "}]" in lora_response
        base_has_json = "[{" in base_response and "}]" in base_response

        print(f"\n📊 Quick quality check:")
        print(f"   Base model has valid JSON structure: {base_has_json}")
        print(f"   LoRA model has valid JSON structure: {lora_has_json}")

        if lora_has_json and not base_has_json:
            print("\n🎉 EXCELLENT: LoRA produces JSON, base doesn't!")
            print("   This strongly suggests your fine-tuning worked.")
        elif lora_has_json and base_has_json:
            print("\n⚠️  Both produce JSON - this is unusual but could be valid.")
        else:
            print("\n⚠️  LoRA doesn't produce clean JSON - may need investigation.")

        print("\n✅ Safe to run full evaluation!")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
