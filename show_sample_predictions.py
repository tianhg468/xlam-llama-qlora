import json
import sys

checkpoint_file = "outputs/eval_checkpoints/lora_eval_checkpoint.json"

try:
    with open(checkpoint_file) as f:
        data = json.load(f)
    
    predictions = data.get("predictions", [])
    ground_truths = data.get("ground_truths", [])
    
    print("="*80)
    print("SAMPLE PREDICTIONS FROM LORA MODEL")
    print("="*80)
    print(f"Total predictions: {len(predictions)}\n")
    
    # Show first 10 examples
    num_samples = min(10, len(predictions))
    
    for i in range(num_samples):
        pred = predictions[i]
        gt = ground_truths[i] if i < len(ground_truths) else None
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        
        print(f"\n📊 Valid JSON: {pred.get('valid', False)}")
        
        if pred.get('valid'):
            print(f"\n🤖 PREDICTED:")
            pred_calls = pred.get('calls', [])
            print(f"   Raw: {json.dumps(pred_calls, indent=2)}")
            
            # Check for specific issues
            if isinstance(pred_calls, list):
                for call in pred_calls:
                    if isinstance(call, dict):
                        if "-name" in call:
                            print(f"   ⚠️  WARNING: Found '-name' key instead of 'name'!")
                        if "name" in call:
                            print(f"   ✓ Tool name: {call.get('name')}")
                        if "arguments" in call:
                            print(f"   ✓ Arguments: {call.get('arguments')}")
        else:
            print(f"\n🤖 PREDICTED (INVALID JSON):")
            print(f"   Raw text: {pred.get('raw_text', 'N/A')[:200]}")
        
        if gt:
            print(f"\n✅ GROUND TRUTH:")
            gt_calls = gt.get('calls', [])
            print(f"   {json.dumps(gt_calls, indent=2)}")
        
        print()
    
    # Summary statistics
    valid_count = sum(1 for p in predictions if p.get('valid'))
    invalid_count = len(predictions) - valid_count
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Valid JSON: {valid_count}/{len(predictions)} ({valid_count/len(predictions)*100:.1f}%)")
    print(f"Invalid JSON: {invalid_count}/{len(predictions)} ({invalid_count/len(predictions)*100:.1f}%)")
    
except FileNotFoundError:
    print(f"❌ Checkpoint file not found: {checkpoint_file}")
    print("   Make sure you're in the xlam-llama-qlora directory")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
