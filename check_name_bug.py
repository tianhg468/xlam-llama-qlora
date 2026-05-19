import json
import sys

checkpoint_file = "outputs/eval_checkpoints/LoRA_Model_checkpoint.json"

try:
    with open(checkpoint_file) as f:
        data = json.load(f)
    
    predictions = data.get("predictions", [])
    
    dash_name_count = 0
    valid_name_count = 0
    total_calls = 0
    
    for pred in predictions:
        if not pred.get("valid"):
            continue
            
        calls = pred.get("calls", [])
        if not isinstance(calls, list):
            continue
            
        for call in calls:
            if not isinstance(call, dict):
                continue
            total_calls += 1
            
            # Check if it has "-name" key
            if "-name" in call:
                dash_name_count += 1
                
            # Check if it has "name" key  
            if "name" in call:
                valid_name_count += 1
    
    print("="*60)
    print("DIAGNOSIS: Checking for -name bug")
    print("="*60)
    print(f"Total tool calls analyzed: {total_calls}")
    print(f"Calls with '-name' key: {dash_name_count} ({dash_name_count/total_calls*100 if total_calls > 0 else 0:.1f}%)")
    print(f"Calls with 'name' key: {valid_name_count} ({valid_name_count/total_calls*100 if total_calls > 0 else 0:.1f}%)")
    print()
    
    if dash_name_count > total_calls * 0.1:  # More than 10% have the bug
        print("❌ CRITICAL: Widespread '-name' bug detected!")
        print("   This explains the 0% tool name accuracy.")
        print("   The model is outputting '-name' instead of 'name' in JSON.")
        print()
        print("   Possible causes:")
        print("   1. Training data was corrupted/malformed")
        print("   2. Tokenization issue during training")
        print("   3. Model learned incorrect JSON format")
    elif dash_name_count > 0:
        print("⚠️  Some '-name' occurrences found but not widespread.")
        print("   May not fully explain the poor metrics.")
    else:
        print("✅ No '-name' bug found.")
        print("   The 0% metrics must be caused by something else.")
        
except FileNotFoundError:
    print(f"❌ Checkpoint file not found: {checkpoint_file}")
    print("   Make sure you're in the xlam-llama-qlora directory")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
