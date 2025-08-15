#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_bf16_support():
    print("Testing bf16 support...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # Test model loading with bf16
    try:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
        
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("\nLoading model with bf16...")
            model = AutoModelForCausalLM.from_pretrained(
                'Qwen/Qwen2.5-1.5B-Instruct',
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            print("\nbf16 not supported, loading with fp32...")
            model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
        
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test a simple forward pass
        print("\nTesting forward pass...")
        input_text = "Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"Output logits dtype: {outputs.logits.dtype}")
        print(f"Output logits shape: {outputs.logits.shape}")
        print("bf16 test completed successfully!")
        
    except Exception as e:
        print(f"Error during bf16 test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bf16_support() 