#!/usr/bin/env python3
"""
Test GRPO training without Unsloth dependency
Uses fallback implementation for testing and demonstration
"""

import os
import sys
import torch
import json
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grpo_unsloth_trainer import UnslothGRPOTrainer, UnslothGRPODataset
from config_20b import GPT20BGRPOConfig

def create_test_dataset():
    """Create a small test dataset"""
    test_data = [
        {"prompt": "What is 2+2?", "answer": "4", "instruction": "Solve math", "input": ""},
        {"prompt": "What is 3+3?", "answer": "6", "instruction": "Solve math", "input": ""},
        {"prompt": "What is 4+4?", "answer": "8", "instruction": "Solve math", "input": ""},
        {"prompt": "What is 5+5?", "answer": "10", "instruction": "Solve math", "input": ""},
        {"prompt": "What is 6+6?", "answer": "12", "instruction": "Solve math", "input": ""},
    ] * 4  # 20 samples total
    
    return test_data

def test_fallback_training():
    """Test training with fallback implementation"""
    print("🧪 Testing GRPO Training with Fallback Implementation")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a GPU.")
        return False
    
    print(f"✅ CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create configuration
    print("\n📋 Creating configuration...")
    config = GPT20BGRPOConfig()
    print(f"   Model: {config.model_name}")
    print(f"   Max sequence length: {config.max_seq_length}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    
    # Create test dataset
    print("\n📊 Creating test dataset...")
    test_data = create_test_dataset()
    print(f"   Created {len(test_data)} samples")
    
    # Save dataset
    dataset_path = "./data/test_dataset.json"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"   Dataset saved to: {dataset_path}")
    
    try:
        # Initialize trainer (will use fallback)
        print("\n🤖 Initializing trainer...")
        trainer = UnslothGRPOTrainer(config)
        print("   ✅ Trainer initialized with fallback model")
        
        # Create dataset
        print("\n📚 Loading dataset...")
        dataset = UnslothGRPODataset(dataset_path, trainer.tokenizer, max_samples=20)
        print(f"   ✅ Dataset loaded with {len(dataset)} samples")
        
        # Test generation
        print("\n🎯 Testing generation...")
        test_prompts = ["What is 2+2?", "What is 3+3?"]
        responses = trainer.generate_responses(test_prompts, use_vllm=False)
        
        for prompt, response in zip(test_prompts, responses):
            print(f"   Prompt: {prompt}")
            print(f"   Response: {response[:100]}...")
            print()
        
        # Test reward computation
        print("🎁 Testing reward computation...")
        rewards = trainer.compute_rewards(test_prompts, responses, ["4", "6"])
        print(f"   Rewards: {rewards.tolist()}")
        
        # Test advantage computation
        print("📈 Testing advantage computation...")
        advantages = trainer.compute_advantages(rewards)
        print(f"   Advantages: {advantages.tolist()}")
        
        # Simulate a few training steps
        print("\n🏋️ Simulating training steps...")
        for step in range(3):
            # Create a small batch
            batch = {
                'prompt': test_prompts,
                'answer': ["4", "6"]
            }
            
            # Run training step
            metrics = trainer.train_step(batch)
            print(f"   Step {step + 1}: Loss={metrics['loss']:.4f}, "
                  f"Rewards={metrics['rewards_mean']:.4f}")
        
        print("\n✅ All tests completed successfully!")
        print("\n🎉 Fallback implementation is working correctly!")
        print("\nNext steps:")
        print("1. Install Unsloth for full functionality: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        print("2. Install vLLM for faster generation: pip install vllm")
        print("3. Use with your actual dataset and 20B model")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 GRPO Training Test (Fallback Mode)")
    print("   Testing without Unsloth dependency")
    print("   Optimized for NVIDIA 4080 Super")
    print("=" * 60)
    
    success = test_fallback_training()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("The fallback implementation works and is ready for testing.")
    else:
        print("\n❌ Test failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
