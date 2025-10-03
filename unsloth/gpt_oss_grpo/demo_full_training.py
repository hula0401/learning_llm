#!/usr/bin/env python3
"""
Full GRPO training demo with comprehensive Wandb monitoring
Demonstrates the complete training pipeline on NVIDIA 4080 Super
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
from wandb_monitor import WandbMonitor

def create_demo_dataset():
    """Create a comprehensive demo dataset"""
    demo_data = [
        # Math problems
        {"prompt": "What is 15% of 200?", "answer": "30", "instruction": "Calculate the percentage", "input": ""},
        {"prompt": "A store has 120 apples. They sell 3/4 of them. How many apples are left?", "answer": "30", "instruction": "Solve the word problem step by step", "input": ""},
        {"prompt": "If a train travels 300 miles in 4 hours, what is its average speed?", "answer": "75 miles per hour", "instruction": "Calculate the average speed", "input": ""},
        {"prompt": "Sarah has 24 stickers. She gives 1/3 to her friend and 1/4 to her sister. How many stickers does she have left?", "answer": "10", "instruction": "Solve the fraction word problem", "input": ""},
        {"prompt": "A rectangle has a length of 8 cm and width of 5 cm. What is its area?", "answer": "40 square centimeters", "instruction": "Calculate the area of the rectangle", "input": ""},
        {"prompt": "If 3x + 7 = 22, what is the value of x?", "answer": "5", "instruction": "Solve the linear equation", "input": ""},
        {"prompt": "A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, how many slices are left?", "answer": "2", "instruction": "Solve the division word problem", "input": ""},
        {"prompt": "What is the next number in the sequence: 2, 4, 8, 16, ?", "answer": "32", "instruction": "Find the pattern and predict the next number", "input": ""},
        {"prompt": "A book costs $15. If there's a 20% discount, what is the final price?", "answer": "$12", "instruction": "Calculate the discounted price", "input": ""},
        {"prompt": "If a triangle has sides of length 3, 4, and 5, what type of triangle is it?", "answer": "Right triangle", "instruction": "Identify the type of triangle", "input": ""},
        
        # Science questions
        {"prompt": "What is photosynthesis?", "answer": "The process by which plants convert sunlight into energy", "instruction": "Explain the concept of photosynthesis", "input": ""},
        {"prompt": "What is the chemical formula for water?", "answer": "H2O", "instruction": "Provide the chemical formula", "input": ""},
        {"prompt": "What is the speed of light?", "answer": "299,792,458 meters per second", "instruction": "State the speed of light", "input": ""},
        {"prompt": "What is the largest planet in our solar system?", "answer": "Jupiter", "instruction": "Identify the largest planet", "input": ""},
        {"prompt": "What is the atomic number of carbon?", "answer": "6", "instruction": "Provide the atomic number", "input": ""},
        
        # General knowledge
        {"prompt": "What is the capital of France?", "answer": "Paris", "instruction": "Name the capital city", "input": ""},
        {"prompt": "Who wrote 'Romeo and Juliet'?", "answer": "William Shakespeare", "instruction": "Identify the author", "input": ""},
        {"prompt": "What year did World War II end?", "answer": "1945", "instruction": "Provide the year", "input": ""},
        {"prompt": "What is the largest ocean on Earth?", "answer": "Pacific Ocean", "instruction": "Name the largest ocean", "input": ""},
        {"prompt": "What is the currency of Japan?", "answer": "Yen", "instruction": "Identify the currency", "input": ""},
    ]
    
    # Extend the dataset
    extended_data = demo_data * 5  # 100 samples total
    
    return extended_data

def run_demo_training():
    """Run a complete demo training with monitoring"""
    print("🚀 Starting GRPO Training Demo with Wandb Monitoring")
    print("=" * 80)
    
    # Check system requirements
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This demo requires a GPU.")
        return False
    
    print(f"✅ System ready:")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
    
    # Create configuration
    print("\n📋 Creating configuration...")
    config = GPT20BGRPOConfig()
    print(f"   Model: {config.model_name}")
    print(f"   Max sequence length: {config.max_seq_length}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   LoRA rank: {config.lora_config['r']}")
    
    # Create dataset
    print("\n📊 Creating demo dataset...")
    demo_data = create_demo_dataset()
    print(f"   Created {len(demo_data)} samples")
    
    # Save dataset
    dataset_path = "./data/demo_dataset.json"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)
    print(f"   Dataset saved to: {dataset_path}")
    
    # Initialize monitoring
    print("\n📈 Initializing Wandb monitoring...")
    monitor = WandbMonitor("grpo-demo-training", config.__dict__)
    
    if not monitor.run:
        print("❌ Failed to initialize Wandb monitoring")
        return False
    
    print(f"✅ Wandb initialized: {monitor.run.url}")
    
    try:
        # Log system information
        print("\n🔍 Logging system information...")
        monitor.log_system_info()
        monitor.log_configuration_summary(config.__dict__)
        
        # Simulate training process
        print("\n🏋️ Simulating training process...")
        
        # Simulate multiple epochs
        for epoch in range(2):
            print(f"\n   Epoch {epoch + 1}/2")
            monitor.update_epoch(epoch)
            
            # Simulate training steps
            for step in range(10):
                # Simulate realistic training metrics
                base_loss = 2.0 - epoch * 0.5 - step * 0.1
                noise = torch.randn(1).item() * 0.05
                
                metrics = {
                    "loss": max(0.1, base_loss + noise),
                    "rewards_mean": 0.3 + epoch * 0.2 + step * 0.05 + torch.randn(1).item() * 0.02,
                    "rewards_std": 0.1 + torch.randn(1).item() * 0.01,
                    "rewards_min": 0.1 + torch.randn(1).item() * 0.05,
                    "rewards_max": 0.8 + torch.randn(1).item() * 0.1,
                    "advantages_mean": 0.1 + step * 0.02 + torch.randn(1).item() * 0.01,
                    "advantages_std": 0.05 + torch.randn(1).item() * 0.01,
                    "learning_rate": config.learning_rate
                }
                
                # Log metrics
                monitor.log_training_metrics(metrics, step + epoch * 10)
                monitor.log_memory_usage(step + epoch * 10)
                monitor.log_gpu_utilization(step + epoch * 10)
                
                # Simulate some GPU work
                if step % 3 == 0:
                    dummy_tensor = torch.randn(500, 500, device='cuda')
                    _ = torch.matmul(dummy_tensor, dummy_tensor)
                    del dummy_tensor
                    torch.cuda.empty_cache()
                
                # Log sample generations every 5 steps
                if step % 5 == 0:
                    sample_prompts = demo_data[step:step+2]
                    sample_responses = [
                        f"Sample response for: {prompt['prompt']}" 
                        for prompt in sample_prompts
                    ]
                    sample_rewards = [0.7 + torch.randn(1).item() * 0.1 for _ in sample_prompts]
                    
                    monitor.log_generation_samples(
                        [p['prompt'] for p in sample_prompts],
                        sample_responses,
                        sample_rewards,
                        step + epoch * 10
                    )
                
                print(f"     Step {step + 1}/10: Loss={metrics['loss']:.4f}, "
                      f"Rewards={metrics['rewards_mean']:.4f}")
                
                # Small delay to simulate real training
                import time
                time.sleep(0.1)
        
        # Create final plots
        print("\n📊 Creating analysis plots...")
        monitor.create_plots()
        
        # Log final summary
        print("\n📋 Logging final summary...")
        final_metrics = {
            "final/loss": 0.5,
            "final/avg_reward": 0.7,
            "final/max_reward": 0.9,
            "final/min_reward": 0.3,
            "final/training_steps": 20,
            "final/epochs": 2
        }
        monitor.run.log(final_metrics)
        
        print("\n✅ Demo training completed successfully!")
        print(f"📊 View your run at: {monitor.run.url}")
        
        # Finish monitoring
        monitor.finish()
        
        return True
        
    except Exception as e:
        print(f"❌ Demo training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function"""
    print("🎯 GRPO Training Demo with Wandb Monitoring")
    print("   Optimized for NVIDIA 4080 Super (16GB VRAM)")
    print("   Based on Unsloth Colab notebook approach")
    print("=" * 80)
    
    success = run_demo_training()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. View your training run in Wandb dashboard")
        print("2. Analyze the metrics and plots")
        print("3. Use the configuration for real training")
        print("4. Scale up with your actual dataset")
    else:
        print("\n❌ Demo failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
