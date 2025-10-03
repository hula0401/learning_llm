#!/usr/bin/env python3
"""
Test Wandb monitoring on NVIDIA 4080 Super GPU
Comprehensive test of all monitoring features
"""

import os
import sys
import torch
import time
import json
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wandb_monitor import WandbMonitor, test_wandb_monitoring
from grpo_unsloth_trainer import UnslothGRPOTrainer, UnslothGRPODataset
from config_20b import GPT20BGRPOConfig

def test_gpu_monitoring():
    """Test GPU-specific monitoring features"""
    print("Testing GPU monitoring on NVIDIA 4080 Super...")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a GPU.")
        return False
    
    print(f"✅ CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    
    # Test configuration
    config = {
        "model_name": "unsloth/gpt-2",
        "max_seq_length": 1024,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "num_generations": 4,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3
    }
    
    # Initialize monitor
    print("\nInitializing Wandb monitor...")
    monitor = WandbMonitor("grpo-gpu-test", config)
    
    if not monitor.run:
        print("❌ Wandb not available. Please check your setup.")
        return False
    
    print(f"✅ Wandb initialized: {monitor.run.url}")
    
    try:
        # Test 1: System information logging
        print("\n1. Testing system information logging...")
        monitor.log_system_info()
        print("   ✅ System info logged")
        
        # Test 2: Memory usage monitoring
        print("\n2. Testing memory usage monitoring...")
        for i in range(5):
            # Simulate memory usage
            if i == 2:
                # Allocate some GPU memory
                dummy_tensor = torch.randn(1000, 1000, device='cuda')
            
            monitor.log_memory_usage(i)
            time.sleep(0.5)
        
        # Clean up
        if 'dummy_tensor' in locals():
            del dummy_tensor
        torch.cuda.empty_cache()
        print("   ✅ Memory usage logged")
        
        # Test 3: GPU utilization monitoring
        print("\n3. Testing GPU utilization monitoring...")
        for i in range(5):
            # Simulate some GPU work
            if i % 2 == 0:
                dummy_tensor = torch.randn(500, 500, device='cuda')
                _ = torch.matmul(dummy_tensor, dummy_tensor)
                del dummy_tensor
            
            monitor.log_gpu_utilization(i)
            time.sleep(0.5)
        
        torch.cuda.empty_cache()
        print("   ✅ GPU utilization logged")
        
        # Test 4: Training metrics simulation
        print("\n4. Testing training metrics simulation...")
        for step in range(10):
            # Simulate training metrics
            metrics = {
                "loss": 2.0 - step * 0.15 + torch.randn(1).item() * 0.1,
                "rewards_mean": 0.3 + step * 0.05 + torch.randn(1).item() * 0.02,
                "rewards_std": 0.1 + torch.randn(1).item() * 0.01,
                "rewards_min": 0.1 + torch.randn(1).item() * 0.05,
                "rewards_max": 0.8 + torch.randn(1).item() * 0.1,
                "advantages_mean": 0.1 + step * 0.02 + torch.randn(1).item() * 0.01,
                "advantages_std": 0.05 + torch.randn(1).item() * 0.01,
                "learning_rate": 1e-5
            }
            
            monitor.log_training_metrics(metrics, step)
            monitor.update_step(step)
            
            # Simulate some GPU work
            if step % 3 == 0:
                dummy_tensor = torch.randn(200, 200, device='cuda')
                _ = torch.matmul(dummy_tensor, dummy_tensor)
                del dummy_tensor
                torch.cuda.empty_cache()
            
            time.sleep(0.2)
        
        print("   ✅ Training metrics logged")
        
        # Test 5: Generation samples logging
        print("\n5. Testing generation samples logging...")
        prompts = [
            "What is 15% of 200?",
            "A store has 120 apples. They sell 3/4 of them. How many apples are left?",
            "If a train travels 300 miles in 4 hours, what is its average speed?",
            "Sarah has 24 stickers. She gives 1/3 to her friend and 1/4 to her sister. How many stickers does she have left?",
            "A rectangle has a length of 8 cm and width of 5 cm. What is its area?"
        ]
        
        responses = [
            "To find 15% of 200, I multiply 200 by 0.15: 200 × 0.15 = 30. So 15% of 200 is 30.",
            "The store sells 3/4 of 120 apples. 3/4 × 120 = 90 apples sold. 120 - 90 = 30 apples left.",
            "Average speed = distance ÷ time = 300 miles ÷ 4 hours = 75 miles per hour.",
            "Sarah gives away 1/3 + 1/4 = 4/12 + 3/12 = 7/12 of her stickers. 7/12 × 24 = 14 stickers given away. 24 - 14 = 10 stickers left.",
            "Area = length × width = 8 cm × 5 cm = 40 square centimeters."
        ]
        
        rewards = [0.8, 0.9, 0.7, 0.85, 0.75]
        
        monitor.log_generation_samples(prompts, responses, rewards, 10)
        print("   ✅ Generation samples logged")
        
        # Test 6: Configuration summary
        print("\n6. Testing configuration summary...")
        monitor.log_configuration_summary(config)
        print("   ✅ Configuration summary logged")
        
        # Test 7: Create plots
        print("\n7. Testing plot creation...")
        monitor.create_plots()
        print("   ✅ Plots created and logged")
        
        # Test 8: Model artifacts (simulate)
        print("\n8. Testing model artifacts logging...")
        # Create a temporary directory with dummy model files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy model files
            dummy_config = {"model_type": "gpt2", "vocab_size": 50257}
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(dummy_config, f)
            
            # Create a dummy model file
            with open(os.path.join(temp_dir, "pytorch_model.bin"), "w") as f:
                f.write("dummy model content")
            
            monitor.log_model_artifacts(temp_dir, 10)
            print("   ✅ Model artifacts logged")
        
        print("\n" + "=" * 60)
        print("🎉 All GPU monitoring tests completed successfully!")
        print(f"View your run at: {monitor.run.url}")
        
        # Finish the test run
        monitor.finish()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_simulation():
    """Test a simulated training run with monitoring"""
    print("\nTesting simulated training run...")
    print("=" * 60)
    
    try:
        # Create configuration
        config = GPT20BGRPOConfig()
        
        # Create dummy dataset
        dummy_data = [
            {"prompt": "What is 2+2?", "answer": "4", "instruction": "Solve math", "input": ""},
            {"prompt": "What is 3+3?", "answer": "6", "instruction": "Solve math", "input": ""},
            {"prompt": "What is 4+4?", "answer": "8", "instruction": "Solve math", "input": ""},
        ] * 10  # 30 samples total
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dummy_data, f)
            temp_file = f.name
        
        try:
            # Create dataset (without tokenizer for simulation)
            dataset = UnslothGRPODataset(temp_file, None, max_samples=30)
            
            # Initialize trainer (this will fail without Unsloth, but we can test the monitoring)
            print("Note: This test simulates training without actual model loading...")
            
            # Test the monitoring components directly
            from wandb_monitor import WandbMonitor
            
            monitor = WandbMonitor("grpo-training-simulation", config.__dict__)
            
            if monitor.run:
                print(f"✅ Training simulation monitor initialized: {monitor.run.url}")
                
                # Simulate training steps
                for epoch in range(2):
                    for step in range(5):
                        # Simulate metrics
                        metrics = {
                            "loss": 1.5 - epoch * 0.3 - step * 0.1 + torch.randn(1).item() * 0.05,
                            "rewards_mean": 0.4 + epoch * 0.1 + step * 0.02 + torch.randn(1).item() * 0.01,
                            "rewards_std": 0.1 + torch.randn(1).item() * 0.01,
                            "advantages_mean": 0.2 + step * 0.01 + torch.randn(1).item() * 0.005,
                            "learning_rate": config.learning_rate
                        }
                        
                        monitor.log_training_metrics(metrics, step + epoch * 5)
                        monitor.log_memory_usage(step + epoch * 5)
                        monitor.log_gpu_utilization(step + epoch * 5)
                        
                        time.sleep(0.1)
                
                # Create final plots
                monitor.create_plots()
                
                print("✅ Training simulation completed successfully!")
                monitor.finish()
                
                return True
            else:
                print("❌ Failed to initialize monitoring for training simulation")
                return False
                
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"❌ Training simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Running comprehensive Wandb monitoring tests on NVIDIA 4080 Super...")
    print("=" * 80)
    
    tests = [
        ("GPU Monitoring Test", test_gpu_monitoring),
        ("Training Simulation Test", test_training_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Wandb monitoring tests passed! Ready for production use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
