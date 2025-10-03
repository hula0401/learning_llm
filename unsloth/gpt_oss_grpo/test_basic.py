#!/usr/bin/env python3
"""
Basic test script for GRPO Unsloth implementation
Tests basic functionality without full training
"""

import os
import sys
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from grpo_unsloth_trainer import UnslothGRPOConfig, UnslothGRPODataset
        print("✅ GRPO trainer imports successful")
    except ImportError as e:
        print(f"❌ GRPO trainer import failed: {e}")
        return False
    
    try:
        from config_20b import GPT20BGRPOConfig, check_memory_compatibility
        print("✅ Configuration imports successful")
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration creation and memory compatibility"""
    print("\nTesting configuration...")
    
    try:
        from config_20b import GPT20BGRPOConfig
        
        config = GPT20BGRPOConfig()
        print(f"✅ Configuration created successfully")
        print(f"   Model: {config.model_name}")
        print(f"   Max sequence length: {config.max_seq_length}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   LoRA rank: {config.lora_config['r']}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_memory_compatibility():
    """Test memory compatibility check"""
    print("\nTesting memory compatibility...")
    
    try:
        from config_20b import GPT20BGRPOConfig, check_memory_compatibility
        
        config = GPT20BGRPOConfig()
        is_compatible = check_memory_compatibility(config)
        
        print(f"✅ Memory compatibility check: {'PASS' if is_compatible else 'FAIL'}")
        return is_compatible
    except Exception as e:
        print(f"❌ Memory compatibility test failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation"""
    print("\nTesting dataset creation...")
    
    try:
        from grpo_unsloth_trainer import UnslothGRPODataset
        import json
        import tempfile
        
        # Create temporary test data
        test_data = [
            {"prompt": "What is 2+2?", "answer": "4", "instruction": "Solve math", "input": ""},
            {"prompt": "What is 3+3?", "answer": "6", "instruction": "Solve math", "input": ""},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        # Test dataset creation (without tokenizer for now)
        try:
            dataset = UnslothGRPODataset(temp_file, None)
            print(f"✅ Dataset created with {len(dataset)} samples")
            success = True
        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            success = False
        finally:
            os.unlink(temp_file)
        
        return success
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ CUDA not available")
        return False

def main():
    """Run all tests"""
    print("Running basic tests for GRPO Unsloth implementation...")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("CUDA Test", test_cuda_availability),
        ("Memory Compatibility Test", test_memory_compatibility),
        ("Dataset Test", test_dataset_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
