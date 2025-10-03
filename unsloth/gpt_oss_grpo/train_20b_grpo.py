#!/usr/bin/env python3
"""
Training script for 20B model GRPO with Unsloth on NVIDIA 4080 Super
Based on the Colab notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb
"""

import os
import sys
import json
import torch
import wandb
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grpo_unsloth_trainer import UnslothGRPOTrainer, UnslothGRPODataset
from config_20b import (
    GPT20BGRPOConfig,
    GPT20BGRPOConfigConservative,
    GPT20BGRPOConfigAggressive,
    check_memory_compatibility,
)

def create_sample_dataset(output_path: str = "./data/sample_dataset.json"):
    """Create a sample dataset for training"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample data similar to GSM8K format
    sample_data = [
        {
            "prompt": "What is 15% of 200?",
            "answer": "30",
            "instruction": "Calculate the percentage",
            "input": ""
        },
        {
            "prompt": "A store has 120 apples. They sell 3/4 of them. How many apples are left?",
            "answer": "30",
            "instruction": "Solve the word problem step by step",
            "input": ""
        },
        {
            "prompt": "If a train travels 300 miles in 4 hours, what is its average speed?",
            "answer": "75 miles per hour",
            "instruction": "Calculate the average speed",
            "input": ""
        },
        {
            "prompt": "Sarah has 24 stickers. She gives 1/3 to her friend and 1/4 to her sister. How many stickers does she have left?",
            "answer": "10",
            "instruction": "Solve the fraction word problem",
            "input": ""
        },
        {
            "prompt": "A rectangle has a length of 8 cm and width of 5 cm. What is its area?",
            "answer": "40 square centimeters",
            "instruction": "Calculate the area of the rectangle",
            "input": ""
        },
        {
            "prompt": "If 3x + 7 = 22, what is the value of x?",
            "answer": "5",
            "instruction": "Solve the linear equation",
            "input": ""
        },
        {
            "prompt": "A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, how many slices are left?",
            "answer": "2",
            "instruction": "Solve the division word problem",
            "input": ""
        },
        {
            "prompt": "What is the next number in the sequence: 2, 4, 8, 16, ?",
            "answer": "32",
            "instruction": "Find the pattern and predict the next number",
            "input": ""
        },
        {
            "prompt": "A book costs $15. If there's a 20% discount, what is the final price?",
            "answer": "$12",
            "instruction": "Calculate the discounted price",
            "input": ""
        },
        {
            "prompt": "If a triangle has sides of length 3, 4, and 5, what type of triangle is it?",
            "answer": "Right triangle",
            "instruction": "Identify the type of triangle",
            "input": ""
        }
    ]
    
    # Repeat data to create more samples
    extended_data = sample_data * 20  # 200 samples total
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample dataset created with {len(extended_data)} samples at {output_path}")
    return output_path

def main(
    preset: str = "balanced",
    model_name: str = None,
    batch_size: int = None,
    max_samples: int = 100,
    epochs: int = None,
    use_wandb: bool = False,
    output_dir: str = None,
    dataset_path_arg: str = None,
):
    """Main training function"""
    print("Starting 20B GRPO training with Unsloth...")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
        return
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create configuration from preset
    preset_map = {
        "memory": GPT20BGRPOConfigConservative,
        "balanced": GPT20BGRPOConfig,
        "performance": GPT20BGRPOConfigAggressive,
    }
    if preset not in preset_map:
        print(f"Unknown --config preset '{preset}', defaulting to 'balanced'")
        preset = "balanced"
    config = preset_map[preset]()

    # Apply CLI overrides
    if model_name:
        config.model_name = model_name
    if batch_size:
        config.batch_size = batch_size
    if epochs:
        config.num_epochs = epochs
    if output_dir:
        config.output_dir = output_dir
    if use_wandb is not None:
        config.use_wandb = use_wandb
    
    # Check memory compatibility
    print("\nChecking memory compatibility...")
    is_compatible = check_memory_compatibility(config)
    if not is_compatible:
        print("Warning: Configuration may not fit in available memory. Proceeding due to --non-interactive default.")
    
    # Create sample dataset
    if dataset_path_arg:
        dataset_path = dataset_path_arg
        print(f"\nUsing provided dataset: {dataset_path}")
    else:
        print("\nCreating sample dataset...")
        dataset_path = create_sample_dataset()
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = UnslothGRPOTrainer(config)
    
    # Create dataset
    print("Loading dataset...")
    dataset = UnslothGRPODataset(dataset_path, trainer.tokenizer, max_samples=max_samples)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Start training
    print("\nStarting training...")
    print("=" * 60)
    
    try:
        trainer.train(dataset)
        print("\nTraining completed successfully!")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nError: Out of memory - {e}")
            print("Try using a more conservative configuration or reducing batch size.")
        else:
            print(f"\nError during training: {e}")
            raise
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving current model state...")
        trainer.save_model()
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        print("Memory cleared.")

def test_configurations():
    """Test different configurations to find the best one for your setup"""
    print("Testing different configurations...")
    print("=" * 60)
    
    from config_20b import (
        GPT20BGRPOConfigConservative,
        GPT20BGRPOConfig,
        GPT20BGRPOConfigAggressive
    )
    
    configs = [
        ("Conservative", GPT20BGRPOConfigConservative()),
        ("Standard", GPT20BGRPOConfig()),
        ("Aggressive", GPT20BGRPOConfigAggressive()),
    ]
    
    for name, config in configs:
        print(f"\n=== {name} Configuration ===")
        print(f"Max sequence length: {config.max_seq_length}")
        print(f"Batch size: {config.batch_size}")
        print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"Number of generations: {config.num_generations}")
        print(f"LoRA rank: {config.lora_config['r']}")
        
        is_compatible = check_memory_compatibility(config)
        print(f"Compatible: {'Yes' if is_compatible else 'No'}")
        print("-" * 40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 20B model with GRPO using Unsloth")
    parser.add_argument("--test-configs", action="store_true", help="Test different configurations")
    # New CLI flags
    parser.add_argument("--config", type=str, default="balanced", choices=["memory", "balanced", "performance"], help="Preset configuration")
    parser.add_argument("--model-name", type=str, default=None, help="HF model id for OSS 20B (e.g., openai-oss/20b)")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to use")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--dataset", type=str, default=None, help="Path to custom dataset JSON file")
    
    args = parser.parse_args()
    
    if args.test_configs:
        test_configurations()
    else:
        main(
            preset=args.config,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            epochs=args.epochs,
            use_wandb=args.use_wandb,
            output_dir=args.output_dir,
            dataset_path_arg=args.dataset,
        )
