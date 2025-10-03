#!/usr/bin/env python3
"""
Configuration for 20B model GRPO training on NVIDIA 4080 Super (16GB VRAM)
Optimized based on the Colab notebook approach
"""

from unsloth.gpt_oss_grpo.grpo_unsloth_trainer import UnslothGRPOConfig

class GPT20BGRPOConfig(UnslothGRPOConfig):
    """Configuration specifically optimized for 20B parameter models on 16GB VRAM"""
    
    def __init__(self):
        super().__init__()
        
        # Model configuration for 20B models
        self.model_name = "unsloth/gpt-2"  # Change to your 20B model
        self.max_seq_length = 1024  # Reduced for memory efficiency
        self.dtype = "bfloat16" if True else "float16"  # Use bfloat16 if available
        self.load_in_4bit = True  # Essential for 20B models on 16GB
        self.load_in_8bit = False
        
        # Memory optimizations
        self.use_gradient_checkpointing = True
        self.use_flash_attention = True
        self.use_vllm = True  # Use vLLM for generation
        
        # GRPO parameters optimized for 20B
        self.num_generations = 2  # Reduced for memory efficiency
        self.max_prompt_length = 256  # Shorter prompts
        self.max_generate_length = 512  # Shorter responses
        self.batch_size = 1  # Single batch
        self.gradient_accumulation_steps = 16  # Increased for effective larger batch size
        
        # Training parameters
        self.learning_rate = 5e-6  # Lower learning rate for stability
        self.num_epochs = 2  # Fewer epochs
        self.save_steps = 50
        self.logging_steps = 5
        
        # GRPO algorithm parameters
        self.beta = 0.1  # KL divergence coefficient
        self.clip_eps = 0.2
        self.reward_weights = [1.0, 0.5, 0.3]  # Multiple reward functions
        
        # vLLM configuration for 20B models
        self.vllm_config = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,  # Use 90% of VRAM
            "max_model_len": 1024,
            "dtype": "bfloat16",
            "quantization": "awq",  # Use AWQ quantization
            "trust_remote_code": True,
            "enforce_eager": True,  # Disable CUDA graph for memory efficiency
        }
        
        # LoRA configuration for 20B models
        self.lora_config = {
            "r": 8,  # Lower rank for memory efficiency
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
            "use_rslora": False,
        }


# Alternative configurations for different scenarios
class GPT20BGRPOConfigConservative(UnslothGRPOConfig):
    """More conservative configuration for maximum memory efficiency"""
    
    def __init__(self):
        super().__init__()
        
        self.model_name = "unsloth/gpt-2"
        self.max_seq_length = 512  # Very short sequences
        self.dtype = "float16"
        self.load_in_4bit = True
        self.load_in_8bit = False
        
        self.num_generations = 1  # Single generation per prompt
        self.max_prompt_length = 128
        self.max_generate_length = 256
        self.batch_size = 1
        self.gradient_accumulation_steps = 32
        
        self.learning_rate = 1e-6
        self.num_epochs = 1
        
        self.lora_config = {
            "r": 4,  # Very low rank
            "lora_alpha": 8,
            "lora_dropout": 0.2,
            "target_modules": ["q_proj", "v_proj"],  # Fewer target modules
            "bias": "none",
        }


class GPT20BGRPOConfigAggressive(UnslothGRPOConfig):
    """More aggressive configuration for better performance (risky on 16GB)"""
    
    def __init__(self):
        super().__init__()
        
        self.model_name = "unsloth/gpt-2"
        self.max_seq_length = 2048
        self.dtype = "bfloat16"
        self.load_in_4bit = True
        self.load_in_8bit = False
        
        self.num_generations = 4
        self.max_prompt_length = 512
        self.max_generate_length = 1024
        self.batch_size = 1
        self.gradient_accumulation_steps = 8
        
        self.learning_rate = 1e-5
        self.num_epochs = 3
        
        self.lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
        }


# Memory monitoring utilities
def get_memory_usage():
    """Get current GPU memory usage"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
        }
    return {"error": "CUDA not available"}


def check_memory_compatibility(config: UnslothGRPOConfig) -> bool:
    """Check if configuration is compatible with available memory"""
    memory_info = get_memory_usage()
    
    if "error" in memory_info:
        print("CUDA not available")
        return False
    
    free_memory = memory_info["free_gb"]
    
    # Rough estimates for 20B model
    model_memory = 20 * 0.5  # 20B parameters * 0.5 bytes (4-bit quantization)
    sequence_memory = config.max_seq_length * config.batch_size * 0.001  # Rough estimate
    generation_memory = config.num_generations * 0.1  # Additional memory for generation
    
    total_estimated = model_memory + sequence_memory + generation_memory
    
    print(f"Free GPU memory: {free_memory:.2f} GB")
    print(f"Estimated memory usage: {total_estimated:.2f} GB")
    
    if total_estimated > free_memory * 0.9:  # Use 90% of available memory
        print("Warning: Configuration may exceed available memory")
        return False
    
    print("Configuration appears compatible with available memory")
    return True


if __name__ == "__main__":
    import torch
    
    # Test different configurations
    configs = [
        ("Conservative", GPT20BGRPOConfigConservative()),
        ("Standard", GPT20BGRPOConfig()),
        ("Aggressive", GPT20BGRPOConfigAggressive()),
    ]
    
    for name, config in configs:
        print(f"\n=== {name} Configuration ===")
        print(f"Model: {config.model_name}")
        print(f"Max sequence length: {config.max_seq_length}")
        print(f"Batch size: {config.batch_size}")
        print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"Number of generations: {config.num_generations}")
        print(f"LoRA rank: {config.lora_config['r']}")
        
        check_memory_compatibility(config)
