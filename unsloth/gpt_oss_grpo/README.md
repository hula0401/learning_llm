# GRPO Training with Unsloth for 20B Models on 16GB VRAM

This implementation provides an optimized GRPO (Group Relative Policy Optimization) training setup using Unsloth, specifically designed for training 20B parameter models on NVIDIA 4080 Super with 16GB VRAM.

Based on the [Unsloth Colab notebook](http://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) that uses 14GB VRAM for training OpenAI OSS 20B with GRPO.

## Features

- **Memory Efficient**: Optimized for 16GB VRAM using 4-bit quantization and LoRA
- **vLLM Integration**: Fast inference using vLLM for response generation
- **Multiple Configurations**: Conservative, Standard, and Aggressive settings
- **GRPO Algorithm**: Group-based relative policy optimization
- **Unsloth Optimizations**: Flash attention, gradient checkpointing, and memory optimizations

## Quick Start

### 1. Install Dependencies

```bash
# Install Unsloth and dependencies
pip install -r requirements.txt

# Or install individually
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install vllm wandb tqdm
```

### 2. Test Configuration

```bash
# Test different configurations to find the best one for your setup
python train_20b_grpo.py --test-configs
```

### 3. Start Training

```bash
# Train with default configuration
python train_20b_grpo.py

# Train with custom dataset
python train_20b_grpo.py --dataset /path/to/your/dataset.json
```

## Configuration Options

### Conservative (Maximum Memory Efficiency)
- Max sequence length: 512
- Batch size: 1
- Gradient accumulation: 32
- LoRA rank: 4
- Single generation per prompt

### Standard (Balanced)
- Max sequence length: 1024
- Batch size: 1
- Gradient accumulation: 16
- LoRA rank: 8
- 2 generations per prompt

### Aggressive (Better Performance, Risky)
- Max sequence length: 2048
- Batch size: 1
- Gradient accumulation: 8
- LoRA rank: 16
- 4 generations per prompt

## Memory Usage

The implementation is designed to use approximately 14-15GB of VRAM for a 20B model:

- **Model (4-bit quantized)**: ~10GB
- **LoRA parameters**: ~1GB
- **Activations and gradients**: ~3-4GB
- **vLLM inference**: ~1-2GB

## Dataset Format

The training script expects JSON data in the following format:

```json
[
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
  }
]
```

## Key Optimizations

### 1. Memory Optimizations
- 4-bit quantization for model weights
- LoRA fine-tuning instead of full fine-tuning
- Gradient checkpointing
- Flash attention
- Efficient data loading

### 2. vLLM Integration
- Fast inference for response generation
- Memory-efficient generation
- Batch processing support

### 3. GRPO Algorithm
- Group-based advantage computation
- PPO clipping for stable training
- Multiple reward functions
- KL divergence regularization

## Monitoring

The training process is monitored using:
- **Wandb**: Real-time metrics and logging
- **Memory monitoring**: GPU memory usage tracking
- **Progress bars**: Training progress visualization

## Troubleshooting

### Out of Memory Errors
1. Try the conservative configuration
2. Reduce `max_seq_length`
3. Reduce `num_generations`
4. Increase `gradient_accumulation_steps`

### Slow Training
1. Enable vLLM for faster generation
2. Use flash attention
3. Increase batch size if memory allows

### Installation Issues
1. Make sure you have CUDA 11.8+ installed
2. Install PyTorch with CUDA support
3. Install vLLM with compatible CUDA version

## Example Usage

```python
from grpo_unsloth_trainer import UnslothGRPOTrainer
from config_20b import GPT20BGRPOConfig
from grpo_unsloth_trainer import UnslothGRPODataset

# Create configuration
config = GPT20BGRPOConfig()

# Initialize trainer
trainer = UnslothGRPOTrainer(config)

# Load dataset
dataset = UnslothGRPODataset("data.json", trainer.tokenizer)

# Start training
trainer.train(dataset)
```

## Performance Expectations

On NVIDIA 4080 Super (16GB VRAM):
- **Training speed**: ~0.5-1.0 steps/second
- **Memory usage**: 14-15GB VRAM
- **Convergence**: 2-3 epochs for basic tasks
- **Model size**: ~10GB (4-bit quantized)

## References

- [Unsloth Colab Notebook](http://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [vLLM Documentation](https://docs.vllm.ai/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
