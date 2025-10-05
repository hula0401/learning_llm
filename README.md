# LLM Learning Project

A comprehensive exploration of Large Language Model training and reinforcement learning techniques, with focus on GRPO (Group Relative Policy Optimization) for mathematical reasoning tasks.

## 📁 Project Structure

```
learning_llm/
├── grpo/                           # Original GRPO implementation
│   ├── trainer.py                  # Legacy GRPO trainer
│   └── reward.py                   # Reward function utilities
│
├── training_with_unsloth/          # Production GRPO training framework
│   ├── framework/                  # Core training infrastructure
│   │   ├── grpo_trainer.py        # Refactored GRPO trainer with detailed eval
│   │   ├── config.py              # Experiment configuration management
│   │   └── dataset.py             # GSM8K dataset utilities
│   │
│   ├── rewards/                    # Reward function library
│   │   └── reward_functions.py    # Correctness, format, and mark rewards
│   │
│   ├── experiments/                # Training scripts
│   │   ├── run_experiment.py      # Main experiment runner
│   │   └── run_eval_test.py       # Evaluation-focused training
│   │
│   └── experiment_configs/         # YAML configurations
│       ├── eval_test_improved.yaml # Recommended config (30 train/20 test)
│       └── qwen3_grpo_*.yaml      # Production configs
│
└── .cursor/                        # Documentation
    └── rules                       # WandB logging standards
```

## 🎯 Key Features

### 1. **GRPO Training Pipeline**
- Group Relative Policy Optimization for LLM fine-tuning
- Multiple reward functions (correctness, format, digit validation)
- KL divergence regularization with reference model
- Gradient accumulation and checkpointing for memory efficiency

### 2. **Efficient 4B Model Training on 16GB**
- **QLoRA (4-bit quantization)**: 2GB base model (vs 8GB FP16)
- **LoRA adapters**: Train 100M params instead of 4B
- **Gradient checkpointing**: 5x memory reduction on activations
- **Unsloth optimizations**: 2x faster training with custom kernels

### 3. **Comprehensive Evaluation**
- **Pass@K metrics**: Pass@1 (accuracy), Pass@5, Pass@10
- **Detailed logging**: Every answer for every eval question
- **Baseline comparison**: Pre-training vs post-training analysis
- **WandB integration**: Real-time metrics and visualization

### 4. **GSM8K Mathematical Reasoning**
- Grade school math word problems
- Configurable train/test splits
- Robust answer extraction and normalization
- Format enforcement with partial credit

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install torch transformers datasets wandb unsloth bitsandbytes

# Or use uv (recommended)
uv sync
```

### Run Training
```bash
cd /path/to/learning_llm
UV_PYTHON=python3 uv run python3 \
  training_with_unsloth/experiments/run_eval_test.py \
  training_with_unsloth/experiment_configs/eval_test_improved.yaml
```

### Configuration
Edit `experiment_configs/eval_test_improved.yaml`:
```yaml
model_name: Qwen/Qwen3-4B-Instruct-2507
train_samples: 30
test_samples: 20
learning_rate: 5e-5
num_epochs: 100
temperature: 0.8
eval_temperature: 0.5
beta: 0.05  # KL regularization
```

## 📊 Key Metrics

### Pass@K Definition
- **Pass@1**: Greedy decoding accuracy (temperature=0)
- **Pass@5**: Success if ≥1 of 5 samples correct (temperature=0.5)
- **Pass@10**: Success if ≥1 of 10 samples correct (temperature=0.5)

### Expected Performance
```
Baseline (pre-trained Qwen3-4B):
  Pass@1:  0.75-0.85
  Pass@5:  0.85-0.90
  Pass@10: 0.90-0.95

After GRPO training (30 samples, 10 epochs):
  Pass@1:  0.85-0.92
  Pass@5:  0.90-0.95
  Pass@10: 0.92-0.97
```

## 🐛 Critical Bug Fixes

### normalize_number() Fix
**Issue**: Model answers marked wrong due to formatting
```python
# Before: '$108' ≠ '108' → Marked WRONG
# After:  '$108' → '108' → Marked CORRECT ✓
```

**Impact**: +40% improvement on Pass@1 baseline

See `CRITICAL_BUG_FIX.md` for details.

## 💡 Technical Highlights

### Memory Optimization
```
Full Fine-Tuning:  50GB VRAM ❌
LoRA:             15GB VRAM ⚠️
QLoRA (4-bit):     8GB VRAM ✓ (fits in 16GB!)
```

### Reward Functions
1. **correctness_reward** (2.0): Numerical answer matches ground truth
2. **digit_reward** (0.5): Answer contains only digits
3. **hard_format_reward** (0.25-0.5): Uses `<answer>` tags (partial credit)
4. **mark_reward** (0.5): Complete `<think>...</think><answer>...</answer>` format

### GRPO Loss Components
```python
# Advantage-based policy gradient with clipping
ratio = exp(log_prob_new - log_prob_old)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio × advantage, clipped_ratio × advantage)
```

## 📈 WandB Logging

All experiments log to WandB with:
- Training: `grpo_loss`, `rewards/{mean,std,max,min}`, `timestamp`
- Evaluation: `eval/pass@{1,5,10}`, `epoch/accumulated_reward`
- Samples: `sample/{prompt,response,answer}`

View at: https://wandb.ai/[your-entity]/qwen3_4b_grpo

## 📚 Documentation

- `GRPO_TRAINING_EXPLANATION.md`: Deep dive into GRPO algorithm
- `CRITICAL_BUG_FIX.md`: normalize_number() bug analysis
- `DETAILED_EVAL_LOGGING.md`: Evaluation system guide
- `IMPROVEMENTS_SUMMARY.md`: All optimizations applied
- `.cursor/rules`: WandB logging standards

## 🔬 Experiments

### Recommended Starting Config
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Dataset**: GSM8K (30 train, 20 test)
- **Learning Rate**: 5e-5
- **Batch Size**: 1 (effective 4 with gradient accumulation)
- **KL Beta**: 0.05 (prevents model drift)
- **Temperature**: 0.8 (training), 0.5 (eval)

### Hyperparameter Tuning Tips
- ↑ LR if Pass@1 not improving (try 1e-4)
- ↑ Temperature if Pass@5 = Pass@1 (try 0.9)
- ↑ Beta if training unstable (try 0.1)
- ↑ Samples if dataset too small (try 100+)

## 🛠️ Development

### Running Tests
```bash
# Test reward functions
python3 -c "from training_with_unsloth.rewards.reward_functions import *; ..."

# Test normalization
python3 -c "from training_with_unsloth.rewards.reward_functions import normalize_number; print(normalize_number('\$108'))"
```

### Adding New Reward Functions
```python
# In rewards/reward_functions.py
def my_reward(prompts, responses, answers):
    """Custom reward function."""
    rewards = []
    for response in responses:
        # Your logic here
        reward = compute_reward(response)
        rewards.append(reward)
    return rewards

# In run_experiment.py
reward_funcs = [correctness_reward, my_reward, ...]
```

## 🚨 Common Issues

### OOM (Out of Memory)
```yaml
# Reduce memory usage:
batch_size: 1
gradient_accumulation_steps: 8  # Increase this
num_generations: 2  # Reduce this
load_in_4bit: true  # Must be true
use_gradient_checkpointing: true  # Must be true
```

### Pass@1 Not Improving
1. Check answer normalization (dollar signs, units?)
2. Increase learning rate (5e-5 → 1e-4)
3. More training samples (30 → 100+)
4. Check reward signals are non-zero

### Pass@5 = Pass@1
- Increase `eval_temperature` (0.5 → 0.8)
- Check if samples are actually different
- May indicate model is very confident

## 📖 References

- **GRPO Paper**: [Group Relative Policy Optimization]
- **QLoRA**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **GSM8K**: [Grade School Math 8K Problems](https://github.com/openai/grade-school-math)
- **Unsloth**: [2x Faster LLM Training](https://github.com/unslothai/unsloth)

## 🤝 Contributing

This is a learning project. Key areas for improvement:
- [ ] Add more reward functions (step-by-step reasoning)
- [ ] Experiment with other models (Llama3, Mistral)
- [ ] Try other datasets (MATH, AQuA)
- [ ] Implement PPO comparison
- [ ] Add model merging after training

## 📝 License

Educational project for learning purposes.

## 🙏 Acknowledgments

- OpenAI for GSM8K dataset
- Unsloth team for optimization framework
- Hugging Face for transformers library
- WandB for experiment tracking

