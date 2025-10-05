# Training with Unsloth - Production GRPO Framework

Efficient GRPO (Group Relative Policy Optimization) training framework for fine-tuning LLMs on mathematical reasoning tasks, optimized for 16GB VRAM using Unsloth and QLoRA.

## 🎯 Overview

This framework enables training 4B parameter models on consumer GPUs through:
- **QLoRA**: 4-bit quantization + LoRA adapters
- **Unsloth**: 2x faster training with custom CUDA kernels
- **GRPO**: Group-based reinforcement learning with advantage normalization
- **Comprehensive Evaluation**: Detailed Pass@K metrics with full answer logging

## 📁 Structure

```
training_with_unsloth/
├── framework/
│   ├── grpo_trainer.py         # Core GRPO trainer with detailed evaluation
│   ├── config.py               # ExperimentConfig dataclass + YAML I/O
│   └── dataset.py              # GSM8K dataset utilities
│
├── rewards/
│   └── reward_functions.py     # Reward functions library
│
├── experiments/
│   ├── run_experiment.py       # Production training script
│   └── run_eval_test.py        # Evaluation-focused training
│
├── experiment_configs/
│   ├── eval_test_improved.yaml # Recommended: 30 train/20 test
│   ├── qwen3_grpo_bs1_lr1e-5_ng4_ga4_seed42_rmk.yaml
│   └── smoke.yaml              # Quick sanity check
│
└── output/                     # Training outputs
    └── eval_test_v2/
        ├── checkpoint_*/       # Model checkpoints
        └── experiment_config.yaml
```

## 🚀 Quick Start

### 1. Run Training
```bash
cd /home/hula0401/learning_llm

# Recommended config (30 train, 20 test, 100 epochs)
UV_PYTHON=python3 uv run python3 \
  training_with_unsloth/experiments/run_eval_test.py \
  training_with_unsloth/experiment_configs/eval_test_improved.yaml
```

### 2. Monitor Progress
```bash
# Follow WandB link in output:
# https://wandb.ai/[user]/qwen3_4b_grpo/runs/[run_id]

# Or watch logs
tail -f wandb/latest-run/logs/debug.log
```

### 3. Check Results
Training outputs detailed evaluation every epoch:
```
Question 1/20: "Stefan goes to restaurant..."
Ground Truth: 108
Pass@1 (Greedy): '$108' → '108' ✓ CORRECT
Pass@5: ✓ (4/5 correct)
Pass@10: ✓ (9/10 correct)
```

## ⚙️ Configuration

### Recommended Settings
```yaml
# eval_test_improved.yaml
model_name: Qwen/Qwen3-4B-Instruct-2507
train_samples: 30               # Scale based on GPU time budget
test_samples: 20                # Keep 10-20 for fast eval
learning_rate: 5e-5             # Higher than standard (1e-5)
num_epochs: 100
batch_size: 1
gradient_accumulation_steps: 4  # Effective batch size = 4
num_generations: 4              # Responses per prompt
temperature: 0.8                # Training diversity
eval_temperature: 0.5           # Eval sampling temperature
beta: 0.05                      # KL regularization (prevents drift)
clip_eps: 0.2                   # PPO-style clipping
```

### Memory vs Performance Tradeoffs
```yaml
# Low Memory (8-10GB)
batch_size: 1
gradient_accumulation_steps: 8
num_generations: 2
use_gradient_checkpointing: true
load_in_4bit: true

# High Memory (14-16GB)
batch_size: 1
gradient_accumulation_steps: 4
num_generations: 4
use_gradient_checkpointing: true
load_in_4bit: true

# Maximum Performance (24GB+)
batch_size: 2
gradient_accumulation_steps: 2
num_generations: 8
use_gradient_checkpointing: false
load_in_4bit: false  # Use BF16
```

## 📊 Metrics & Evaluation

### Pass@K Metrics
**Pass@1 (Accuracy)**: Greedy decoding (temp=0.0, deterministic)
```python
correct_answers / total_questions
```

**Pass@5**: Success if ≥1 of 5 samples correct (temp=0.5, sampling)
```python
(prompts_with_≥1_correct_in_5) / total_prompts
```

**Pass@10**: Success if ≥1 of 10 samples correct (temp=0.5, sampling)
```python
(prompts_with_≥1_correct_in_10) / total_prompts
```

### Reward Functions
```python
# Total reward = sum of weighted rewards
correctness_reward:     2.0 if answer matches, else 0.0
digit_reward:           0.5 if answer is digits only
hard_format_reward:     0.25 for <answer>, 0.5 for <think>+<answer>
mark_reward:            0.125 per tag (4 tags max = 0.5)
```

### WandB Logging
```python
# Training (every step)
grpo_loss, rewards/{mean,std,max,min}, step, timestamp

# Evaluation (every epoch)
eval/pass@{1,5,10}, epoch/accumulated_reward, epoch/mean_reward

# Samples (periodic)
sample/{prompt,response,answer}
```

## 🔧 Technical Details

### GRPO Algorithm
```python
# 1. Generate N responses per prompt
responses = model.generate(prompt, num_return_sequences=N)

# 2. Compute rewards
rewards = reward_func(prompt, responses, ground_truth)

# 3. Normalize advantages within group
advantage = (reward - mean(rewards)) / (std(rewards) + ε)

# 4. Compute policy ratio
ratio = exp(log_prob_new - log_prob_old)

# 5. Clipped surrogate objective (PPO-style)
loss = -min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)

# 6. Optional KL penalty
loss += β × KL(new_policy || ref_policy)
```

### Memory Optimization Stack
```
Layer 1: Model Quantization (4-bit)
  - Base model: 8GB → 2GB (4x reduction)
  - Quantization: NF4 with double quantization

Layer 2: LoRA Parameter-Efficient Training
  - Trainable params: 4B → 100M (40x reduction)
  - Gradients: 8GB → 0.2GB
  - Optimizer states: 32GB → 0.8GB

Layer 3: Gradient Checkpointing
  - Activations: 10-20GB → 2-4GB (5x reduction)
  - Tradeoff: 20-30% slower training

Layer 4: Gradient Accumulation
  - Batch size in VRAM: 1 (regardless of effective batch)
  - No memory overhead from larger batches

Layer 5: Unsloth Optimizations
  - Custom CUDA kernels
  - Optimized attention (Flash Attention)
  - Better memory layout

Total: 50GB → 8-10GB ✓
```

### Answer Normalization
Critical for correct evaluation:
```python
def normalize_number(text: str) -> str:
    """Remove $, commas, units to extract pure number."""
    # '$108' → '108'
    # '$360,000' → '360000'
    # '1350 minutes' → '1350'
    # '$3,450' → '3450'
```

**Bug Fix (2024)**: Previously failed to remove $ and units, causing 40% false negatives!

## 🐛 Troubleshooting

### Pass@1 Not Improving
**Symptoms**: Pass@1 stays flat (e.g., 0.5 → 0.5 → 0.5)

**Diagnosis**:
```bash
# Check if greedy answers are changing
grep "Pass@1 (Greedy" training_log.txt | head -20

# Look for pattern:
# Epoch 0: '14' ✗
# Epoch 1: '14' ✗  ← Same wrong answer = stuck
# Epoch 5: '10' ✗  ← Different = exploring
# Epoch 10: '7' ✓  ← Correct = learning!
```

**Solutions**:
1. Increase learning rate: `1e-5 → 5e-5 → 1e-4`
2. More training data: `30 → 100 samples`
3. Check reward signals: `grep "Reward 'correctness" | head`
4. Verify normalization: Check for $ or unit mismatches

### Pass@5 = Pass@1 (No Diversity)
**Symptoms**: Pass@5 equals Pass@1 (samples all identical)

**Solutions**:
1. Increase `eval_temperature`: `0.5 → 0.8`
2. Check sampling: Should see different answers
3. Verify `do_sample=True` in generation

### OOM Error
**Solutions**:
```yaml
# Reduce memory:
gradient_accumulation_steps: 8  # ↑ Increase
num_generations: 2              # ↓ Decrease  
max_generate_length: 128        # ↓ Decrease
batch_size: 1                   # Must be 1

# Verify:
load_in_4bit: true             # Must be true
use_gradient_checkpointing: true  # Must be true
```

### Rewards Always 0
**Symptom**: `hard_format_reward` and `mark_reward` always 0

**Cause**: Model not using `<answer>` tags

**Check**:
```bash
grep "Full Response" training_log.txt | head -5
# Look for: <answer>123</answer>
# vs wrong: "The answer is 123"
```

**Solution**: Reward functions now give partial credit (0.25 for `<answer>` only)

## 📈 Expected Performance

### Baseline (Qwen3-4B, No Training)
```
Pass@1:  0.75-0.85
Pass@5:  0.85-0.90
Pass@10: 0.90-0.95
```

### After GRPO (30 samples, 20 epochs, LR=5e-5)
```
Pass@1:  0.80-0.90 (+5-10%)
Pass@5:  0.88-0.93 (+3-5%)
Pass@10: 0.92-0.96 (+2-3%)

Accumulated Reward: ↑ 20-30% over baseline
```

### After GRPO (100 samples, 50 epochs, LR=5e-5)
```
Pass@1:  0.85-0.92 (+10-15%)
Pass@5:  0.90-0.95 (+5-8%)
Pass@10: 0.93-0.97 (+3-5%)
```

## 🔬 Advanced Usage

### Custom Reward Functions
```python
# In rewards/reward_functions.py
def step_by_step_reward(prompts, responses, answers):
    """Reward for showing reasoning steps."""
    rewards = []
    for response in responses:
        steps = response.count('\n')  # Count reasoning steps
        reward = min(steps * 0.1, 1.0)  # Max 1.0
        rewards.append(reward)
    return rewards

# In run_experiment.py
reward_funcs = [
    correctness_reward,      # 2.0
    digit_reward,           # 0.5
    step_by_step_reward,    # 1.0 (custom)
]
```

### Hyperparameter Sweeps
```python
# Use WandB sweeps
sweep_config = {
    'method': 'bayes',
    'parameters': {
        'learning_rate': {'values': [1e-5, 5e-5, 1e-4]},
        'beta': {'values': [0.0, 0.05, 0.1]},
        'temperature': {'values': [0.7, 0.8, 0.9]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="qwen3_4b_grpo")
wandb.agent(sweep_id, function=train)
```

### Multi-GPU Training
```python
# Currently single-GPU optimized
# For multi-GPU, use DistributedDataParallel:
model = torch.nn.parallel.DistributedDataParallel(model)
```

## 📚 Documentation

- `GRPO_TRAINING_EXPLANATION.md`: Algorithm deep dive
- `CRITICAL_BUG_FIX.md`: normalize_number() bug details
- `DETAILED_EVAL_LOGGING.md`: Evaluation system guide
- `IMPROVEMENTS_SUMMARY.md`: All optimizations
- `.cursorrules_wandb`: WandB logging standards

## 🤝 Contributing

Key improvement areas:
- [ ] Multi-GPU support
- [ ] More sophisticated reward functions
- [ ] Online RL (vs offline dataset)
- [ ] Other model architectures
- [ ] Alternative RL algorithms (PPO, DPO)

## 📖 References

- QLoRA: https://arxiv.org/abs/2305.14314
- Unsloth: https://github.com/unslothai/unsloth
- GSM8K: https://github.com/openai/grade-school-math
- LoRA: https://arxiv.org/abs/2106.09685
