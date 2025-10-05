# GRPO - Group Relative Policy Optimization

Original GRPO (Group Relative Policy Optimization) implementation for reinforcement learning from human feedback (RLHF) on language models.

## ⚠️ Status

**Legacy Implementation** - For production use, see `../training_with_unsloth/`

This directory contains the original GRPO trainer that was used as reference for building the refactored, production-ready implementation with Unsloth optimizations.

## 📁 Contents

```
grpo/
├── trainer.py          # Original GRPO trainer class
└── reward.py           # Basic reward utilities
```

## 🎯 What is GRPO?

**Group Relative Policy Optimization** is a reinforcement learning algorithm for training language models that:

1. **Groups responses**: Generates N responses per prompt
2. **Relative rewards**: Normalizes rewards within each group (advantage = reward - group_mean)
3. **Policy gradient**: Updates policy to maximize advantages
4. **Clipping**: PPO-style clipping for stable training

### Key Difference from PPO
```python
# PPO: Uses absolute advantages
advantage = reward - baseline_value_function(state)

# GRPO: Uses group-relative advantages  
advantage = (reward - mean(group_rewards)) / (std(group_rewards) + ε)
```

**Benefits**:
- No need to train separate value function
- More stable when rewards have high variance
- Better for comparing responses to same prompt

## 🔧 Algorithm

```python
for prompt in dataset:
    # 1. Generate group of responses
    responses = model.generate(prompt, num_samples=N)
    
    # 2. Compute rewards for each response
    rewards = [reward_fn(prompt, r, answer) for r in responses]
    
    # 3. Normalize within group
    advantages = (rewards - mean(rewards)) / (std(rewards) + ε)
    
    # 4. Compute log probabilities
    log_probs_old = get_log_probs(responses, detached=True)
    log_probs_new = get_log_probs(responses, detached=False)
    
    # 5. Policy ratio
    ratio = exp(log_probs_new - log_probs_old)
    
    # 6. Clipped loss (PPO-style)
    loss1 = ratio * advantages
    loss2 = clip(ratio, 1-ε, 1+ε) * advantages
    loss = -mean(min(loss1, loss2))
    
    # 7. Update model
    loss.backward()
    optimizer.step()
```

## 📊 Key Components

### 1. Group Normalization
```python
# Within each prompt's N responses
mean_reward = mean([r1, r2, ..., rN])
std_reward = std([r1, r2, ..., rN])
advantages = (rewards - mean_reward) / (std_reward + 1e-8)
```

**Effect**: Responses with above-average rewards get positive advantage, below-average get negative.

### 2. Advantage-Based Learning
```python
# Positive advantage → increase probability
if advantage > 0:
    loss = -log_prob * advantage  # Minimize negative = maximize prob
    
# Negative advantage → decrease probability  
if advantage < 0:
    loss = -log_prob * advantage  # Minimize negative of negative = decrease prob
```

### 3. Policy Clipping
```python
ratio = exp(log_prob_new - log_prob_old)
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε typically 0.2

# Prevents too large policy updates
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

## 🆚 Comparison with Other Methods

| Method | Value Function | Advantage Calc | Stability |
|--------|----------------|----------------|-----------|
| **GRPO** | ❌ None needed | Group-relative | High |
| **PPO** | ✅ Trained V(s) | Absolute (A=R-V) | Medium |
| **REINFORCE** | ❌ None | Absolute (R) | Low |
| **DPO** | ❌ None | Preference pairs | High |

## 🔬 Why Use GRPO?

### Advantages
✅ **No value function**: Simpler than PPO, one less model to train  
✅ **Variance reduction**: Group normalization reduces gradient variance  
✅ **Natural comparison**: Directly compares responses to same prompt  
✅ **Stable**: PPO-style clipping prevents policy collapse  

### Disadvantages
❌ **Needs multiple samples**: Requires N>1 responses per prompt (more compute)  
❌ **Group dependency**: Quality depends on group diversity  
❌ **Memory**: Must store N responses + log probs simultaneously  

## 📈 Typical Hyperparameters

```python
num_generations = 4           # N responses per prompt
clip_eps = 0.2               # PPO clipping threshold
learning_rate = 1e-5         # Conservative LR for stability
gradient_accumulation = 4    # Effective batch size
beta = 0.05                  # KL penalty (optional)
```

## 🚀 Migration to New Framework

The original implementation has been refactored in `../training_with_unsloth/` with:

**Improvements**:
- ✅ QLoRA / 4-bit quantization support
- ✅ Unsloth optimizations (2x faster)
- ✅ Comprehensive evaluation (Pass@K metrics)
- ✅ Detailed logging (every answer tracked)
- ✅ WandB integration
- ✅ Better memory efficiency (16GB VRAM)
- ✅ Configuration management (YAML)
- ✅ Multiple reward functions

**Maintained**:
- ✅ Same GRPO algorithm
- ✅ Group-relative advantages
- ✅ PPO-style clipping
- ✅ Compatible API

## 🔄 Migration Guide

### Old (this directory):
```python
from grpo.trainer import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    args=args
)
trainer.train()
```

### New (recommended):
```python
from training_with_unsloth.framework.grpo_trainer import GRPOTrainer, GRPOArguments
from training_with_unsloth.framework.config import ExperimentConfig

# Load from YAML
config = ExperimentConfig.from_yaml("config.yaml")

# Setup trainer
args = GRPOArguments()
args.lr = config.learning_rate
# ... etc

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward, format_reward],
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer
)
trainer.train()
```

## 📖 Key Differences

| Feature | grpo/ (legacy) | training_with_unsloth/ (new) |
|---------|----------------|------------------------------|
| Quantization | FP16/BF16 only | 4-bit, 8-bit, FP16, BF16 |
| VRAM (4B model) | 24GB+ | 8-10GB |
| Training speed | 1x | 2x (Unsloth) |
| Evaluation | Basic metrics | Pass@K with full logs |
| Configuration | Python args | YAML + dataclass |
| Logging | Print only | WandB + detailed eval |
| Reward functions | Single function | Multiple composable |

## 🔍 Understanding GRPO Loss

```python
# Example with 4 responses to "What is 2+2?"
responses = ["4", "5", "4", "3"]
rewards = [2.0, 0.0, 2.0, 0.0]  # Correctness only

# Group statistics
mean_reward = 1.0
std_reward = 1.0

# Advantages
advantages = [
    (2.0 - 1.0) / 1.0 = +1.0,  # Above average → push up
    (0.0 - 1.0) / 1.0 = -1.0,  # Below average → push down
    (2.0 - 1.0) / 1.0 = +1.0,  # Above average → push up
    (0.0 - 1.0) / 1.0 = -1.0,  # Below average → push down
]

# Loss for each
loss[0] = -log_prob("4") * 1.0   # Increase P("4")
loss[1] = -log_prob("5") * -1.0  # Decrease P("5")
loss[2] = -log_prob("4") * 1.0   # Increase P("4")
loss[3] = -log_prob("3") * -1.0  # Decrease P("3")

# Result: Model learns "4" is better than "3" or "5"
```

## 📚 References

- **GRPO Original**: [Check paper/blog if available]
- **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **RLHF**: [Training language models to follow instructions](https://arxiv.org/abs/2203.02155)

## 🤝 Contributing

This is legacy code kept for reference. For new features, contribute to `../training_with_unsloth/`.

## 📝 License

See parent directory LICENSE.

