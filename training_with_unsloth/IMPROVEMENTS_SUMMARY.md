# GRPO Training Improvements Summary

## Issues Identified

### 1. **Dataset Too Small**
- **Problem**: Only 20 training samples, 10 test samples
- **Impact**: Model cannot learn generalizable patterns
- **Solution**: Increased to 200 training, 50 test samples

### 2. **Pass@1 Not Improving**
- **Problem**: Greedy accuracy stuck at 50% across epochs
- **Impact**: Model's best answer isn't getting better
- **Root Cause**: Small dataset + low learning rate + no KL regularization
- **Solution**: Higher LR (5e-5), added KL penalty (beta=0.05)

### 3. **Pass@5 = Pass@1 (Should be ≥)**
- **Problem**: Pass@5 should always be ≥ Pass@1, but they're equal
- **Impact**: Model not generating diverse samples
- **Root Cause**: Temperature too low
- **Solution**: Increased training temp (0.7→0.9), eval temp (0.75→0.8)

### 4. **Reward Functions Always 0**
- **Problem**: `hard_format_reward` and `mark_reward` always return 0
- **Impact**: Model gets no feedback on formatting, only correctness
- **Root Cause**: 
  - Model generates: `<answer>123</answer>` (no newlines, no `<think>` tags)
  - Reward expects: `<think>\n...\n</think>\n<answer>\n...\n</answer>\n`
- **Solution**: Made rewards more lenient with partial credit

## Changes Implemented

### 1. **Improved Config** (`eval_test_improved.yaml`)
```yaml
# Dataset
train_samples: 200  # was 20
test_samples: 50    # was 10

# Training
learning_rate: 5e-5  # was 1e-5 (5x higher)
num_epochs: 10       # was 50 (reduced for faster iteration)
beta: 0.05           # was 0.0 (added KL regularization)

# Generation
temperature: 0.9          # was 0.7 (more diversity)
eval_temperature: 0.8     # was 0.75 (more diversity in eval)
```

### 2. **More Lenient Reward Functions**

#### **Before: `hard_format_reward`**
```python
# Required exact pattern with newlines
pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
return [0.5 if match else 0.0]
```

#### **After: `hard_format_reward`**
```python
# Partial credit system:
if has_think and has_answer:
    reward = 0.5  # Full credit
elif has_answer:
    reward = 0.25  # Partial credit for answer tag only
else:
    reward = 0.0  # No tags
```

#### **Before: `mark_reward`**
```python
# Required newlines after each tag
if text.count("<think>\n") == 1:
    reward += 0.125
# ... strict newline requirements
```

#### **After: `mark_reward`**
```python
# Accept tags with or without newlines
if "<think>" in text:
    reward += 0.125
if "</think>" in text:
    reward += 0.125
# ... etc (4 tags × 0.125 = 0.5 total)
```

### 3. **Added Epoch Reward Tracking**

**New Metrics Logged to WandB:**
- `epoch/accumulated_reward`: Sum of all rewards in the epoch
- `epoch/mean_reward`: Average reward per sample in the epoch

**Console Output Example:**
```
=== Epoch 1/10 Summary ===
Accumulated Reward: 180.5
Mean Reward: 2.256
Total Samples: 80
```

### 4. **Updated Metric Definitions**

| Metric | Definition | What It Measures |
|--------|------------|------------------|
| **Pass@1** | Greedy (temp=0) accuracy | Best single answer correctness |
| **Pass@5** | ≥1 correct in 5 samples (temp=0.8) | Success with 5 diverse attempts |
| **Pass@10** | ≥1 correct in 10 samples (temp=0.8) | Success with 10 diverse attempts |
| **epoch/accumulated_reward** | Sum of all rewards in epoch | Total reward signal strength |
| **epoch/mean_reward** | Average reward per sample | Per-sample reward quality |

**Expected Relationship:**
```
Pass@1 ≤ Pass@5 ≤ Pass@10
```

## Expected Improvements

### With Current Setup:
- ❌ Pass@1 = 0.5, flat (model not improving)
- ❌ Pass@5 = 0.5 (should be > Pass@1)
- ❌ Pass@10 = 0.5-0.6 (barely better)
- ❌ `hard_format_reward` = 0 (no formatting feedback)
- ❌ `mark_reward` = 0 (no tag feedback)

### With Improved Setup:
- ✅ Pass@1 should increase over epochs (0.5 → 0.6 → 0.7+)
- ✅ Pass@5 > Pass@1 (more diverse samples help)
- ✅ Pass@10 > Pass@5 (even more chances)
- ✅ `hard_format_reward` > 0 (partial credit for `<answer>` tags)
- ✅ `mark_reward` > 0 (credit for all tags)
- ✅ `epoch/accumulated_reward` increases over time

## How to Run Improved Config

```bash
cd /home/hula0401/learning_llm
UV_PYTHON=python3 uv run python3 \
  training_with_unsloth/experiments/run_eval_test.py \
  training_with_unsloth/experiment_configs/eval_test_improved.yaml
```

## Why These Changes Help

### 1. **Larger Dataset**
- More examples → better generalization
- Reduces overfitting risk
- Model sees more diverse problem types

### 2. **Higher Learning Rate**
- Faster convergence
- Stronger updates from reward signals
- Better for small datasets with clear signals

### 3. **KL Regularization (beta=0.05)**
- Prevents model from drifting too far from initialization
- Stabilizes training
- Reduces catastrophic forgetting

### 4. **Higher Temperature**
- More diverse generation during training
- Explores answer space better
- Helps with Pass@5/10 metrics

### 5. **Lenient Reward Functions**
- Model gets feedback even without perfect formatting
- Encourages gradual improvement
- Partial credit guides toward full credit

### 6. **Epoch Reward Tracking**
- Shows if training is actually working
- `accumulated_reward` should increase over epochs
- Early warning if something is wrong

## Monitoring Training Success

### Good Signs:
✅ `epoch/accumulated_reward` trending up
✅ `epoch/mean_reward` increasing
✅ Pass@1 improving over epochs
✅ Pass@5 > Pass@1 consistently
✅ `hard_format_reward` and `mark_reward` > 0

### Bad Signs:
❌ `epoch/accumulated_reward` flat or decreasing
❌ Pass@1 not changing
❌ Pass@5 = Pass@1 (no diversity)
❌ All format rewards still 0

## Next Steps

If training still doesn't improve:
1. Try even higher LR (1e-4)
2. Reduce `clip_eps` from 0.2 to 0.1 (smaller policy updates)
3. Increase `beta` from 0.05 to 0.1 (more KL constraint)
4. Check WandB for reward distribution patterns
5. Add more reward functions for intermediate steps

