# Detailed Evaluation Logging

## Overview
The evaluation now prints **every single answer** for **every question** to help debug why Pass@1 isn't changing.

## What Gets Logged

### **Baseline Evaluation (Epoch 0 - Before Training)**
Shows how the model performs before any training:
```
################################################################################
# BASELINE EVALUATION (Before Training - Epoch 0)
################################################################################

================================================================================
Question 1/20
================================================================================
Prompt: Marie has 98 unread messages...
Ground Truth Answer: 7
--------------------------------------------------------------------------------
Pass@1 (Greedy, temp=0.0):
  Raw: <answer>14</answer>
  Extracted: '14' → Normalized: '14'
  ✗ WRONG (expected '7')
--------------------------------------------------------------------------------
Pass@5/10 Samples (temp=0.5):
  Sample  1: '14' ✗
  Sample  2: '7' ✓
  Sample  3: '14' ✗
  Sample  4: '7' ✓
  Sample  5: '14' ✗
  Sample  6: '7' ✓
  Sample  7: '14' ✗
  Sample  8: '14' ✗
  Sample  9: '7' ✓
  Sample 10: '14' ✗
--------------------------------------------------------------------------------
Summary for Q1:
  Pass@1:  ✗ (greedy)
  Pass@5:  ✓ (2/5 correct)
  Pass@10: ✓ (4/10 correct)
```

### **After Each Epoch**
- **Epochs 1-3**: Full verbose output (all questions, all answers)
- **Every 10 epochs** (10, 20, 30...): Full verbose output
- **Other epochs**: Summary only (no detailed per-question output)

## What Each Section Means

### **Pass@1 (Greedy)**
```
Pass@1 (Greedy, temp=0.0):
  Raw: <answer>14</answer>
  Extracted: '14' → Normalized: '14'
  ✗ WRONG (expected '7')
```
- **temperature=0.0**: Deterministic, always same answer
- **do_sample=False**: Greedy decoding (picks highest probability)
- **This is the model's "best" answer**
- If Pass@1 doesn't change, the greedy answer isn't improving

### **Pass@5/10 Samples**
```
Pass@5/10 Samples (temp=0.5):
  Sample  1: '14' ✗
  Sample  2: '7' ✓
  Sample  3: '14' ✗
  ...
```
- **temperature=0.5**: Stochastic sampling
- **do_sample=True**: Samples from probability distribution
- **Each sample can be different**
- Shows answer diversity and success rate

### **Summary**
```
Summary for Q1:
  Pass@1:  ✗ (greedy)
  Pass@5:  ✓ (2/5 correct)
  Pass@10: ✓ (4/10 correct)
```
- **Pass@1**: Did greedy decoding get it right?
- **Pass@5**: Was at least 1 of first 5 samples correct?
- **Pass@10**: Was at least 1 of all 10 samples correct?

## Example Output Pattern

### **Good Training (Model Improving)**
```
BASELINE:
  Q1: Pass@1 ✗ (greedy='14', GT='7')
  Q2: Pass@1 ✓ (greedy='20', GT='20')

EPOCH 1:
  Q1: Pass@1 ✗ (greedy='14', GT='7')  ← Still wrong
  Q2: Pass@1 ✓ (greedy='20', GT='20')  ← Still correct

EPOCH 5:
  Q1: Pass@1 ✗ (greedy='10', GT='7')  ← Different wrong answer (changing!)
  Q2: Pass@1 ✓ (greedy='20', GT='20')

EPOCH 10:
  Q1: Pass@1 ✓ (greedy='7', GT='7')   ← NOW CORRECT! 🎉
  Q2: Pass@1 ✓ (greedy='20', GT='20')

→ Pass@1: 0.5 → 0.5 → 0.5 → 1.0 (improved!)
```

### **Bad Training (Model Not Learning)**
```
BASELINE:
  Q1: Pass@1 ✗ (greedy='14', GT='7')
  Q2: Pass@1 ✓ (greedy='20', GT='20')

EPOCH 1:
  Q1: Pass@1 ✗ (greedy='14', GT='7')  ← Same wrong answer
  Q2: Pass@1 ✓ (greedy='20', GT='20')

EPOCH 10:
  Q1: Pass@1 ✗ (greedy='14', GT='7')  ← STILL same wrong answer
  Q2: Pass@1 ✓ (greedy='20', GT='20')

→ Pass@1: 0.5 → 0.5 → 0.5 → 0.5 (not improving!)
```

## Debugging with Detailed Logs

### **Scenario 1: Pass@1 Flat, Same Wrong Answers**
```
Q1: Always answers '14' (GT='7')
Q2: Always answers '100' (GT='50')
```
**Problem**: Model is stuck in local minimum, not exploring
**Solution**: 
- Increase learning rate
- Increase training temperature
- Check if rewards are actually flowing

### **Scenario 2: Pass@1 Flat, Different Wrong Answers**
```
Epoch 0: Q1='14'
Epoch 1: Q1='10'
Epoch 2: Q1='21'
Epoch 3: Q1='7' ✓
```
**Problem**: Model is exploring but slowly
**Solution**: This is normal! Just needs more epochs

### **Scenario 3: Pass@5 = Pass@1**
```
Pass@1:  ✗ (greedy='14')
Samples: ['14', '14', '14', '14', '14']
```
**Problem**: No diversity in sampling
**Solution**:
- Increase eval_temperature (currently 0.5)
- Check if temperature is being applied

### **Scenario 4: Pass@5 >> Pass@1 but Pass@1 not improving**
```
Pass@1:  ✗ (greedy='14')
Samples: ['14', '7', '14', '7', '14', '7', '14', '7', '14', '7']
```
**Problem**: Model can generate correct answers but greedy picks wrong one
**Solution**:
- Reward structure might favor diversity over correctness
- Check reward weights
- May need to emphasize correctness_reward

## Key Insights to Look For

### ✅ **Good Signs:**
1. **Greedy answers change over epochs** (even if still wrong)
2. **More correct samples in Pass@5/10 over time**
3. **Eventually greedy becomes correct**
4. **Pass@5 > Pass@1** (diversity working)

### ❌ **Bad Signs:**
1. **Greedy answer never changes** (stuck)
2. **All 10 samples are identical** (no diversity)
3. **Pass@1 stays flat for many epochs** (not learning)
4. **Samples get worse over time** (reward hacking)

## System Prompt Used

The evaluation uses the **same system prompt as training**:
```python
SYSTEM_PROMPT = "Please answer in English with the following format, keep your answer short:\n<think> your step-by-step reasoning </think>\n<answer> the final short answer only </answer>\n"
```

This ensures consistency between training and evaluation.

## Implementation Details

### **Pass@1 Calculation**
```python
# Greedy decoding
temperature = 0.0
do_sample = False
→ Always produces same answer (deterministic)
→ Count correct / total
```

### **Pass@5 Calculation**
```python
# Sample 10 responses
temperature = eval_temperature (0.5)
do_sample = True
num_return_sequences = 10
→ Take first 5 samples
→ Success if ANY are correct
```

### **Pass@10 Calculation**
```python
# Same 10 samples as Pass@5
→ Use all 10 samples
→ Success if ANY are correct
```

**Note**: Pass@5 and Pass@10 use the same generation batch, just check different amounts!

## Output File Redirect

To save detailed logs:
```bash
UV_PYTHON=python3 uv run python3 \
  training_with_unsloth/experiments/run_eval_test.py \
  training_with_unsloth/experiment_configs/eval_test_improved.yaml \
  2>&1 | tee training_log.txt
```

Then search the log:
```bash
# Find all baseline evaluations
grep -A 20 "BASELINE EVALUATION" training_log.txt

# Find all epoch 1 evaluations
grep -A 20 "EVALUATION - End of Epoch 1" training_log.txt

# Find all Pass@1 results
grep "Pass@1 (Greedy" training_log.txt
```

