# Critical Bug Fix: normalize_number Function

## **🐛 Bug Discovered**

The `normalize_number()` function was **too strict** - it only removed commas but not dollar signs or units!

### **What Was Happening:**
```python
Model Answer: "$108"
Ground Truth: "108"
normalize_number("$108") = "$108"  # ❌ Didn't remove $!
Result: ✗ WRONG (even though numerically correct!)
```

### **Examples of False Negatives:**

| Question | Model Answer | Ground Truth | Old Result | Actual |
|----------|--------------|--------------|------------|--------|
| Q1 | `$108` | `108` | ✗ WRONG | ✓ Numerically correct! |
| Q6 | `$1350` | `1350` | ✗ WRONG | ✓ Numerically correct! |
| Q7 | `1350 minutes` | `1350` | ✗ WRONG | ✓ Numerically correct! |
| Q8 | `40 minutes` | `40` | ✗ WRONG | ✓ Numerically correct! |
| Q11 | `$3,450` | `3450` | ✗ WRONG | ✓ Numerically correct! |
| Q14 | `$26` | `26` | ✗ WRONG | ✓ Numerically correct! |
| Q16 | `$56` | `56` | ✗ WRONG | ✓ Numerically correct! |
| Q19 | `$360,000` | `360000` | ✗ WRONG | ✓ Numerically correct! |

**Impact**: At least **8 out of 20** questions were marked wrong due to this bug!

## **✅ Fix Applied**

### **Old normalize_number (BUGGY):**
```python
def normalize_number(text: str) -> str:
    """Normalize numbers by removing commas and extra whitespace."""
    # Remove commas from numbers (e.g., "360,000" -> "360000")
    normalized = re.sub(r'(\d),(\d)', r'\1\2', text)
    return normalized.strip()
```

**Problems:**
- ❌ Doesn't remove dollar signs
- ❌ Doesn't remove units (minutes, dollars, etc.)
- ❌ Leaves "$108" as "$108"

### **New normalize_number (FIXED):**
```python
def normalize_number(text: str) -> str:
    """
    Normalize numbers by removing:
    - Dollar signs ($)
    - Commas (360,000 -> 360000)
    - Units (minutes, dollars, etc.)
    - Extra whitespace
    """
    # Remove dollar signs
    normalized = text.replace('$', '')
    
    # Remove commas from numbers
    normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)
    
    # Remove units - keep only the leading number
    match = re.match(r'^(\d+(?:\.\d+)?)', normalized.strip())
    if match:
        normalized = match.group(1)
    
    return normalized.strip()
```

**Improvements:**
- ✅ Removes dollar signs: `$108` → `108`
- ✅ Removes commas: `360,000` → `360000`
- ✅ Removes units: `40 minutes` → `40`
- ✅ Handles combinations: `$3,450` → `3450`

## **Test Results**

```
✓ '$108'           → '108'
✓ '$1350'          → '1350'
✓ '1350 minutes'   → '1350'
✓ '40 minutes'     → '40'
✓ '$360,000'       → '360000'
✓ '$3,450'         → '3450'
✓ '$26'            → '26'
✓ '$56'            → '56'
```

## **Expected Impact**

### **Before Fix:**
```
BASELINE (Epoch 0):
  Pass@1: ~0.40 (8+ questions marked wrong due to formatting)
  Pass@5: ~0.60
  Pass@10: ~0.70
```

### **After Fix:**
```
BASELINE (Epoch 0):
  Pass@1: ~0.80+ (formatting issues resolved!)
  Pass@5: ~0.85+
  Pass@10: ~0.90+
```

**Expected improvement: +40% on Pass@1!**

## **Why This Is Critical**

### **Training Impact:**
1. **Reward Signal**: Model was getting 0 reward for numerically correct answers
2. **Learning**: Model couldn't learn what's actually correct
3. **Pass@1 Stuck**: This is why Pass@1 wasn't improving!

### **Before Fix:**
```
Model: "$108"
Ground Truth: "108"
Correctness Reward: 0.0  ❌ (mismatch due to $)
→ Model learns: "I'm wrong, try something else"
→ Model changes answer to something actually wrong
```

### **After Fix:**
```
Model: "$108"
Ground Truth: "108"
Both normalized to: "108"
Correctness Reward: 2.0  ✓ (match!)
→ Model learns: "I'm right, keep doing this"
→ Model reinforces correct behavior
```

## **Root Cause Analysis**

### **Why Didn't We Catch This Earlier?**

1. **Incomplete Testing**: Only tested comma removal, not other formatting
2. **GSM8K Format**: Ground truth answers are plain numbers, but models often add context
3. **Silent Failure**: No warning when normalization didn't match
4. **Looked Correct**: Model output looked right to humans (`$108` IS 108!)

### **Why It Matters More for Pass@1:**

- **Greedy decoding** (Pass@1) tends to add formatting (learned during pretraining)
- **Sampling** (Pass@5/10) sometimes produces bare numbers by chance
- So Pass@5/10 were less affected, but Pass@1 was severely impacted

## **Verification**

To verify the fix is working, check the training logs:

### **Look for:**
```bash
# After fix, these should show ✓ CORRECT:
Question 1/20
Ground Truth Answer: 108
Pass@1 (Greedy, temp=0.0):
  Extracted: '$108' → Normalized: '108'  # Now matches!
  ✓ CORRECT  # ← Should be correct now!
```

### **Before fix showed:**
```
  Extracted: '$108' → Normalized: '$108'  # Didn't match!
  ✗ WRONG (expected '108')
```

## **Lessons Learned**

1. ✅ **Test normalization thoroughly** with real model outputs
2. ✅ **Add unit tests** for all formatting variations
3. ✅ **Debug with actual examples** from evaluation logs
4. ✅ **Compare raw and normalized** values in logs
5. ✅ **Don't assume model outputs match GT format**

## **Next Steps**

1. ✅ Re-run baseline evaluation with fix
2. ✅ Check if Pass@1 is now higher (~0.80 instead of ~0.40)
3. ✅ Monitor if Pass@1 now improves during training
4. ✅ Verify rewards are now correctly assigned

## **Commands to Re-Test**

```bash
# Run training with fixed normalization
cd /home/hula0401/learning_llm
UV_PYTHON=python3 uv run python3 \
  training_with_unsloth/experiments/run_eval_test.py \
  training_with_unsloth/experiment_configs/eval_test_improved.yaml \
  2>&1 | tee training_log_fixed.txt

# Check baseline Pass@1 (should be much higher now)
grep "BASELINE RESULTS" -A 5 training_log_fixed.txt
```

This fix should **dramatically improve** training effectiveness! 🎉

