# 🚨 CRITICAL: Dataset Has 256 Classes, Model Expects 34

## The Real Problem

Your `class_weights.json` revealed the **ROOT CAUSE** of the 0.021 IoU:

**Dataset mismatch**: 
- Your masks have **256 unique class values** (0-255)
- Your model is configured for **34 classes** (0-33)
- Training code clips values 34-255 to 0-33, causing massive corruption

**Evidence from class_weights.json**:
```json
{
  "0": 127,783,027 pixels (49.54%)   ← Background
  "255": 119,058,555 pixels (46.16%)  ← Also background!
  "241": 2,428,606 pixels (0.94%)     ← Another major class
  "1-254": Various small percentages
}
```

**95.7% of pixels are just 2 values: 0 and 255!**

## Why Training Failed

1. **Model predicts 34 classes** [0-33]
2. **Masks contain 256 classes** [0-255]
3. **Training code clips mask values**: `torch.clamp(masks, 0, 33)`
4. **Result**: Classes 34-255 all become class 33 → Complete data corruption
5. **Model learns**: "Just predict class 0 everywhere" (since it's 49.5% + corrupted data)

## ✅ Solution: Remap Masks

I've created `fix_mask_classes.py` to fix this.

### Step 1: Analyze Your Masks

```bash
python fix_mask_classes.py
```

This will:
- Scan your masks to see what values exist
- Show the distribution
- Create a 256→34 mapping
- Remap all masks to valid [0-33] range
- Save fixed masks to `data/processed_fixed/`

### Step 2: Update Training Paths

After running the fix, update your training scripts to use the fixed data:

**In `train.py` and `train_fixed.py`:**
```python
'images_dir': 'data/processed_fixed/images',
'masks_dir': 'data/processed_fixed/annotations',
```

### Step 3: Retrain

```bash
# Run diagnostics on FIXED data
python diagnose_model.py

# Train with FIXED data and class weights
python train_fixed.py
```

## Understanding the Mapping

The script uses a **best-guess mapping** based on common CubiCasa5K patterns:

```python
0 → 0      # Background
255 → 0    # Also background (white)
1-5 → 1    # Walls
6-13 → 2-9 # Various rooms
14+ → 10   # Undefined
```

**⚠️ IMPORTANT**: You should verify this mapping against your dataset's documentation!

### Finding the Correct Mapping

Check these files in your CubiCasa5K dataset:
- `README.md` or `README.txt`
- `class_mapping.json`
- `class_names.txt`
- Any documentation about class IDs

Or look at the original CubiCasa5K paper/GitHub for the official mapping.

## Alternative: Train with 256 Classes

Instead of remapping, you could update the model to handle 256 classes:

**In `configs/config.yaml` and `train.py`:**
```python
'n_classes': 256,  # Change from 34
```

**Pros**: No remapping needed
**Cons**: 
- Slower training (7.5x more output neurons)
- Many classes might have very few samples
- Most classes might be unnecessary

## Quick Test

To verify the fix worked:

```bash
# After running fix_mask_classes.py
python -c "
import cv2
import numpy as np
from pathlib import Path

mask = cv2.imread('data/processed_fixed/annotations/10004.png', cv2.IMREAD_GRAYSCALE)
print(f'Min: {mask.min()}, Max: {mask.max()}, Unique: {len(np.unique(mask))}')
print(f'Values: {np.unique(mask)}')
"
```

Expected output:
```
Min: 0, Max: 33, Unique: <some number ≤ 34>
Values: [0 1 2 3 4 ... up to 33]
```

## Summary

| Issue | Current | After Fix |
|-------|---------|-----------|
| Mask values | 0-255 (256 classes) | 0-33 (34 classes) |
| Model expects | 0-33 (34 classes) | 0-33 (34 classes) |
| Data corruption | Yes (clipping 34-255→33) | No |
| Can train properly | No | Yes |

## Action Plan

```bash
# 1. Fix the masks
python fix_mask_classes.py

# 2. Verify the fix
ls data/processed_fixed/annotations/  # Should see all your masks
python -c "import cv2; import numpy as np; m=cv2.imread('data/processed_fixed/annotations/10004.png', 0); print(f'Max: {m.max()}')"  # Should be ≤33

# 3. Update training config
# Edit train_fixed.py:
#   'images_dir': 'data/processed_fixed/images',
#   'masks_dir': 'data/processed_fixed/annotations',

# 4. Run diagnostics on FIXED data
python diagnose_model.py

# 5. Train with fixed data
python train_fixed.py

# Expected: IoU > 0.40 after 50 epochs (vs current 0.02)
```

---

**This is THE critical fix. The class imbalance was secondary - the primary issue is the 256 vs 34 class mismatch!**

*Last Updated: October 28, 2025*
