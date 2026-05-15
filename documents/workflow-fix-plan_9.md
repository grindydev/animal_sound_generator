# Workflow Fix Plan v9 — Class Balance + More Epochs

> **Date:** May 15, 2026  
> **Status:** Reviewing.  
> **Builds on:** v8 ABC (GAN discriminator — first working audio, 6/8 classes structued)  
> **Problem:** Model plateaued because 68% of training is Noise + Dog. Rare classes invisible.

---

## 1. What v8 Achieved

| Class | Files | v8 Audio | Problem |
|-------|:---:|:---:|------|
| Noise | 1222 (42%) | ✅ | Unnecessary — this is background, not an animal |
| Dog | 750 (26%) | ✅ | Low hum, not bark yet |
| Insect | 371 (13%) | ⚠️ | Borderline |
| Cat | 303 (10%) | ⚠️ | Near-noise |
| Rooster | 136 (4%) | ✅ | Low hum |
| Hen | 86 (3%) | ✅ | Low hum |
| Crow | 72 (2%) | ✅ | Low hum |
| Frog | 61 (2%) | ⚠️ | Worst — 2 batches per epoch |

The model sees Frog ~2 times per epoch. Dog ~25 times. Of course it doesn't learn Frog.

## 2. Fix 1: Class-Balanced Sampling

Each batch: guaranteed 1 sample from each of 8 classes (effective batch=8 with balance).

```python
class BalancedDiffusionDataset(Dataset):
    """Each epoch: oversample rare classes to equalize exposure."""
    def __init__(self):
        # Group samples by class
        self.class_samples = {c: [] for c in range(8)}
        for path, label in all_samples:
            self.class_samples[label].append((path, label))
        
        # Target: each class appears N_max times per epoch
        self.max_per_class = max(len(v) for v in self.class_samples.values())
    
    def __len__(self):
        return self.max_per_class * 8  # balanced
    
    def __getitem__(self, idx):
        cls = idx % 8
        samples = self.class_samples[cls]
        s = samples[idx // 8 % len(samples)]  # wrap around for rare classes
        return self._load_mel(s)
```

Result: Dog (750) stays at natural frequency. Frog (61) gets repeated 12× per epoch with different augmentation → equivalent to 732 effective samples.

## 3. Fix 2: Reduce/Remove Noise Class

Option A: Remove Noise class entirely → 7 classes, 1,779 files.

Option B: Keep Noise but downsample to 200 files (match Dog count).

**Recommendation:** Option A. Noise is background ambiance, not an animal sound. Removing it gives each remaining class more attention and removes the dominant signal.

## 4. Fix 3: More Epochs

50 epochs with balanced sampling = each rare class seen 600+ times instead of 100×.  
100 epochs gives 2× more passes. Val was still dropping at epoch 50.

## 5. Fix 4: Simpler Augmentation for Rare Classes

Frog (61 files) with current strong augmentation (freq mask + time mask + noise + gain + dropout) → 61 × 2^5 = 1,952 effective samples. More than enough to converge if seen frequently.

## 6. Changes

| File | Change | Impact |
|------|--------|--------|
| `src/diffusion/config.py` | `num_classes: 7`, `balance_classes: True`, `noise_drop: True` | Class balance |
| `src/diffusion/train.py` | `BalancedDiffusionDataset` class, drop Noise | Sampling |
| `colab_train.ipynb` | Update cell labels for 7 classes | Colab |

## 7. Expected Outcome

| Metric | v8 (current) | v9 Target |
|--------|:---:|:---:|
| Best val | 0.91 | 0.60-0.75 |
| Recognizable classes | 0/8 | 3+/7 (Dog bark, Crow caw, Hen cluck) |
| Frog audible | Noise | Low croak |
| Dog σ | 0.14 | 0.25+ |
| Training time | 70 min | 100 min (100 epochs) |
