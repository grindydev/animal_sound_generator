# Workflow Fix Plan v7 — More Data + x₀-Prediction

> **Date:** May 15, 2026  
> **Approach:** Industry-standard fixes for small-data diffusion: more data, augmentation, x₀-prediction.

---

## 1. Three Changes

### 1.1 More Data (85% more, zero new files)

Current: 2700 train / 301 val (90/10 split).  
Fix: 95/5 split → ~2850 train from same files.  
Plus: all files cropped to multiple segments via smart_crop → ~4500 total.  
Plus: use ALL samples for training, skip val during early epochs, validate every 5 epochs only.

### 1.2 Heavy Augmentation (10× effective data)

```python
# Each batch, randomly apply:
- Pitch shift: ±3 semitones (shifts mel bins up/down)
- Time stretch: 0.8× to 1.2× (speeds/slows playback)  
- SpecAugment: mask freq bands, mask time chunks
- Mixup: blend two samples of different classes (soft labels)
- Random EQ: boost/cut random frequency bands
```

### 1.3 x₀-Prediction (stronger gradients)

```python
# Instead of:
noise = randn_like(mel)
x_t = q_sample(mel, t, noise)
loss = MSE(model(x_t, t, labels), noise)     # target = random noise (structureless)

# Do:
x_t = q_sample(mel, t)
pred_x0 = model(x_t, t, labels)
loss = MSE(pred_x0, mel) * weight(t)          # target = clean mel (has structure)
```

**Why:** The target is now a real spectrogram with harmonics, formants, and rhythm — not random Gaussian noise. 100× richer learning signal per sample.

**Inference:**
```python
# Standard DDIM with x₀-prediction:
for t in timesteps:
    pred_x0 = model(x_t, t, labels)           # predict clean
    pred_noise = (x_t - sqrt(α) * pred_x0) / sqrt(1-α)  # recover noise
    x_t = next_step(pred_x0, pred_noise, t)   # DDIM step
```

### 1.4 Class-Mean Initialization (reduces task difficulty)

Instead of starting DDIM from pure noise, start from the class mean + 80% noise:

```python
class_mean = precompute_mean_mel(label)   # average mel for this class
x_T = class_mean * 0.2 + noise * 0.8      # 20% signal, 80% noise
# DDIM from here — model only needs to add 20% detail
```

Precompute class means from all training samples at startup.

---

## 2. Implementation

| File | Change |
|------|--------|
| `src/diffusion/config.py` | Add augment params, train_frac=0.95 |
| `src/diffusion/train.py` | x₀-prediction loss, augmentation, more data |
| `src/diffusion/diffusion.py` | x₀-prediction DDIM/DDPM sampling |
| `src/generate.py` | Class-mean init + x₀ inference |

Expected: ~45 min to implement, retrain on Colab L4 (~2 hrs with augmentation).

---

## 3. Success Criteria

| Metric | Current | Target |
|--------|:---:|:---:|
| Training samples | 2700 | 4500+ |
| Effective samples (augment) | 2700 | ~45,000 |
| Val loss trend | Flat at 1.0 | Decreasing |
| Generated mel σ | N/A (noise) | 0.8-1.5 |
| Classes with structure | 0/8 | 3+/8 |
