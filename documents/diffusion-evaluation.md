# Diffusion Evaluation — May 14, 2026

> **Model:** `diffusion_unet_train_best.pth` (61M params, 232 MB)  
> **Training:** 50 epochs, batch=4×4, real mel spectrograms, DDPM

---

## 1. Epoch Comparison (v4 vs v5)

| Model | Epoch | Val Loss | Mel σ | Classes OK | Trend |
|-------|:-----:|:--------:|:-----:|:----------:|-------|
| v4 | 6 | 0.60 | 7.2 | 3/8 | — |
| v4 | 36 | 0.09 | 2.5 | 1/8 | ↓ degrading |
| v5 | 10 | 0.60 | **1.2** | **2/8** | ✅ Best so far |
| v5 | 25 | 0.19 | 0.8 | 1/8 | ↓ degrading |

**Key finding:** The best generation quality happens at medium val loss (0.60), not low val loss (0.09-0.19). As loss drops, the model converges to predicting the mean noise → flat, uninteresting outputs. This is **diffusion posterior collapse** on a small dataset.

## 2. v5 Epoch 25 Results

| Class | Mel σ | RMS | Peak Hz | Bass% | Verdict |
|-------|:-----:|:---:|:-------:|:-----:|---------|
| Dog | 0.86 | 0.03 | 11025 | 9% | ❌ Noise |
| Cat | 0.72 | 0.02 | 11025 | 14% | ❌ Noise |
| Rooster | 0.79 | 0.04 | 11025 | 10% | ❌ Noise |
| Frog | 0.74 | 0.02 | 11025 | 14% | ❌ Noise |
| Crow | 0.66 | 0.04 | 11025 | 2% | ❌ Noise |
| Insect | 0.80 | 0.03 | 11025 | 10% | ❌ Noise |
| Hen | 0.81 | 0.03 | 11025 | 12% | ❌ Noise |
| **Noise** | 0.69 | 0.02 | **65** | **19%** | ✅ Structure |

Mel σ is now 0.7-0.9 — below the target of 1.0. The model is producing spectrograms that are too flat and missing high-frequency content. Only the Noise class (which is genuinely flat in real data) survives.

## 3. Root Cause: Diffusion Posterior Collapse on Small Data

With 2700 samples and 61M params, the UNet has enough capacity to memorize individual noise patterns. The MSE loss for noise prediction is minimized by predicting near-zero noise (the expected value across the dataset). At high val loss (epoch 10, val=0.60), the model still produces diverse outputs. At low val loss (epoch 25, val=0.19), it converges to flat predictions.

**The best checkpoint for generation is around epoch 10-15, not epoch 50.**

## 4. Fixes

| Fix | Expected Impact |
|-----|----------------|
| **Use epoch ~12 checkpoint** | Likely best quality (val ~0.40) |
| **Reduce model to 30M params** | Less capacity → forced to generalize |
| **Add SpecAugment during training** | Prevents memorization |
| **Use L1 loss instead of MSE** | Penalizes mean regression less |
| **Classifier-free guidance** | Sharpens output at inference |
| **Get more data** | More samples → harder to memorize |
