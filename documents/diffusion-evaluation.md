# Diffusion Evaluation — May 14, 2026

> **Model:** `diffusion_unet_train_best.pth` (61M params, 232 MB)  
> **Training:** 50 epochs, batch=4×4, real mel spectrograms, DDPM

---

## 1. Epoch 6 vs Epoch 36 Comparison

| Metric | Epoch 6 (val=0.60) | Epoch 36 (val=0.09) | Trend |
|--------|:---:|:---:|:---:|
| Mel σ | **7.2** | **2.5** | ✅ 3× better |
| Mel mean | ~0 | ~0.6 | ⚠️ Drifted |
| Classes with structure | **3/8** | **1/8** | ❌ Getting worse |
| Audio RMS | 0.6-0.7 | 0.5-0.8 | ≈ Same |
| Peak at Nyquist | 5/8 classes | **7/8 classes** | ❌ More noise |

**Lower val loss ≠ better generation.** The model is overfitting — memorizing training noise patterns that don't generalize to pure noise denoising.

---

## 2. Epoch 36 Per-Class Results

| Class | Mel μ | Mel σ | RMS | Peak Hz | Bass% | Verdict |
|-------|:-----:|:-----:|:---:|:-------:|:-----:|---------|
| Dog | +0.52 | 2.68 | 0.54 | 11025 | 18% | ❌ Noise |
| Cat | +1.00 | 2.67 | 0.84 | 11025 | 7% | ❌ Noise |
| **Rooster** | +0.65 | 2.45 | 0.55 | **883** | **27%** | ✅ Structure |
| Frog | +0.79 | 2.07 | 0.65 | 11025 | 16% | ❌ Noise |
| Crow | +1.12 | 2.37 | 0.78 | 11025 | 14% | ❌ Noise |
| Insect | +0.24 | 2.39 | 0.77 | 11025 | 9% | ❌ Noise |
| Hen | +0.10 | 2.48 | 0.50 | 11025 | 20% | ❌ Noise |
| Noise | +0.62 | 1.97 | 0.70 | 11025 | 12% | ❌ Noise |

---

## 3. Key Finding: Mel Statistics Improving, Content Degrading

```
Epoch 6 (high val loss):  mel σ=7.2, 3/8 classes show structure
Epoch 36 (low val loss): mel σ=2.5, 1/8 classes show structure
                    Target: mel σ=1.0, 8/8 classes show structure
```

The model is learning the CORRECT AMPLITUDE (σ improving toward 1.0), but the SPECTRAL CONTENT is degrading. This is a classic **mean-prediction** problem in diffusion — as loss drops, the model starts predicting the expected (average) noise, not the specific noise for each sample. The result: outputs converge to the class average spectrogram, which lacks detail.

---

## 4. Recommended Fixes (NOT APPLIED)

| Fix | Expected impact |
|-----|----------------|
| **Tighten DDIM clamp: [-10,10] → [-4,4]** | Forces output to valid range |
| **Use classifier-free guidance (CFG=1.5)** | Sharpens output by comparing conditioned vs unconditioned predictions |
| **Increase number of training samples (more data)** | 2700 samples / 61M params = massive overfit risk |
| **Try checkpoint at epoch ~20-25** | Lower val loss than epoch 6, but still has content diversity |
| **Reduce model size** | 30M params might be enough for this dataset size |
| **Data augmentation** | Time shift, pitch shift, SpecAugment |
