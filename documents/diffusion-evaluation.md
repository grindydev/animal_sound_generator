# Diffusion Evaluation — May 14, 2026

> **Model:** `diffusion_unet_train_best.pth` (61M params, 232 MB)  
> **Training:** 50 epochs, batch=4×4, real mel spectrograms, DDPM

---

## 1. Mel Spectrogram Quality

| Class | Mean | Std | Min | Max | Issue |
|-------|:---:|:---:|:---:|:---:|-------|
| Dog | +0.72 | **7.19** | -10 | +10 | ⚠️ std=7× too high |
| Cat | -0.47 | **7.28** | -10 | +10 | ⚠️ |
| Rooster | +0.15 | **7.17** | -10 | +10 | ⚠️ |
| Frog | -0.13 | **7.20** | -10 | +10 | ⚠️ |
| Crow | +0.01 | **7.31** | -10 | +10 | ⚠️ |
| Insect | +0.20 | **7.07** | -10 | +10 | ⚠️ |
| Hen | -0.29 | **7.20** | -10 | +10 | ⚠️ |
| Noise | -1.04 | **7.22** | -10 | +10 | ⚠️ |

**Expected:** mean≈0, std≈1 (normalized dB spectrogram).  
**Actual:** std≈7.2 for ALL classes. Values clamped at [-10, +10] by DDIM `x_0_pred.clamp(-10, 10)`.

**Root cause:** The DDIM sampling amplifies prediction errors. At high timesteps (t near T), the formula `x_0 = (x_t - sqrt(1-α_t)·pred_noise) / sqrt(α_t)` divides by sqrt(α_t) ≈ 0.02, amplifying any noise prediction error 50×. If the UNet hasn't fully converged, this produces values far outside the training range.

The clamp at [-10, 10] catches the worst but allows values 10× the normal range through.

## 2. Audio Quality

| Class | RMS | Peak Freq | Bass | Flatness | Verdict |
|-------|:---:|:---------:|:----:|:--------:|---------|
| Dog | 0.68 | 11025 Hz | 21% | 0.21 | ⚠️ Hissy |
| **Cat** | 0.61 | **43 Hz** | 23% | 0.21 | ✅ Has structure |
| Rooster | 0.64 | 11025 Hz | 23% | 0.19 | ⚠️ Hissy |
| **Crow** | 0.63 | **65 Hz** | 24% | 0.17 | ✅ Has structure |
| **Insect** | 0.63 | **22 Hz** | 22% | 0.18 | ✅ Has structure |
| Hen | 0.66 | 11025 Hz | 17% | 0.16 | ⚠️ Hissy |
| Noise | 0.73 | 11025 Hz | 9% | 0.31 | ❌ White noise |
| Frog | 0.63 | 11025 Hz | 21% | 0.18 | ⚠️ Hissy |

- **3/8 classes** (Cat, Crow, Insect) produce non-white-noise output — the UNet IS learning
- All outputs are **very loud** (RMS 0.6-0.7 vs real 0.02-0.15) because mel std=7 gets unnormalized to huge dB values
- All outputs have **bass 17-24%** (white noise would have ~50%) — there IS frequency shaping
- Noise class is worst (least structured training data)

## 3. Class Distinctiveness

**All classes produce COMPLETELY different outputs** (cosine similarity ≈ 0 between any two classes). The class embedding IS working — the UNet generates class-specific patterns.

## 4. Issues Summary

| Issue | Severity | Cause |
|-------|:---:|-------|
| Mel std = 7.2 (should be 1.0) | 🔴 Critical | UNet not converged / DDIM amplifies errors at high t |
| Values hit clamp at ±10 | 🔴 Critical | Same as above — output too large |
| Audio too loud (RMS 0.6) | 🟡 Medium | Consequence of mel std issue |
| 5/8 classes are hissy | 🟡 Medium | Mel with wrong std → HiFi-GAN gets wrong input range |
| Class conditioning works | 🟢 Good | Off-diagonal cos sim ≈ 0 |
| 3/8 classes show structure | 🟢 Good | Cat, Crow, Insect have non-Nyquist peaks |

## 5. Possible Fixes (NOT APPLIED)

| Fix | Expected impact |
|-----|----------------|
| **Tighten DDIM clamp: [-10,10] → [-4,4]** | Forces output to valid mel range. Quick fix. |
| **Continue training (50 → 150 epochs)** | UNet learns correct amplitude. Requires more Colab time. |
| **Reduce learning rate after epoch 50** | Fine-tune the converged model. |
| **Use EMA weights for inference** | Smooths out noise prediction errors. |
| **Increase batch size (reduce grad_accum)** | Better gradient estimates. |
| **Add L1 loss** | Penalizes large deviations more than MSE. |
