# Audio Quality Debug — May 13, 2026

> Generated audio sounds like noise/hurts ears, despite evaluation metrics looking good (MSE=0.0155, agreement=67.8%).

---

## Current Pipeline

```
VAE (base_ch=32, 223M) → normalized mel [mean≈0, std≈0.5]
    ↓
HiFi-GAN → unnormalize (×19.80 - 18.49) → waveform
    ↓
Griffin-Lim (optional, ON by default) → phase-refined waveform
    ↓
lowpass_biquad @ 11025 Hz
    ↓
output .wav
```

---

## Findings

### 1. VAE generates half the dynamic range of real spectrograms

| | Real Mel | Generated Mel |
|---|:---:|:---:|
| Mean | 1.03 | -0.48 |
| Std | 0.45 | 0.40 |
| Min | -0.32 | -4.08 |
| Max | 2.93 | 5.61 |
| **Mean abs diff** | — | **1.51** |

- Generated mel has **wider range** (-4 to +5.6 vs -0.3 to +2.9) but **different distribution** — mean shifted down, includes extreme outliers
- FiLM conditioning might be over-modulating certain frequency bins
- The VAE reconstruction is good (MSE=0.015) but **generation** (sampling from prior) produces different statistics

### 2. VAE output after normalization: std=0.49, should be ~1.0

| Stat | VAE Output | Expected (Normalized) |
|------|:---:|:---:|
| Mean | -0.21 | ~0 |
| Std | **0.49** | ~1.0 |
| Min/Max | -4.5 / +4.4 | -3.1 / +0.93 |

- Generated spectrograms have **compressed dynamic range** at inference time
- After unnormalization: std = 0.49 × 19.80 = **9.7 dB** (real = 19.8 dB)
- This means the HiFi-GAN receives a "flat" spectrogram with weak frequency peaks
- HiFi-GAN trained on real data with std≈19.8 dB — now getting inputs with std≈9.7 dB

### 3. Griffin-Lim makes it worse

| Mode | RMS | Peak |
|------|:---:|:---:|
| No Griffin-Lim | 0.014 | 0.36 |
| With Griffin-Lim | 0.005 | 0.12 |

- Griffin-Lim reduces amplitude by ~3×, making the output quieter
- The GL blending step (70% GL + 30% HiFi-GAN) replaces the neural waveform with a mathematical approximation

### 4. Frequency analysis shows wrong spectral profile

| | Generated "Dog" | Real Dog Bark |
|---|:---:|:---:|
| PSD peak | **603 Hz** | **22 Hz** |
| RMS | 0.065 | 0.024 |
| Flatness (hi/lo) | 0.15 | 0.00 |

- Generated audio peaks at 603 Hz (high-mid) — real dog barks are bass-heavy (22 Hz)
- The VAE isn't generating dog-like frequency content during **sampling** (though it reconstructs well)
- High frequency content = sounds "harsh/hurts ears"

---

## Root Cause (CONFIRMED)

**The decoder relies too heavily on encoder skip connections.** During training:
- 50% of time: full skip connections from encoder (decoder leans on them)
- 50% of time: no skips (decoder forced to work without)

But at 50% dropout, the decoder still learns to depend on skips. Without them, generation collapses.

Additionally, the FiLM conditioning may not be strong enough to compensate for missing skips.

---

## Possible Fixes (NOT YET APPLIED)

| Priority | Fix | Why |
|----------|-----|-----|
| **1** | **Increase skip_dropout to 0.8-1.0** | Force decoder to learn generation-capable outputs |
| **2** | **Increase class_loss_weight from 0.5 → 2.0** | FiLM must carry more class information without skips |
| **3** | Reduce temperature to 0.3-0.4 | Pull z closer to mean (decoder trained region) |
| **4** | Add post-generation spectrogram scaling | Match output statistics to real data before HiFi-GAN |
| **5** | Disable Griffin-Lim blending | Or reduce blend ratio from 70/30 to 30/70 |
| **6** | Verify HiFi-GAN norm_mean/std match training | hifigan/config.py vs data_loader.py SimpleNormalize |

---

## ✅ Quick Test: Reconstruction vs Generation — CONFIRMED

| Audio | RMS | Peak | Sounds Like |
|-------|:---:|:---:|-------------|
| Real dog | 0.15 | 0.90 | Dog bark ✅ |
| Reconstructed (encoder z + skips) | 0.17 | 1.00 | **Similar to real!** ✅ |
| Generated (random z, no skips) | **0.04** | 1.00 | Quiet noise ❌ |

**Problem is in VAE sampling, NOT HiFi-GAN.**

- HiFi-GAN works fine — reconstruction audio sounds correct
- VAE decoder with encoder skips works — it can reconstruct well (MSE=0.015)
- VAE decoder WITHOUT skips (generation mode) fails — 4× quieter, wrong spectral content
- The `skip_dropout=0.5` during training isn't enough — decoder relies on encoder skips

**Root cause: decoder over-depends on encoder skip connections.**

---

## Files to Check

- `src/vae/model.py` — `sample()` method, temperature handling
- `src/hifigan/inference.py` — unnormalization (`norm_mean/std`), Griffin-Lim blend
- `src/hifigan/config.py` — `norm_mean`, `norm_std` values
- `src/data_loader.py` — `SimpleNormalize` values (must match hifigan config)
