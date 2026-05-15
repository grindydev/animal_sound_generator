# Workflow Fix Plan v7 — x₀-Prediction Diffusion

> **Date:** May 15, 2026  
> **Status:** Implementing.  
> **Approach:** Industry-standard x₀-prediction. Model predicts clean mel, not noise.  
> **Why:** ε-prediction (v4-v6) target is random numbers → model learns to predict 0. x₀-prediction target is real spectrograms → model learns structure.

---

## 1. The One Change That Matters

```
ε-prediction (v4-v6, FAILED):
  x_t = √ᾱ·x₀ + √(1-ᾱ)·ε    →    model(x_t) → ε̂    →    L(ε̂, ε)
  Target = random noise. Model learns: predict 0. Output = silence/noise.

x₀-prediction (v7):
  x_t = √ᾱ·x₀ + √(1-ᾱ)·ε    →    model(x_t) → x̂₀    →    L(x̂₀, x₀)
  Target = real spectrogram. Model learns: harmonics, formants, rhythm.
```

## 2. Architecture (unchanged)

Same UNet, same diffusion, same HiFi-GAN. Only the training target and inference math change.

## 3. Training

```python
# One line changes:
#   OLD: loss = MSE(pred_noise, noise)
#   NEW: loss = MSE(pred_x0, mel)
```

Plus augmentation: pitch shift ±3 semitones, time stretch 0.8-1.2×.

## 4. Inference

DDIM adapted for x₀-prediction:
```python
for t in timesteps:
    pred_x0 = model(x_t, t, labels)
    pred_x0 = clamp(pred_x0, -4, 4)
    # Recover noise from x₀ prediction
    noise_pred = (x_t - √ᾱₜ · pred_x0) / √(1-ᾱₜ)
    # DDIM step using x₀
    direction = √(1-ᾱₙₑₓₜ) · noise_pred
    x_t = √ᾱₙₑₓₜ · pred_x0 + direction
```

## 5. Files Changed

| File | Change |
|------|--------|
| `src/diffusion/config.py` | Add augment params |
| `src/diffusion/train.py` | x₀ loss, pitch/time augment, 95/5 split |
| `src/diffusion/diffusion.py` | x₀-prediction DDIM/DDPM sampling |
| `src/diffusion/inference.py` | x₀ generate_from_noise |
| `src/generate.py` | --x0 flag, updated pipeline |
| `colab/colab_train.ipynb` | Updated training cell |

## 6. Success Criteria

| Metric | Current (ε) | Target (x₀) |
|--------|:---:|:---:|
| Val loss decreasing | ❌ Flat at 1.0 | ✅ Decreasing |
| Generated mel σ | N/A (noise) | 0.8-1.5 |
| Audio peak | 11025 Hz | <5000 Hz |
| Classes distinct | 0/8 | 3+/8 |
