# Workflow Fix Plan v8 — Frequency-Weighted Loss + Strong Augmentation

> **Date:** May 15, 2026  
> **Status:** Implementing.  
> **Builds on:** v7 (x₀-prediction — produces audio but sounds like low hum)  
> **Approach:** Force model to learn high-frequency detail from existing 2700 samples.

---

## 1. The Problem

```
Real dog bark mel:    Low freq ▁▂▃▄▅   +   High freq ▂▁▃▂▄ (harmonics, texture)
Model output (v7):    Low freq ▁▂▃▄▅   +   High freq ▁▁▁▁▁ (nothing)
                                                        ↑
                                              Model never punished for this
```

L1 loss treats all 64 mel bins equally. The low bins have 10× more energy, so they dominate the gradient. Model learns "get low freq right, ignore high freq." Output = low hum.

## 2. Fix A: Frequency-Weighted Loss

```python
# Current:
loss = L1(pred_mel, real_mel)           # all 64 bins equal weight

# v8:
freq_weights = linspace(1.0, 3.0, 64)   # bin 0→1.0, bin 63→3.0
loss = (abs(pred_mel - real_mel) * freq_weights).mean()
```

3× penalty on highest frequencies. Model MUST learn harmonics to survive.

## 3. Fix B: Stronger Augmentation

Current augmentation (v7): `±4 bin shift + 20% time stretch` → too weak.

v8 augmentation (applied per-sample, random):
- Frequency masking: zero out 2-8 consecutive mel bins
- Time masking: zero out 20-80 consecutive frames  
- Gaussian noise: add σ=0.05 to mel
- Gain jitter: multiply mel by [0.8, 1.2]
- Bin dropout: randomly zero 10% of all bins

Result: model never sees the same mel twice → must learn actual structure, not memorize.

## 4. Changes

| File | Change |
|------|--------|
| `src/diffusion/config.py` | Add freq_weight_max, strong_augment params |
| `src/diffusion/train.py` | Freq-weighted L1 loss, stronger augmentation function |

## 5. Success Criteria

| Metric | v7 | v8 Target |
|--------|:---:|:---:|
| Generated mel L/H ratio | 1.3 | 0.7-1.0 (more high-freq) |
| Audio peak frequency | 139-292 Hz | 500-3000 Hz |
| Audio ZCR | 0.33 | 0.15-0.40 (animal-like) |
| Audio σ | 0.01-0.10 | 0.10-0.30 |

## 6. Backup: Fix C — Discriminator (GAN Loss)

If A+B still produces "low hum," the model needs per-pixel feedback.
A PatchGAN discriminator classifies "real mel" vs "fake mel" on 32×32 patches.

```python
# 5 conv layers, ~2M params
class MelDiscriminator(nn.Module):
    def forward(self, mel):  # [B,1,64,552]
        return patch_logits  # [B,1,2,34] — real/fake score per patch

# Training:
#   UNet loss = L1(x0, real) * freq_weights + 0.1 * BCE(disc(fake), "real")
#   Disc loss = BCE(disc(real), "real") + BCE(disc(fake), "fake")
```

| Outcome | A+B only | A+B+C (GAN) |
|--------|:---:|:---:|
| High-freq detail | Some | Sharp |
| Audible animal | Maybe | Yes |
| Training time | 2h | 3h |
| Code change | 25 lines | 90 lines |
