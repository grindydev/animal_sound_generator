# Workflow Fix Plan v13 — Root Cause: Time Conditioning Never Learned

> **Date:** May 16, 2026  
> **Status:** Root cause CONFIRMED.  
> **Builds on:** v12 (150 epochs trained, but all output is white noise/static).

---

## 0. The Definitive Root Cause (CONFIRMED)

### What I found

```
Model prediction from pure noise at ALL timesteps:
  t=   0: mean=0.1009, std=0.2861
  t= 100: mean=0.1008, std=0.2901
  t= 200: mean=0.1008, std=0.2871
  t= 400: mean=0.1008, std=0.2911
  t= 600: mean=0.1009, std=0.2933
  t= 800: mean=0.1008, std=0.2934
  t= 999: mean=0.1008, std=0.2914
```

**THE MODEL PREDICTS THE EXACT SAME OUTPUT FOR ALL TIMESTEPS AND ALL CLASSES.**

- Mean: 0.1008 ± 0.0003 (constant across all 7 classes, all 1000 timesteps)
- Std: 0.291 ± 0.005 (constant)
- The time embedding is **COMPLETELY IGNORED**
- Class differences: only 0.16-0.20 (tiny, barely above noise)

### Why this produces white noise

```
DDIM inference from pure noise:

Step 1 (t=999):
  x_t = randn (pure noise)
  pred_x0 = model(randn, t=999) = CONSTANT (mean=0.1008, std=0.291)
  noise_pred = (x_t - 0.006 * pred_x0) / 0.999 ≈ x_t ≈ randn
  x_t_next = 0.006 * pred_x0 + 0.999 * randn ≈ randn

Step 2 (t=994):
  x_t ≈ randn (barely changed)
  pred_x0 = model(randn, t=994) = SAME CONSTANT
  → same result

... 200 identical steps ...

Final output: randn (pure noise) → HiFi-GAN → WHITE NOISE / STATIC
```

The model's constant prediction is weighted by `sqrt(α_cumprod)` which is ~0.006 at t=999. It has **ZERO EFFECT** on the output. Every DDIM step just passes noise through.

### Why the model collapsed

**The model learned a shortcut:** Use the INPUT SIGNAL for predictions, ignore time and class conditioning.

During training:
```
Input: x_t = sqrt(α) * real_mel + sqrt(1-α) * noise
       ↑ This contains the REAL MEL (the signal)
       
Model: "I can see the real mel in the input! 
        I'll just extract it and output it.
        I don't need time conditioning or class conditioning."
```

The model never learned to use `t` or `class_label` because the input signal was sufficient. When given pure noise at inference (no signal), the model has nothing to use → predicts its training mean → constant output.

### Proof: training metrics were misleading

| Metric | What it showed | What it actually meant |
|--------|:---:|:---:|
| Val loss dropped | ✅ Model learning | Model learned to extract signal from input |
| Classifier guidance dropped (0.45→0.10) | ✅ Model learned class structure | Model used input signal, not class conditioning |
| Spectral balance stable | ✅ Realistic spectra | Realistic because input was real spectra |

**All training metrics measure RECOVERY quality (noisy→clean), not GENERATION capability (noise→mel).**

---

## 1. Why v1-v12 All Failed

| Version | What was tried | Why it failed |
|---------|:---:|:---:|
| v1-v3: VAE generation | Skip dropout, attention | VAE decoder needs encoder skips for content |
| v4-v6: DDIM/DDPM | Cosine/linear schedules, CFG | Model predicts zero noise → no denoising |
| v7-v8: x₀ prediction | Freq-weighted loss, augmentation | Model predicts mean x₀ → flat output |
| v9: Class balance | Oversample rare classes | Same recovery-only training |
| v10: ESC-50 data | Clean data, same arch | Clean data, but model still ignores conditioning |
| v11: More data sources | UrbanSound8K, Xeno-Canto | More data doesn't fix broken conditioning |
| v12: 3 new losses | Spectral balance, smoothness, classifier | Losses improve recovery, not generation |

**The pattern:** Every version fixes something in the TRAINING pipeline, but the fundamental issue is that the model never learned to USE its conditioning (time + class). All versions trained the model on: "given noisy real mel, recover the real mel." None trained: "given pure noise, CREATE a realistic mel using class label."

---

## 2. The Fix: Force Time/Class Conditioning

### Core insight

The model takes the shortcut of using the INPUT SIGNAL. We must REMOVE this shortcut.

### How: Train with pure noise at high timesteps

```python
# In train_epoch():
t = torch.randint(0, diffusion.timesteps, (B,))
noise = torch.randn_like(mel)

# V13 FIX: For t > 600, use PURE NOISE instead of noisy real mel
# This forces the model to use class conditioning (no input signal to cheat with)
use_pure_noise = (t > 600).float()  # 40% of training uses pure noise
x_t = use_pure_noise * noise + (1 - use_pure_noise) * diff.q_sample(mel, t, noise)

pred = model(x_t, t, labels)
loss = freq_weighted_loss(pred, mel)  # Always predict the real mel
```

**What this does:**
- t ≤ 600 (60%): Standard denoising task (noisy real mel → clean mel)
- t > 600 (40%): **Generation task** (pure noise → clean mel, conditioned on class label)

At high t, the model sees pure noise and MUST use the class label to predict the right mel. It can't cheat by extracting signal from the input.

### Why this works

```
Before (broken):
  t=999, input=noisy_real_mel → model extracts signal → predicts real mel
  t=999, input=pure_noise → no signal → model predicts mean

After (fixed):
  t=999, input=pure_noise + class="Dog" → model creates dog mel
  t=999, input=pure_noise + class="Cat" → model creates cat mel
  (No shortcut available — must use conditioning)
```

The model is forced to learn: "pure noise at t=999 + class label = create class-specific mel."

### Additional fixes

**1. Reduce model size further: 3.1M → 1.5M params**
- Less capacity to memorize, forced to learn general patterns
- `base_channels=24, channel_multipliers=(1,1,2,2)`

**2. Increase CFG scale: 1.5 → 2.5**
- Stronger class-specific signal amplification at inference

**3. Use cosine noise schedule**
- More signal at high timesteps, making the generation task easier
- `use_linear_schedule=False, cosine_s=0.008`

**4. Increase dropout: 0.3 → 0.4**
- More regularization, prevents memorization

**5. Add spectral stats matching at inference**
- Rescale generated mels before HiFi-GAN
- Already implemented in v12, keep it

---

## 3. Config Changes

```python
# config.py
base_channels: int = 24               # v13: ~1.5M params (was 32 → 3.1M)
channel_multipliers: tuple = (1, 1, 2, 2)  # [24, 24, 48, 48]
use_linear_schedule: bool = False     # v13: cosine schedule (was True)
cosine_s: float = 0.008              # cosine schedule smoothing
dropout: float = 0.4                  # v13: more dropout (was 0.3)
cfg_scale: float = 2.5               # v13: stronger CFG (was 1.5)

# New: pure noise training ratio
pure_noise_t_threshold: int = 600     # above this t, use pure noise
```

---

## 4. Training Changes

```python
# In train_epoch():
t = torch.randint(0, diffusion.timesteps, (B,), device=device)
noise = torch.randn_like(mel)

# V13: Pure noise generation task at high timesteps
x_t = diffusion.q_sample(mel, t, noise)
pure_noise_mask = (t > cfg.pure_noise_t_threshold).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
x_t = torch.where(pure_noise_mask.expand_as(x_t), noise, x_t)

# ... rest of training unchanged ...
```

This is a **ONE-LINE CHANGE** to the training loop that fundamentally changes what the model learns.

---

## 5. Expected Outcome

| Metric | v12 (current) | v13 target |
|--------|:---:|:---:|
| Model params | 3.1M | 1.5M |
| Timestep conditioning | IGNORED | ACTIVE |
| Class distinction at t=500 | 0.16-0.20 | >0.50 |
| Prediction varies by t | NO | YES |
| Generated audio | White noise | Animal-like |
| Recognizable classes | 0/7 | 4-5/7 |

---

## 6. Why This is Different from All Previous Attempts

| Previous | Problem | v13 Difference |
|----------|:-------:|:---:|
| v4-v6: DDPM/DDIM | Model predicted zero noise | Model MUST predict class-specific output (no signal to extract) |
| v7-v8: x₀ prediction | Model predicted mean x₀ | Model learns: noise + class → specific mel |
| v12: 3 new losses | Improved recovery, not generation | Forces generation training (pure noise at high t) |

**Previous versions:** Model always had the input signal as a shortcut.  
**v13:** Signal is removed at high t → model MUST use conditioning.

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|:----------:|------------|
| Pure noise task too hard for 640 samples | Medium | Pure noise is only 40% of training; rest is standard denoising |
| Model still collapses to mean | Low | 1.5M params can't memorize; cosine schedule helps |
| Training unstable | Low | Same optimizer, same LR; just different input distribution |
| CFG amplifies artifacts | Medium | Start with scale=2.0, increase to 2.5 if needed |

---

## 8. Execution Plan

```
Step 1: Update config.py (smaller model, cosine schedule, pure_noise_threshold)
Step 2: Update train.py (one-line: pure noise at high t)
Step 3: Delete old checkpoints
Step 4: Train on Colab (150 epochs, ~2 hrs)
Step 5: Verify timestep conditioning is active (diagnostic test)
Step 6: Generate audio and evaluate
```

---

*Root cause confirmed with diagnostic tests. The model's time embedding is completely unused — it predicts the same constant output for all timesteps and classes. The fix: remove the input signal at high timesteps, forcing the model to use its conditioning.*
