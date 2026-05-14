# Workflow Fix Plan v5 — Diffusion Sampling & Training Fixes

> **Date:** May 14, 2026  
> **Status:** Review complete. Ready for implementation.  
> **Context:** Epoch 36 (val=0.09) produces worse generation than epoch 6 (val=0.60).  
> Lower val loss ≠ better generation. Model overfits, DDIM amplifies errors.

---

## 0. Full File Review

Files reviewed and their status:

| File | Role | Status |
|------|------|--------|
| `src/diffusion/diffusion.py` | DDIM/DDPM sampling, noise schedule | 🔴 Issues found |
| `src/diffusion/train.py` | Training loop, data loading, EMA | ✅ Solid |
| `src/diffusion/config.py` | Model/diffusion config | ✅ Good |
| `src/diffusion/unet.py` | UNet architecture | ✅ Good |
| `src/diffusion/inference.py` | generate_from_noise() API | ✅ Solid |
| `src/generate.py` | CLI | ✅ Solid |
| `src/data_loader.py` | Normalization | ✅ Verified match |
| `src/hifigan/config.py` | HiFi-GAN normalization | ✅ Verified match |
| `src/hifigan/inference.py` | mel_to_waveform | ✅ Verified |

---

## 1. Issues Found

### Issue 1: DDIM amplifies prediction errors at high timesteps 🔴

```
At t=999 (first DDIM step):
  α_cumprod ≈ 0.0006  →  √α ≈ 0.0245
  x_0_pred = (x_t - √(1-α) · ε̂) / √α
           = (x_t - 0.9997 · ε̂) / 0.0245
```

If ε̂ deviates from true ε by just 0.01:
```
x_0 error = 0.9997 × 0.01 / 0.0245 = 0.41
```
A 0.01 noise error → 0.41 in x_0 output. The error gets **amplified 40×**.

This error compounds over 50 DDIM steps because:
- Each step's x_0_pred feeds into the next step's x_t
- Out-of-range values at step 1 corrupt all subsequent steps
- The clamp at [-10, 10] doesn't help — it just ceilings at ±10

**Training data range:** [-3, +4] (from data validation output).  
**DDIM clamp:** [-10, 10] — allows values 2.5× outside training range.

### Issue 2: Uniform timestep training mismatches DDIM inference 🔴

```
Training:    t ~ Uniform(0, 999) — all timesteps equally
Inference:   t = {999, 979, 958, ...} — 50 specific steps
```

The UNet was trained equally on all timesteps but DDIM only uses 50 specific ones. Some of these may be poorly sampled during training (high-t steps are rare: only ~0.1% of training batches see t near 999).

**Cosine schedule makes high-t worse:** α_cumprod at t=999 is ~0.0006 (cosine) vs ~0.004 (linear β_start=0.0001). Cosine has 6.7× LESS signal at high t → harder to predict x_0 accurately.

### Issue 3: Model overfitting (61M params on 2700 samples) 🟡

```
2700 train samples / 61M params = 22,600 params per sample
```

Each training sample must constrain ~22,600 parameters. The model has enough capacity to memorize individual noise patterns. At epoch 36 (val=0.09), it's learning the average noise prediction, not the distribution. This explains why generation quality degrades while val loss drops.

**Evidence:** Epoch 6 (val=0.60) had 3/8 good classes. Epoch 36 (val=0.09) had 1/8 good classes.

### Issue 4: Clamp range too wide 🟡

```
DDIM clamp:       [-10, 10]  
Training range:   [-3, 4]     (from first batch data validation)
HiFi-GAN expects: [-3, 4]     (normalized dB mel)
```

Clamp at [-10, 10] lets values pass that are 2.5× outside what HiFi-GAN expects. This causes the generated audio to be too loud (RMS=0.5-0.8 vs real=0.02-0.15).

### Issue 5: No DDPM option in generate.py 🟡

Current `generate.py --from-scratch` always uses DDIM (deterministic). DDPM (stochastic, step-by-step) would add randomness at each step, which helps prevent mean-regression. The `diffusion.py` has `p_sample_loop()` for full DDPM but nothing calls it.

---

## 2. Fix Plan

### Fix 1: Add DDPM Sampling Option 🔴 HIGH

**File:** `src/diffusion/inference.py`

Add `use_ddpm` parameter to `generate_from_noise()`:

```python
@torch.no_grad()
def generate_from_noise(
    label_idx=0, num_samples=1, num_steps=50,
    device=None, spec_shape=(64, 552),
    use_ddpm=False,  # NEW: use full DDPM instead of DDIM
):
    model, diffusion = get_diffusion_model(device)
    labels = torch.full((num_samples,), label_idx, device=device, dtype=torch.long)
    x_t = torch.randn(num_samples, cfg.spec_channels, *spec_shape, device=device)
    
    if use_ddpm:
        generated = diffusion.p_sample_loop(model, x_t.shape, labels, device)
    else:
        generated = diffusion.ddim_sample(model, x_t, labels, num_steps=num_steps)
    
    return generated.cpu()
```

**File:** `src/generate.py` — add `--ddpm` flag:
```python
parser.add_argument("--ddpm", action="store_true",
                    help="Use full DDPM sampling (slower but more diverse)")
```

**Why:** DDPM adds noise at each step, preventing the model from collapsing to the mean. The stochasticity introduces the diversity that DDIM's deterministic path lacks.

### Fix 2: Tighten DDIM Clamp 🔴 HIGH

**File:** `src/diffusion/diffusion.py`

Change both clamp locations from [-10, 10] → [-4, 4]:

```python
x_0_pred = torch.clamp(x_0_pred, -4.0, 4.0)
```

**Why:** Matches training data range. Prevents out-of-range values from accumulating across steps. Catches the 40× error amplification at high t.

### Fix 3: Increase DDIM Steps 🟡 MEDIUM

**File:** `src/diffusion/inference.py` — default `num_steps` from 50 → 100:

```python
def generate_from_noise(..., num_steps=100, ...):
```

**Why:** More steps = smaller jumps between timesteps. At 100 steps, each jump is half the size, reducing the error amplification by ~2×. Trade-off: 2× slower generation.

### Fix 4: Linear Noise Schedule 🟡 MEDIUM

**File:** `src/diffusion/config.py`:

```python
beta_start: float = 0.0001
beta_end: float = 0.02  
# Change from cosine_s=0.008 → use linear schedule
# In diffusion.py:
betas = torch.linspace(beta_start, beta_end, timesteps)  # instead of cosine
```

**Or:** Keep cosine but increase `cosine_s` from 0.008 → 0.02 (more signal at high t).

**Why:** Cosine schedule has less signal at high t than linear. This makes DDIM's first steps much harder. Linear gives ~4× more signal at t=999 (α_cumprod ~0.0025 vs ~0.0006), reducing the error amplification from 40× to 20×.

### Fix 5: Model Size Reduction 🟢 LOW

**File:** `src/diffusion/config.py`:

```python
base_channels: int = 64               # was 96
channel_multipliers: tuple = (1, 2, 3, 4)  # keep
```

**Why:** 30M params is a better match for 2700 samples (~9,000 params per sample). Easier to train, harder to overfit. But this requires retraining from scratch — only do if other fixes don't help.

---

## 3. Implementation Order

```
Step 1: Fix 2 (tighten clamp) + Fix 3 (100 DDIM steps)
        → 5-minute change, test immediately with current model

Step 2: Fix 1 (DDPM sampling option)
        → Test DDPM vs DDIM on current model

Step 3: If still noise: Fix 4 (linear schedule)
        → Requires retraining (3-4 hrs)

Step 4: If still noise: Fix 5 (smaller model)
        → Requires retraining from scratch
```

## 4. Quick Test After Each Fix

```bash
# Test with tightened clamp + more steps
python src/generate.py --label Dog --from-scratch --diffusion-steps 100

# Test with DDPM
python src/generate.py --label Dog --from-scratch --diffusion-steps 100 --ddpm

# Check mel statistics
python -c "
from diffusion.inference import generate_from_noise
mel = generate_from_noise(0, 1, num_steps=100)
print(f'mel σ={mel.std():.2f} (target: ~1.0)')
"
```

---

## 5. Success Criteria

| Test | Current (Epoch 36) | Target |
|------|:---:|:---:|
| Mel σ | 2.5 | **0.8-1.5** |
| Classes with structure | 1/8 | **5+/8** |
| Audio peak freq | Mostly 11025 Hz | **<5000 Hz for 5+ classes** |
| Audio RMS | 0.5-0.8 | **0.05-0.3** |
