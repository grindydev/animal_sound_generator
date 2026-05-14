# Workflow Fix Plan v6 — The Real Problem: DDIM Amplification

> **Date:** May 14, 2026  
> **Status:** Root cause found. DDIM mathematically can't work at high t with α≈0.00004.

---

## 0. What We Know

```
Model          Epoch  Val    Sampling   Mel σ   OK      Note
─────────────────────────────────────────────────────────────
v4 cosine      6      0.60   DDIM 50    7.2     3/8     3× too loud
v4 cosine      36     0.09   DDIM 50    2.5     1/8     overfit
v5 linear      10     0.60   DDIM 100   1.2     2/8     ✅ best σ
v5 linear      25     0.19   DDIM 100   0.8     1/8     too quiet
v5 linear      50     0.05?  DDIM 100   0.6?    0/8?    predicted
```

The clamp fixed mel σ (7.2→1.2). But generation still degrades as val loss drops.  
**Why?** It's not overfitting. It's math.

---

## 1. The Real Problem: DDIM x₀ Explosion

Here's what happens at DDIM step 1 (t=999):

```
x₀_pred = (x_t - √(1-α) · pred_noise) / √α
        = (x_t - 0.9999 · pred_noise) / 0.0063    ← √α = 0.0063
```

If the model predicts noise 99.9% correctly (error = 0.01):

```
x₀ error = 0.9999 × 0.01 / 0.0063 = 1.6
```

A **0.01 noise error → 1.6 spectrogram error**. That's 160× amplification.

Then the clamp at [-4,4] activates, but by that point the spectrogram structure is destroyed. All values jump to ±4 — a flat checkerboard. The remaining 99 DDIM steps can't recover detail from a flat grid.

### Even 99.99% accuracy wouldn't help

At error = 0.001:
```
x₀ error = 0.16  ← still significant on [-4,4] range
```

The DDIM chain requires the model to predict x₀ from t=999. But t=999 is 99.996% noise. Getting x₀ right from 0.004% signal is like identifying a person from a single pixel of their photo.

---

## 2. Why DDPM Doesn't Have This Problem

DDPM never predicts x₀. It only takes small local steps:

```python
# DDPM p_sample (one step):
posterior_mean = (x_t - βₜ/√(1-ᾱₜ) · pred_noise) / √αₜ
              = (x_t - 0.02/1.0 · pred_noise) / 0.99    ← no division by tiny number
```

If pred_noise error = 0.01:
```
posterior_mean error = 0.02 × 0.01 / 0.99 = 0.0002
```

That's **8,000× less** amplification than DDIM (0.0002 vs 1.6).

### But we couldn't test DDPM on MPS

DDPM requires 1000 steps. On MPS, `p_sample_loop` runs but the mel σ exploded to 179 — likely a numerical issue specific to MPS. DDPM needs CUDA (Colab).

---

## 3. So Did v5 Really Fail?

**No.** We haven't actually tested v5 properly. DDIM can't evaluate the model — it mathematically destroys any output from high t. The model might generate great spectrograms, but DDIM's first step flattens them to ±4.

The proper test is **DDPM sampling on CUDA**. Only that can tell us if the model learned anything.

---

## 4. Fix Plan

### A. Test DDPM on Colab (IMMEDIATE)

```bash
# In Colab, after training:
python src/generate.py --label Dog --from-scratch --ddpm

# Test all classes:
for label in Dog Cat Rooster Frog Crow Insect Hen Noise; do
    python src/generate.py --label $label --from-scratch --ddpm
done
```

This tells us if the model actually learned vs DDIM was just breaking it.

### B. Training Changes (v6)

| Change | Why |
|--------|-----|
| **base_channels: 96 → 48** | 15M params (vs 61M). 5,600 params/sample. Forces generalization. |
| **multipliers: (1,2,3,4) → (1,2,2,4)** | Cuts channels at middle levels where model wastes capacity. |
| **res_blocks_per_level: 2 → 1** | Fewer blocks = less memorization capacity. |
| **attention_levels: (2,3) → (3,)** | Only deepest level gets attention. Focus capacity where it matters. |
| **loss_type: "l2" → "l1"** | L1 penalizes mean regression less. Optimal L1 predictor = median, not mean. |
| **dropout: 0.1 → 0.2** | More dropout = less memorization. |
| **add SpecAugment** | Frequency masking + time masking during training. |
| **uncond_prob: 0.15** | 15% of training batches get `label=None`. Enables classifier-free guidance. |

### C. Inference Changes (v6)

| Change | Why |
|--------|-----|
| **Default: DDPM** | Avoid DDIM x₀ explosion entirely. |
| **DDIM start from t=500** | Skip first 500 noise steps. √α ≈ 0.28 at t=500 → 22× less amplification. |
| **CFG scale: 2.0** | `pred = uncond + 2.0*(cond - uncond)` → sharper output. |
| **Save ALL checkpoints** | Evaluate each epoch. Pick best, not last. |

### D. Long-Term Options

| Option | Effort | Expected Quality |
|--------|:------:|:----------------:|
| Collect more data (5000+ samples) | High | **** |
| Latent diffusion (diffuse in VAE latent space) | Medium | *** |
| GAN (small datasets work better with GANs) | Medium | *** |
| Transfer learning from pretrained audio model | Medium | **** |

---

## 5. v6 Config Proposal

```python
# src/diffusion/config.py — v6
base_channels: int = 48                     # was 96
channel_multipliers: tuple = (1, 2, 2, 4)   # was (1, 2, 3, 4)
res_blocks_per_level: int = 1               # was 2
attention_levels: tuple = (3,)              # was (2, 3)
dropout: float = 0.2                        # was 0.1
loss_type: str = "l1"                       # was "l2"
uncond_prob: float = 0.15                   # NEW: classifier-free guidance
# Expected: ~15M params (4× less than current 61M)
```

### v6 architecture per level:

```
Level 0: 48 ch, 1 ResBlock, no attn, 64×552
Level 1: 96 ch, 1 ResBlock, no attn, 32×276
Level 2: 96 ch, 1 ResBlock, no attn, 16×138
Level 3: 192 ch, 1 ResBlock, self-attn, 8×69
Bottleneck: 192 ch, 1 ResBlock, self-attn, 8×69
Decoder: reverse
```

---

## 6. Implementation Order

```
1. Test DDPM on current v5 model in Colab   ← tells us if v5 actually works
2. Apply v6 config changes                   ← reduce overfitting
3. Add SpecAugment to data pipeline          ← prevent memorization
4. Add unconditional training + CFG          ← sharpen output
5. Save all checkpoints, evaluate each       ← pick best epoch
6. Default to DDPM for generation            ← avoid DDIM math
```

---

## 7. Success Criteria

| Metric | Current (DDIM) | Target (DDPM) |
|--------|:---:|:---:|
| Mel σ | 0.8-1.2 | 0.8-1.5 |
| Classes with structure | 1-2/8 | 4+/8 |
| Audio peak <5000Hz | 1/8 | 4+/8 |
| Classes distinct by ear | 0/8 | 3+/8 |
