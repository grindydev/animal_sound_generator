# Workflow Fix Plan v12 — Kill the "Electric" Sound

> **Date:** May 16, 2026  
> **Status:** Proposed.  
> **Builds on:** v11 (ESC-50 data — 4/8 audible, but all sound electric/buzzy).  
> **Root cause:** Spectral imbalance + erratic temporal dynamics + posterior collapse.

---

## 0. The Problem in Numbers

| Metric | Real Dog Bark | Generated (v10) | Gap |
|--------|:---:|:---:|:---:|
| Low band (0-16) energy | -39.4 dB | **-30.4 dB** | +9 dB too loud |
| Mid band (16-32) energy | -37.6 dB | **-47.3 dB** | -10 dB too quiet |
| High band (32-48) energy | -38.5 dB | **-57.6 dB** | -19 dB too quiet |
| **Spectral slope** | **~1 dB** | **27 dB** | **27× worse** |
| Temporal energy std | 8.82 dB | **18.22 dB** | 2× jitter |
| Mel std | 0.48 | **1.43** | 3× erratic |
| Spectral flatness | 0.27 | **0.80** | noise-like |

**Real dog**: Flat energy across frequencies (broadband bark).  
**Generated**: All energy crammed into low bins → low hum → HiFi-GAN converts flat mel to electric buzz.

The "electric" sound is **not** a single bug. It's three bugs stacked:
1. Low-frequency energy concentration → **hum**
2. Erratic temporal variation → **buzz**
3. Extreme mel outliers → **crackle**

---

## 1. Root Cause Analysis

### 1.1 Why spectral slope is 27 dB (should be ~1 dB)

```
Current loss: L1(pred, real) × freq_weights(1.0 → 3.0)

Problem: freq_weights only go to 3×. But the low bins have 100× more
gradient magnitude because they have 100× more energy. The 3× penalty
is drowned out.

Real gradient on bin 0:  100.0 × 1.0 = 100.0
Real gradient on bin 63:   1.0 × 3.0 = 3.0

Model optimizes bin 0 because it's 33× more impactful.
```

### 1.2 Why temporal std is 18 dB (should be ~9 dB)

```
The model predicts x₀ independently for each (t, class) combination.
At high t (near pure noise), x₀ predictions are erratic.
DDIM chains 100 of these erratic predictions together.
Result: each frame looks different → wild temporal variation → buzzing.
```

### 1.3 Why mel std is 1.43 (should be ~0.5)

```
Model predicts the mean of ALL training spectrograms (posterior collapse).
Then adds small noise on top → outliers.
No mechanism to constrain output to realistic statistics.
```

---

## 2. Fix A: Spectral Balance Loss (KILLS the hum)

### The Insight

Instead of weighting individual bins, **penimize the spectral shape directly**.
Compare the energy distribution of predicted vs real spectrograms and punish deviation.

```python
def spectral_balance_loss(pred, real):
    """
    Penalize unrealistic frequency distribution.
    
    Split mel into bands and compare energy ratios.
    """
    # Split into 4 bands
    bands = [pred[..., :16, :], pred[..., 16:32, :], 
             pred[..., 32:48, :], pred[..., 48:, :]]
    
    # Compute band energies
    pred_energy = torch.stack([b.abs().mean() for b in bands])
    real_energy = torch.stack([b.abs().mean() for b in bands])
    
    # Normalize to ratios (invariant to overall volume)
    pred_ratio = pred_energy / pred_energy.sum()
    real_ratio = real_energy / real_energy.sum()
    
    # Penalize deviation from real distribution
    return F.mse_loss(pred_ratio, real_ratio) * 10.0
```

**Why this works**: Instead of fighting per-pixel gradients, it directly targets the *shape* of the spectrum. If the model puts all energy in low bands, the loss is huge regardless of absolute values.

### Apply to training

```python
# In train_epoch():
l1 = freq_weighted_loss(pred, mel)
balance = spectral_balance_loss(pred, mel)
loss = l1 + balance
```

**Expected effect**: Generated mel spectral slope drops from 27 dB → <10 dB.

---

## 3. Fix B: Temporal Smoothness Loss (KILLS the buzz)

### The Insight

Animal sounds have smooth temporal envelopes — a bark has a clear attack, sustain, decay. 
The model produces frame-by-frame jitter because each x₀ prediction is independent.

```python
def temporal_smoothness_loss(pred):
    """
    Penalize erratic frame-to-frame changes in energy.
    Real animal sounds have smooth envelopes, not jitter.
    """
    # pred: [B, 1, F, T]
    # Compute frame energy: [B, 1, T]
    frame_energy = pred.abs().mean(dim=2)
    
    # First difference (frame-to-frame change)
    diff = frame_energy[..., 1:] - frame_energy[..., :-1]
    
    # Penalize large jumps
    return (diff ** 2).mean() * 5.0
```

**Why this works**: Forces the model to produce temporally coherent spectrograms. A dog bark has energy that rises and falls smoothly — not jumps randomly.

### Apply to training

```python
# In train_epoch():
l1 = freq_weighted_loss(pred, mel)
balance = spectral_balance_loss(pred, mel)
smooth = temporal_smoothness_loss(pred)
loss = l1 + balance + smooth
```

**Expected effect**: Temporal energy std drops from 18 dB → <12 dB.

---

## 4. Fix C: Classifier-Guided Training (ENSURES animal-like structure)

### The Insight

We have a 95% accurate audio classifier (`best_audio_cnn_train.pth`). 
Use it as a perceptual loss: if the generated mel doesn't look like the target animal, penalize it.

```python
def classifier_guidance_loss(pred, labels, classifier):
    """
    Force predicted x₀ to be classified as the correct animal class.
    
    This ensures the model produces animal-like structure, not just
    statistically similar spectrograms.
    """
    # classifier expects [B, 1, F, T] → class logits
    logits = classifier(pred)  # [B, 8]
    return F.cross_entropy(logits, labels) * 0.5
```

**Why this works**: The classifier was trained on REAL animal sounds. If it can't recognize the generated mel as a dog bark, the mel doesn't have dog-like structure. This is a much stronger signal than L1 loss on individual pixels.

### Implementation

```python
# Load classifier at training start
classifier = SimpleAudioCNN(num_classes=8)
classifier.load_state_dict(
    torch.load("models/best_audio_cnn_train.pth", map_location=device)["model_state_dict"]
)
classifier.eval()
for p in classifier.parameters():
    p.requires_grad_(False)  # freeze

# In train_epoch():
l1 = freq_weighted_loss(pred, mel)
balance = spectral_balance_loss(pred, mel)
smooth = temporal_smoothness_loss(pred)
cls_loss = classifier_guidance_loss(pred, labels, classifier)
loss = l1 + balance + smooth + cls_loss
```

**Expected effect**: Classifier agreement jumps from ~40% → 70%+. Generated mels have recognizable animal structure.

---

## 5. Fix D: Spectral Stats Matching (KILLS the crackle)

### The Insight

Generated mels have wildly different statistics than real mels. Rescale before HiFi-GAN.

```python
def match_spectral_stats(mel, target_mean=0.0, target_std=0.5):
    """
    Rescale mel to match real data statistics.
    Real mels: mean≈0, std≈0.5 (normalized)
    Generated: mean≈-1.5, std≈1.4
    """
    # Per-sample normalization
    mel_mean = mel.mean(dim=(1, 2, 3), keepdim=True)
    mel_std = mel.std(dim=(1, 2, 3), keepdim=True)
    
    # Rescale
    mel = (mel - mel_mean) / (mel_std + 1e-8) * target_std + target_mean
    
    # Clip extreme outliers
    return torch.clamp(mel, -2.0, 2.0)
```

### Where to apply

```python
# In inference.py, generate_from_noise():
generated = diffusion.ddim_sample_x0(...)
generated = match_spectral_stats(generated, target_mean=0.0, target_std=0.5)
```

**Expected effect**: Mel std drops from 1.43 → 0.5. HiFi-GAN receives proper input → no crackle.

---

## 6. Fix E: Reduce Model Size (PREVENTS overfitting)

### Current vs Target

| | Current | Target |
|---|:---:|:---:|
| base_channels | 64 | **32** |
| channel_multipliers | (1, 2, 2, 4) | **(1, 1, 2, 2)** |
| res_blocks_per_level | 1 | **1** |
| attention_levels | (3,) | **(3,)** |
| time_emb_dim | 512 | **256** |
| class_emb_dim | 256 | **128** |
| dropout | 0.2 | **0.3** |
| **Total params** | **~18M** | **~4M** |

**Why this works**: 4M params / 640 samples = 6,250 params/sample. Still high, but much more manageable than 28,000 params/sample. Smaller model = forced to generalize, not memorize.

### Config changes

```python
# In config.py:
base_channels: int = 32               # was 64
channel_multipliers: tuple = (1, 1, 2, 2)  # was (1, 2, 2, 4)
time_emb_dim: int = 256               # was 512
class_emb_dim: int = 128              # was 256
dropout: float = 0.3                  # was 0.2
```

---

## 7. Fix F: Better Checkpoint Selection

### The Problem

From `diffusion-evaluation.md`:
> "The best generation quality happens at medium val loss (0.60), not low val loss (0.09-0.19). As loss drops, the model converges to predicting the mean noise → flat outputs."

### Solution

Save checkpoints every 3 epochs. Use a **generation-based metric** instead of val loss for checkpoint selection.

```python
# In training_loop():
# Save checkpoints periodically
if (epoch + 1) % 3 == 0:
    torch.save({"unet": ema_model.state_dict()}, 
               f"models/diffusion_checkpoints/epoch_{epoch+1}.pth")

# After training, evaluate each checkpoint for generation quality
# Pick the one with best spectral balance (not lowest val loss)
```

**Or simpler**: Use epoch ~15-20 for 150-epoch training. The model has seen enough data to learn patterns but hasn't collapsed yet.

---

## 8. Fix G: DDIM Sampling Improvements

### The Problem

DDIM at high t values has extreme error amplification (160× at t=999).

### Solution 1: Start from lower noise

```python
# In inference.py, generate_from_noise():
# Instead of starting from pure noise, start from t=500
# This skips the most unstable high-t steps
start_noise = torch.randn(shape) * 0.5  # half-strength noise
```

### Solution 2: Use refinement mode (img2img)

```python
# Generate a "seed" spectrogram from statistical priors
# Then refine with diffusion
seed_mel = generate_statistical_mel(label_idx)
refined = diffusion.refine(model, seed_mel, labels, strength=0.4, num_steps=50)
```

### Solution 3: Increase DDIM steps

```python
# config.py:
inference_steps: int = 200  # was 100
```

More steps = smaller per-step error = less accumulation.

---

## 9. Fix H: Remove Noise Class

### Why

The Noise class (280 files of rain, wind, waves) teaches the model to generate flat, noise-like spectrograms. This directly contributes to the "electric" sound.

```python
# config.py:
num_classes: int = 7  # was 8
CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']
```

All 7 animal classes get equal attention. No more "rain" class polluting the model.

---

## 10. Summary of Changes

| File | Change | Purpose |
|------|--------|---------|
| `src/diffusion/config.py` | Reduce model size, remove Noise class, more inference steps | Prevent overfitting, cleaner training |
| `src/diffusion/train.py` | Add spectral balance loss, temporal smoothness, classifier guidance | Force realistic spectrograms |
| `src/diffusion/inference.py` | Add spectral stats matching | Fix HiFi-GAN input |
| `src/diffusion/unet.py` | Smaller architecture (base_ch=32, etc.) | Prevent memorization |

---

## 11. Training Order

```
Step 1: Update config.py (smaller model, 7 classes)
Step 2: Add loss functions to train.py
Step 3: Add stats matching to inference.py
Step 4: Delete old checkpoints
Step 5: Train on Colab (150 epochs, ~2 hrs)
Step 6: Test generation → check spectral balance, temporal smoothness
Step 7: If still electric → increase loss weights
```

---

## 12. Success Criteria

| Metric | Current (v10) | v12 Target |
|--------|:---:|:---:|
| Spectral slope (low vs high) | 27 dB | <10 dB |
| Temporal energy std | 18 dB | <12 dB |
| Mel std | 1.43 | <0.7 |
| Spectral flatness | 0.80 | <0.5 |
| Classifier agreement | ~40% | >60% |
| Recognizable animal sounds | 4/8 | 6/8 |
| Audio doesn't sound electric | ❌ | ✅ |
| Peak frequency per class | 33-202 Hz | Class-specific |

---

## 13. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|:----------:|------------|
| Classifier loss too strong → mode collapse | Medium | Start with low weight (0.1), ramp up |
| Smaller model can't learn enough | Low | 4M params is still significant |
| Training slower due to classifier | Low | Classifier is frozen, forward-only |
| Spectral balance conflicts with L1 | Low | L1 handles reconstruction, balance handles shape |

---

## 14. Why This Will Work (vs Previous Attempts)

| Attempt | Why It Failed | Why v12 Is Different |
|---------|:-------------:|:--------------------:|
| v4-v6: Pure diffusion | Model predicted zero noise | **Classifier loss** forces structure, not zero |
| v7-v8: x₀ prediction | Still flat spectrograms | **Spectral balance loss** forces realistic frequency distribution |
| v9: Class balance | Same spectral issues | **Temporal smoothness loss** kills jitter |
| v10: Clean data | 4/8 audible but electric | **Stats matching** fixes HiFi-GAN input |
| v11: More data | Not yet implemented | **Smaller model** prevents overfitting on any data size |

The key insight: **Previous versions only optimized reconstruction quality. v12 optimizes for animal-like structure.** The classifier ensures the output "looks like a dog" to a model trained on real dog barks. The spectral balance ensures it has realistic frequency content. The temporal smoothness ensures it doesn't buzz.

---

*Built from analysis of v1-v11 results, actual generated audio measurements, and real vs generated mel comparison.*
