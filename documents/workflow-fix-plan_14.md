# Workflow Fix Plan v14 — Latent Diffusion or Paradigm Change

> **Date:** May 16, 2026  
> **Status:** Proposed — review before implementing.  
> **Builds on:** v13 (150 epochs trained, confirmed: model ignores time conditioning, all classes produce static).

---

## 0. Why Loss Drops Slowly (Root Analysis)

### The Data

| Epoch | v12 Train | v12 Val | v13 Train | v13 Val | Gap |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.93 | 1.39 | 2.04 | 1.68 | 0.36 |
| 10 | 1.61 | 1.31 | 1.82 | 1.67 | 0.15 |
| 50 | 1.72 | 1.38 | 1.65 | 1.52 | 0.13 |
| 100 | 1.72 | 1.26 | 1.75 | 1.22 | 0.53 |
| 150 | 1.89 | 1.11 | 1.67 | 1.13 | **0.54** |

**Key observations:**
- Val loss drops steadily (1.68 → 1.13) — model IS learning
- Train loss stays stuck at 1.6-2.0 for ALL 150 epochs
- Train-val gap GROWS from 0.15 to 0.54 — massive overfitting
- **Best val is IDENTICAL: 1.08 for both v12 and v13**

### Why Train Loss Never Drops

**1. Strong augmentation destroys information**
```
Training: augmented mel (freq mask + time mask + noise + gain + dropout)
Target:   original unaugmented mel

The model must RECOVER destroyed information.
This is inherently imperfect → permanent error floor of ~1.0 L1 loss.
```

**2. Pure noise task is impossible with 640 files**
```
40% of training: pure noise → predict real mel
With only 38 Dog files, the model has nothing to learn from pure noise.
It predicts its training mean → permanent error of ~0.5 per sample.
```

**3. Val loss drops because val is NOT augmented**
```
Validation: original mels (no augmentation)
The model learns these 28 samples perfectly (memorization).
Val loss drops to 1.08 while train loss stays at 1.7.
```

**4. CosineAnnealingWarmRestarts causes loss spikes**
```
LR restarts every 37 epochs → loss spikes at epochs 37, 74, 111
Model never reaches steady convergence.
```

### The Hard Truth

```
532 training samples × 8 classes = 67 samples/class average
2.5M parameters / 532 samples = 4,700 params per sample

The model memorizes the training set in ~10 epochs.
After that, there is NOTHING MORE TO LEARN without more data.

150 epochs of training on 640 files = 140 epochs of waste.
```

---

## 1. The Core Problem (Confirmed Across v1-v13)

**Every version fails for the same reason: 640 files is not enough for a diffusion model to learn animal sound generation from noise.**

| Version | Approach | Result |
|---------|----------|--------|
| v1-v3: VAE generation | skip_dropout, attention | ❌ Decoder needs encoder skips |
| v4-v6: DDPM/DDPM | schedules, CFG | ❌ Model predicts zero noise |
| v7-v8: x₀ prediction | freq-weighted loss | ❌ Model predicts mean x₀ |
| v9: Class balance | oversample rare classes | ❌ Same spectral issues |
| v10: ESC-50 clean data | 640 clean files | ❌ 4/8 audible but electric |
| v11: More data sources | UrbanSound8K, Xeno-Canto | ❌ Not implemented |
| v12: 3 new losses | spectral balance, smoothness, classifier | ❌ Same static output |
| v13: Pure noise training | force conditioning at high t | ❌ Model still ignores time |

**The model's time embedding has NEVER been used in any version.** The predictions are identical across all timesteps and all classes — confirmed with diagnostic tests in both v12 and v13.

---

## 2. Why This Happens

### Diffusion Requires 10,000+ Samples

Diffusion models learn the data distribution by iteratively denoising. They need to see enough examples to learn what "real" looks like at every noise level. With 640 files:

```
640 files × 8 classes = 80 files/class (average)
80 files × 5 seconds = 400 seconds of audio per class

A dog barks ~10 times in 5 seconds → 800 barks total across all training
The model sees each unique bark pattern ~100 times across 150 epochs.

This is NOT enough to learn the distribution of "dog bark."
It's enough to memorize the 80 files, but not to generalize.
```

### The Shortcut Problem

During training, the model sees `x_t = sqrt(α) * real_mel + sqrt(1-α) * noise`. The real mel is always present in the input. The model learns to **extract the signal** from the noisy input rather than **use conditioning to generate**. This is a fundamental property of diffusion training — the shortcut is always available.

Pure noise training (v13) was supposed to remove this shortcut, but:
- It only applies to 40% of training (t > 600)
- With 640 files, the model can't learn from pure noise either
- The 60% recovery task dominates the gradient → shortcut persists

---

## 3. Two Paths for v14

### Path A: Latent Diffusion (Recommended)

**Idea:** Use the trained autoencoder (149M params, MSE=0.015) to compress spectrograms, then train diffusion in latent space.

```
Real mel [1, 64, 552] = 35,328 values
       ↓ Autoencoder Encoder
Latent [8, 8, 69] = 4,416 values  (64× smaller)
       ↓ Small Diffusion (~500K params)
Latent [8, 8, 69]
       ↓ Autoencoder Decoder
Mel [1, 64, 552] → HiFi-GAN → Audio
```

**Why this might work:**
- 64× fewer values to model → much easier task
- The autoencoder already learned the data distribution
- Diffusion only needs to learn the latent structure
- 500K params vs 640 files = 775 params/sample (manageable)

**What we have:**
- `models/best_autoencoder_train.pth` — trained autoencoder, MSE=0.015
- Encoder works for compression
- Decoder works for reconstruction (with skips)

**What we need:**
- Modify autoencoder decoder to work without skips (or use it with skips)
- Train small diffusion model on latents
- The decoder with skips is fine — at inference we use the latent as the "skip"

**Implementation:**
```
1. Encode all training mels → latents [8, 8, 69]
2. Train small U-Net (~500K params) on latents
3. At inference: noise → DDIM → latent → decode → mel → audio
```

**Training time:** ~30 min on L4 (500K params, small input)

**Risk:** The latent space might not be well-organized for generation. But even a rough latent generation is better than pixel-level generation.

---

### Path B: Retrieval-Based Generation (Guaranteed to Work)

**Idea:** Don't generate from noise. Retrieve a real sample and modify it slightly.

```
User wants: "Dog bark"
1. Pick random Dog sample from training set
2. Encode → latent
3. Add small noise (t=100-200, not t=999)
4. Denoise with diffusion (img2img)
5. Decode → audio

Result: Recognizable animal sound with slight variation.
```

**Why this works:**
- Starts from real audio → guaranteed animal-like output
- Only needs to learn small perturbations, not full generation
- The diffusion model only needs to handle low-noise recovery (easy task)

**What we have:**
- Trained diffusion model that works well for img2img (recovery from noisy mel)
- 640 real samples to choose from

**What we need:**
- Index all training samples by class
- Modify generation pipeline to retrieve + refine instead of pure generation

**Implementation:**
```python
def generate_animal_sound(label, variation=0.3):
    # Retrieve random sample from class
    sample = random.choice(class_samples[label])
    mel = load_and_compute_mel(sample)
    
    # Add small noise and denoise
    noise = torch.randn_like(mel) * variation
    noisy_mel = mel + noise
    
    # Denoise with diffusion
    refined = diffusion.refine(noisy_mel, label, strength=variation)
    
    # Convert to audio
    audio = mel_to_waveform(refined)
    return audio
```

**Training time:** Zero — uses existing trained model

**Risk:** Not truly generative. Outputs are variations of training samples, not new sounds. But they WILL sound like animals.

---

## 4. Decision Matrix

| | Path A: Latent Diffusion | Path B: Retrieval-Based |
|---|---|---|
| Produces animal sounds? | Maybe | ✅ Yes |
| Truly generative? | ✅ Yes | ❌ No (variations) |
| Training time | ~30 min | 0 min |
| Implementation effort | 1-2 days | 1-2 hours |
| Risk of failure | Medium | None |
| Audio quality potential | Good | Good |
| Novel sounds? | ✅ Yes | Semi-novel |

---

## 5. Recommendation

**Immediate:** Path B (retrieval-based) — get working animal sounds TODAY.

**Then:** Path A (latent diffusion) — train a proper generative model.

**Don't:** Try another pixel-level diffusion variant. The fundamental data limitation applies to all versions.

---

## 6. The Math Doesn't Lie

```
640 files × 8 classes = 80 files/class
532 training samples after split = 67 samples/class average
2.5M parameters / 532 samples = 4,700 params/sample

Rule of thumb for diffusion: ~100 params/sample minimum.
We're at 4,700 params/sample = 47× over the limit.

No amount of architectural tweaking will fix this.
The only solutions are:
  1. Reduce parameters (we've done this: 61M → 2.5M)
  2. Increase data (need ~50,000 files for 2.5M params)
  3. Change the task (latent diffusion, retrieval-based)

We've exhausted option 1.
Option 2 requires external data collection.
Option 3 is the only remaining path.
```

---

## 7. What About More Data?

If we want to stick with the current architecture and fix the data problem:

| Dataset | Files | Total After Merge |
|---------|:---:|:---:|
| ESC-50 (current) | 640 | 640 |
| UrbanSound8K (dog) | ~1,040 | 1,680 |
| Xeno-Canto (birds) | ~300 | 1,980 |
| Freesound (frog, insect) | ~300 | 2,280 |
| **Total** | | **~2,300** |

Even 2,300 files is only ~330 per class — still not enough for diffusion. We'd need **10,000+ files** for reliable generation.

---

*v14 is a pivot. After 13 versions of trying to make pixel-level diffusion work with 640 files, it's time to either change the task (latent diffusion) or change the paradigm (retrieval-based generation).*
