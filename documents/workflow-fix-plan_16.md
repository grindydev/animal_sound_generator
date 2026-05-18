# Workflow Fix Plan v16 — Class-Conditional GAN

> **Date:** May 17, 2026
> **Status:** Implementing
> **Builds on:** v1-v15 (all diffusion variants failed — model ignores conditioning)
> **Approach:** GAN — generator always starts from pure noise, no shortcut possible

---

## 0. Why GAN Fixes the 15-Version Failure

```
DIFFUSION (v1-v15, ALL failed):
  Training:   x_t = √α × real + √(1-α) × noise → predict clean
              Model extracts signal from input → ignores time/class

  Inference:  x_T = pure noise → predict clean
              No signal to extract → outputs mean → white noise ✓

GAN (v16):
  Training:   z = pure noise → generate fake mel
              No signal in input → MUST use class conditioning

  Inference:  Same as training — generator always starts from noise
              → Works by design, not by luck
```

No architectural tweaks needed. The paradigm itself prevents the shortcut.

---

## 2. Architecture

### Generator (~6M params)

```
Noise z [256] + Class label
     ↓
Class Embedding [256]
     ↓
MLP → conditioning [512]
     ↓
Dense → [256, 4, 18]
     ↓
FiLM UpBlock × 4   (class-conditioned at every level)
     ↓
[1, 64, 576] → crop → [1, 64, 552] mel spectrogram
```

Each UpBlock: Upsample → Conv → FiLM(h, class_cond) → LeakyReLU

### Discriminator (~4M params)

```
Mel [1, 64, 552] → pad → [1, 64, 576]
     ↓
Conv blocks × 6 (stride 2×2, spectral norm)
     ↓
[512, 4, 18]
     ├→ Flatten → FC → 1     (real/fake)
     └→ Class head → FC → 7  (class prediction)
```

### Losses

| Loss | Formula | Purpose |
|------|---------|---------|
| D adversarial | `hinge(D(real), D(fake))` | Distinguish real/fake |
| D class | `CE(D_cls(real), label)` | Learn class structure |
| R1 penalty | `γ/2 × ‖∇D(real)‖²` | Stabilize training |
| G adversarial | `-D(fake)` | Fool discriminator |
| G class | `CE(D_cls(fake), label)` | Generate correct class |
| G L1 (optional) | `L1(fake, nearest_real)` | Anchor to real data |

---

## 3. Key Design Decisions

| Decision | Why |
|----------|-----|
| Class conditioning via FiLM | Per-layer modulation, proven in BigGAN/StyleGAN |
| Spectral norm in discriminator | Industry standard for GAN stability |
| Hinge loss | Better gradient signal than BCE |
| R1 penalty | Prevents discriminator overfitting on small data |
| Auxiliary classifier (both D and G) | Forces G to produce class-distinct output |
| No progressive growing | Simpler training, manageable at 64×552 resolution |
| Griffin-Lim for audio | No HiFi-GAN training mismatch artifacts |

---

## 4. Training

| Parameter | Value |
|-----------|-------|
| Epochs | 300 |
| Batch size | 16 |
| G LR / D LR | 2e-4 / 2e-4 |
| R1 gamma | 10.0 |
| Optimizer | Adam (β₁=0.0, β₂=0.99) |
| AMP | Yes |
| Augmentation | Mild SpecAugment (freq/time mask) |

---

## 5. Success Criteria

| Metric | Target |
|--------|:------:|
| Classifier accuracy on generated (D_cls) | > 60% |
| Audio peak frequency per class | Class-specific (dog ~800Hz, crow ~2000Hz, etc) |
| Spectral flatness | < 0.6 (not noise) |
| Listenable animal sounds | 5+/7 classes |

---

## 6. Scaling

| Dataset Size | Expected Quality |
|:---:|------|
| 4K | Rough but recognizable animal sounds |
| 20K | Distinct per-class character, some detail |
| 100K | Clear, detailed, diverse |
| 1M+ | Production quality |

Same architecture across all scales.

---

## 7. Files

| File | Purpose |
|------|---------|
| `src/gan/config.py` | Configuration |
| `src/gan/generator.py` | Generator with FiLM conditioning |
| `src/gan/discriminator.py` | Discriminator with spectral norm + aux classifier |
| `src/gan/train.py` | Training loop with R1 penalty |
| `src/gan/generate.py` | Inference → audio |
| `colab/colab_train.ipynb` | Updated Colab notebook |
