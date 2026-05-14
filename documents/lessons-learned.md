# What We Learned — Animal Sound Generator

> Lessons from building this project end-to-end. VAE failures, diffusion successes, data matters, architecture traps.

---

## 1. VAE vs Diffusion: When to Use Each

### The VAE Mistake

We spent 4 attempts trying to make the VAE generate audio from random noise. It never worked. Here's why:

```
VAE = Encoder(input) → latent z → Decoder(z, skip_connections) → output
                                  ↑
                          Decoder needs BOTH z AND encoder features
```

The VAE decoder was designed for **reconstruction**, not generation. It takes `z` AND `encoder skip connections` as input. During training, it always had both. During generation (random z, no skips), the decoder lost half its input.

**Lesson:** A VAE trained for reconstruction doesn't automatically become a generator. The decoder needs to be designed for **both** modes from the start.

### When VAE Works

VAEs are great when:
- You want a **compressed representation** (latent z is smaller than input)
- You want **smooth interpolation** between samples (walking through latent space)
- You want **style transfer** (encode source, decode with different style)

VAEs are NOT great for:
- Generating from pure noise (unless decoder is specifically designed for it)
- High-fidelity output (VAE output is always blurry compared to real)

### When Diffusion Works

Diffusion generates by **iteratively denoising**:

```
Pure noise → denoise step 1 → denoise step 2 → ... → denoise step 50 → clean output
```

Every step is a small, easy prediction: "remove 2% of the noise." This is much easier than generating the whole thing at once.

**Lesson:** Diffusion is better for generation because:
- The task is decomposed into 50+ easy steps
- Each step only removes a little noise — low error per step
- The UNet learns the **data distribution**, not a compression
- No decoder/encoder split — one model does everything

---

## 2. Architecture Traps We Hit

### Trap 1: Skip Connections Block Generation

Our FiLMDecoderStage had this path:

```python
# During training (reconstruction):
encoder_features → skip connection → concat with decoder features → output ✅

# During generation (random z):
NO encoder → NO skip connection → decoder alone → garbage output ❌
```

We tried:
- `skip_dropout=0.5` (train 50% without skips) → still noise
- `skip_dropout=1.0` (never see skips) → loss exploded
- Adding self-attention to replace skips → still failed

**Lesson:** If you design a decoder with skip connections from the encoder, it will **depend** on them. The decoder learns to use the skip features because they're easier than generating from scratch. You can't train this dependency away — you need to design the decoder WITHOUT skips from the start.

### Trap 2: Latent Space Collapse

```
Training:   z = encoder_μ + σ·ε     (μ=0.14, σ=0.97 → 86% noise)
Generation: z = random_normal        (100% noise, no content signal)
```

The KL divergence pushes μ→0. The decoder compensates by using **skip connections** instead of z for content. Result: z carries almost no information. During generation, random z has no content.

**Lesson:** The VAE's KL loss and the decoder's skip connections fight each other. KL says "make z simple" while the decoder needs z to carry content. The decoder cheats by using skips instead of z. This is called **posterior collapse** — a well-known VAE failure mode.

### Trap 3: Normalization Must Match Everywhere

We initially had wrong normalization stats in SimpleNormalize. This caused the autoencoder to not learn. After fixing it:

```
All components must agree:
  SimpleNormalize(mean=-18.49, std=19.80)      ← data_loader.py
  mel = (db - 18.49) / 19.80                   ← diffusion/train.py
  HiFi-GAN unnormalize: mel * 19.80 - 18.49    ← hifigan/inference.py
```

One wrong number anywhere = entire pipeline breaks.

**Lesson:** When you have multiple models in a pipeline (autoencoder → VAE → HiFi-GAN → diffusion), the normalization must be **bit-exact identical** at every boundary. Document these constants.

---

## 3. Data Matters — What We Learned

### Dataset Size vs Model Size

| Model | Params | Samples | Ratio | Result |
|-------|:------:|:-------:|:-----:|--------|
| Classifier | 457K | 5000 | 91:1 | ✅ Fast convergence |
| Autoencoder | 149M | 5000 | 30,000:1 | ✅ Still learned (shared weights via skips) |
| VAE | 223M | 5000 | 45,000:1 | ❌ Couldn't generate |
| Diffusion UNet | 61M | 2700 | 22,000:1 | 🟡 Learning slowly |

**Lesson:** For generative models, 22,000 params per sample is a lot. The UNet is learning but slowly. If we had 50,000 samples instead of 5,000, training would be 10× faster. Data augmentation (time shift, pitch shift, SpecAugment) helps but can't replace real diversity.

### Class Imbalance Hurts Generation

| Class | Samples | Generation Quality |
|-------|:-------:|:---:|
| Dog | 750 | Moderate |
| Cat | 303 | Moderate |
| Insect | 371 | Good |
| **Frog** | **61** | Poor |
| **Crow** | **72** | Poor |

The VAE's classification agreement was only 42% for Frog and 34% for Crow — the classes with fewest samples. More data = better generation.

**Lesson:** Generative models need balanced data. 61 Frog samples can't teach a 61M-param model what a frog sounds like. Either get more data, augment heavily, or merge rare classes.

### Spectrogram Resolution Trade-offs

| n_mels | Time frames | Detail | Training Speed |
|:------:|:-----------:|:------:|:--------------:|
| 64 | 552 (5s) | Medium | Fast |
| 128 | 1104 (5s) | High | 4× slower |
| 256 | 2208 (5s) | Very high | 16× slower |

We used 64 mel bins — good enough for animal sounds (dogs bark at low frequencies, birds at high). For music or speech, you'd need 128+.

**Lesson:** 64 mel bins × 552 time frames = a 64×552 "image" that CNNs handle well. This is the sweet spot for memory and quality. More resolution helps but quadratically increases VRAM.

### Smart Crop Helped

Our `smart_crop` function extracts the loudest 5-second segments from longer files. A 20-second audio with 3 barks becomes 3 training samples — 3× more data from the same files.

**Lesson:** Energy-based cropping is free data augmentation for audio. Always do it.

---

## 4. Training Tricks That Worked

### Learning Rate Matters Massively

| Model | LR Too Low | LR Right | LR Too High |
|-------|:---:|:---:|:---:|
| Autoencoder (149M) | 3e-4 → stuck at MSE=1.0 | **1e-3** → MSE=0.015 | — |
| VAE (223M) | — | 3e-4 | 1e-3 → possible instability |
| Diffusion (61M) | — | 2e-4 | — |
| Classifier (457K) | — | 1e-3 | — |

**Lesson:** Bigger models need higher LR (more params = smaller per-param gradient). The autoencoder required 1e-3 to move 149M params; 3e-4 was too slow.

### Warmup + Freeze Strategy

```python
# Phase A: Freeze pretrained encoder/decoder, train only new layers
freeze(encoder, decoder_convs)
train(vae_heads, film, gen_attn)

# Phase B: Unfreeze everything, full training
unfreeze_all()
train(all)
```

This let the VAE learn FiLM conditioning and gen_attn before the conv weights shifted. Without warmup, new layers pull pretrained weights off course.

**Lesson:** When adding layers to a pretrained model, freeze the base for a few epochs so new layers can catch up.

### CosineAnnealingWarmRestarts > CosineAnnealing

The diffusion training uses `CosineAnnealingWarmRestarts` — the LR periodically resets to high, then decays. This helps escape local minima. Standard `CosineAnnealingLR` just decays to zero and stays there.

**Lesson:** For long training runs (>20 epochs), cyclic LR schedules prevent plateauing.

---

## 5. The Full Stack — What Each Piece Does

```
┌─────────────────────────────────────────────────────┐
│                  TRAINING STACK                       │
│                                                       │
│  Raw audio .wav                                        │
│      │                                                 │
│      ├─→ Smart crop → 5s segments                     │
│      │                                                 │
│      ├─→ MelSpectrogram(n_mels=64, hop=200)           │
│      │    → AmplitudeToDB(top_db=80)                   │
│      │    → SimpleNormalize(μ=-18.49, σ=19.80)        │
│      │                                                 │
│      ├─→ Classifier (457K) → 95% accuracy              │
│      │    Used for: evaluation, VAE supervision         │
│      │                                                 │
│      ├─→ Autoencoder (149M) → compress/reconstruct     │
│      │    Used for: VAE weight initialization           │
│      │                                                 │
│      ├─→ VAE (223M) → encode/decode with class         │
│      │    Used for: style transfer, reconstruction      │
│      │    NOT for: generation (decoder needs skips)     │
│      │                                                 │
│      ├─→ Diffusion UNet (61M) ★ GENERATOR              │
│      │    Noise → 50 DDIM steps → mel spectrogram      │
│      │                                                 │
│      └─→ HiFi-GAN (3M) → mel → audio waveform          │
│                                                       │
└─────────────────────────────────────────────────────┘
```

Each model has one job. The pipeline works because each piece agrees on the data format: **normalized mel spectrogram [1, 64, 552] with mean≈0, std≈1**.

---

## 6. Quick Reference: When To Use What

| You want to... | Use | Because |
|----------------|-----|---------|
| Classify sounds | Classifier (CNN on mel) | Fast, accurate, 457K params |
| Compress/reconstruct | Autoencoder | Learned compression, ~30× reduction |
| Interpolate between sounds | VAE latent space | Smooth latent organization |
| Transfer style (Dog→Cat) | VAE + FiLM | Encode source, FiLM target class |
| **Generate from scratch** | **Diffusion UNet** | Iterative denoising, no encoder needed |
| Convert mel→audio | HiFi-GAN | Neural vocoder, 200× upsampling |
| Polish blurry output | Diffusion refinement | img2img denoising |

---

## 7. The Biggest Lesson

**Don't force one model to do two things.** The VAE was asked to both reconstruct (needs skips) and generate (can't have skips). These are conflicting requirements. The fix wasn't better training — it was **using the right model for each job**: VAE for reconstruction/style transfer, diffusion for generation.

This applies everywhere: one model per responsibility, clear interfaces (normalized mel format), and verify each piece independently before connecting them.
