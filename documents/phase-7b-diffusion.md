# Phase 7b — Diffusion Refinement

> **One doc to learn from. You understand this, you can rebuild diffusion for any project.**

---

## 1. The Problem

VAE generates blurry spectrograms — it averages over possibilities, so edges get smoothed out. HiFi-GAN converts whatever mel you give it, blur included. Diffusion is the sharpener.

```
"dog" → VAE → blurry mel → Diffusion → sharp mel → HiFi-GAN → crisp audio ✨
```

---

## 2. The Two Processes

### Training: Add noise, learn to remove it

Take a spectrogram. Add random noise. Ask U-Net: "where's the noise?" Grade the answer. Repeat.

**70% of training data is real mels, 30% is VAE-reconstructed mels.** This teaches the model what VAE-quality data looks like — so at inference, when it receives an actual VAE output, it's not surprised by the blurriness.

```
Real mel or VAE-reconstructed mel → add noise → U-Net guesses → compare → adjust
```

### Inference: Start noisy, clean it up

Take VAE output + some noise. Run U-Net ~50 times, each time removing a little predicted noise. Result: sharp spectrogram. This is "img2img" — start from an existing image, refine it.

```
VAE mel + noise → [clean a bit] → [clean more] → ... → sharp mel
```

---

## 3. Why t Is Essential

The U-Net receives a noisy spectrogram and must predict what noise was added. **But the same noisy result can come from different situations:**

- Clean sound + heavy noise (t=900)
- Already-noisy sound + light noise (t=100)

Both look identical. The model can't know which case it is just by looking.

**t solves this.** t is a number (0–999) saying exactly how much noise was added:

```
Without t:  U-Net sees noisy image → has to guess → often wrong
With t:     U-Net sees noisy image + "t=347" → "ah, 35% noise" → correct
```

t also tells the model the **strategy** to use:

| t | Noise level | Strategy |
|---|------------|----------|
| 0–100 | ~clean | "Almost everything is signal. Predict tiny noise." |
| 400–600 | half-half | "Hardest case. Must separate signal from noise carefully." |
| 900–999 | pure noise | "Almost everything is noise. Just output what I see." |

---

## 4. How t Works Inside the Model

### Step 1: Pick t (training)

```python
# train.py, train_epoch():
t = torch.randint(0, 1000, (B,))   # e.g., [347, 891, 12]  — shape [B]
```

### Step 2: t → clock hands → rich vector

```python
# unet.py, SinusoidalTimeEmbedding.forward():
# t = [347]
# → 128 clock hands spin to different positions based on t
# → sin + cos = 256 numbers (128 hands × 2 coordinates)
# → MLP (256 → 1024 → 1024)
t_emb = self.time_embed(t)     # [B] → [B, 1024]
```

Why 128 hands at different speeds? Fast hands encode "exact step number." Slow hands encode "early, middle, or late." Together they form a unique fingerprint for every t. Same idea as Transformer positional encoding.

### Step 3: t becomes knobs inside every ResBlock (FiLM)

t never touches the spectrogram pixels. Instead, it controls **how** features get processed:

```python
# unet.py, ResBlock.forward():
h = self.conv1(x)                    # process spectrogram features

t_knobs = self.time_mlp(t_emb)       # [B, 1024] → [B, out_ch*2]
scale, shift = t_knobs.chunk(2)      # one volume + one bias per channel

h = h * (1 + scale) + shift          # ← ONLY HERE does t touch features
```

This is **FiLM** (Feature-wise Linear Modulation): time → two numbers per channel → multiply and add.

```
t=10:   scale ≈ 0,  shift ≈ 0    → "pass through, barely change anything"
t=500:  scale ≈ 0.5, shift ≈ 0.2 → "moderate transformation"
t=999:  scale ≈ 2.0, shift ≈ -0.5 → "aggressively transform to find noise"
```

### Step 4: Every ResBlock gets its own time MLP

Each ResBlock has a **separate learned MLP** that converts the same `t_emb` into different knobs:

```
t_emb [B, 1024] ──┬──→ ResBlock 0: time_mlp_0(t_emb) → knobs for edge detection
                  ├──→ ResBlock 1: time_mlp_1(t_emb) → knobs for texture analysis
                  ├──→ ResBlock 2: time_mlp_2(t_emb) → knobs for coarse structure
                  └──→ ...  (every block learns its own knob settings)
```

Shallow layers amplify fine edges. Deep layers suppress textures. Each layer learns what's useful at its resolution.

---

## 5. What the Model Actually Learns

The model doesn't just randomly nudge weights. **It learns animal sound patterns** so it can separate signal from noise.

```
Input: x_t = 0.6 × real_mel + 0.8 × noise       (scrambled together)

Internally, the model separates:

    STRUCTURED SIGNAL              RANDOM NOISE
    (dog bark pattern)             (just static)

    ██░░░░██░░░░  ← harmonic       ░░░░░░░░░░░░░░
    ██░░░░██░░░░  ← overtone       ░░░░░░░░░░░░░░
    ░░████████░░  ← formant        ░░░░░░░░░░░░░░
```

After thousands of dog barks, weights encode: "Dog barks have harmonics here, formants here, burst every ~100ms..." When given `x_t`, the model thinks: "These structured bands match dog patterns → signal. This random fuzz → noise."

The class embedding tells it **which** animal's patterns to look for:

```
Class = "Dog":   look for barking harmonics, 200–2000Hz formants
Class = "Frog":  look for croaking patterns, different bands
Class = "Crow":  look for cawing structure, different temporal pattern
```

---

## 6. The U-Net Architecture

### Why U-Net? Why not just encoder → flat → decoder like VAE?

Different goals:

```
VAE:                          U-Net (Diffusion):
  Goal: generate from code      Goal: edit existing pixels

  Compress to vector            Keep spatial grid throughout
  Decoder REMEMBERS patterns    Decoder DECIDES per pixel
  No skips (building fresh)     Skips (preserving what's there)
```

**VAE decoder remembers:** "I have z=[1024] and I know what a dog looks like. Let me draw one from my stored knowledge."

**U-Net decoder decides:** "Encoder found dog patterns. Skips have the raw pixels. I just need to say: this pixel is signal (keep), this pixel is noise (remove)."

The encoder does the heavy lifting (pattern recognition). The decoder is lightweight — mostly combining and filtering. The time embedding provides the decision strategy.

### The 5 Building Blocks (Reusable Pattern)

```
┌─────────────────────────────────────────────┐
│  1. INPUT PROJECTION                        │
│     Conv2d(1 → 64)                          │
│                                             │
│  2. ENCODER (4 levels)                      │
│     ResBlock × 2 → Downsample → save skip   │  Shrink resolution, grow channels
│                                             │
│  3. BOTTLENECK                              │
│     ResBlock × 2 + Self-Attention           │  Deepest, most abstract
│                                             │
│  4. DECODER (4 levels, reversed)            │
│     Upsample → Concat skip → ResBlock × 2   │  Rebuild with saved detail
│                                             │
│  5. OUTPUT PROJECTION                       │
│     GroupNorm → Conv2d(64 → 1)              │
└─────────────────────────────────────────────┘
```

### Only 2 things are diffusion-specific:

**Addition 1: Time conditioning (FiLM)** — 3 extra lines in each ResBlock

```python
# Standard ResBlock:
def forward(self, x):
    h = conv1(norm(x)); h = conv2(norm(h)); return x + h

# Diffusion ResBlock:
def forward(self, x, t_emb):
    h = conv1(norm(x))
    scale, shift = time_mlp(t_emb).chunk(2)   # ← time → knobs
    h = h * (1 + scale) + shift               # ← apply
    h = conv2(norm(h)); return x + h
```

**Addition 2: Class conditioning** — same FiLM pattern from learned embedding. Optional (drop for unconditional generation).

### To rebuild for any project:

```
Step 1: Build standard U-Net (encoder-decoder with skips)
Step 2: Add FiLM in each ResBlock (time → MLP → scale+shift → apply)
Step 3: Add optional conditioning (class, text) using same FiLM pattern
```

---

## 7. Skip Connections — Two Reasons

**Reason 1 (classic, what you learned): Gradient flow.** In deep networks, gradient shrinks at each layer. Skip gives gradient a direct highway back to early layers.

**Reason 2 (U-Net specific): Detail preservation.** Downsampling throws away 93% of pixels. Without skips, the decoder must guess where edges were. With skips, it receives the original detail back:

```
Without skip:                          With skip:
  [64×552] → compress → [8×69]          [64×552] → compress → [8×69]
  [8×69] → stretch → [64×552]           [8×69] + [saved 64×552] → [64×552]
  "Where was that edge again?"          "Here's the exact edge. Keep or remove?"
  → blurry                              → sharp
```

---

## 8. Bottleneck + Self-Attention

**Bottleneck:** The deepest, most compressed layer. At [8×69], each cell represents an 8×8 region of the original spectrogram. Most abstract understanding — "this region contains a dog bark harmonic."

**Self-attention:** Every cell talks to every other cell. "The 500Hz harmonic at position (3,40) should relate to the 1000Hz overtone at (6,40)."

Only at the bottleneck because it's O(N²) — expensive at high resolution, but at [8×69] it's only 552 positions → 304K pairs → cheap and most useful (long-range relationships matter here).

Without self-attention: each cell processes independently. With: cells share information globally.

---

## 9. Training

### Dataset: 70% real + 30% VAE reconstructions

```python
# train.py, DiffusionDataset.__getitem__():
mel = compute_mel(cropped_wav)           # real mel from audio

if random() < 0.3:                        # 30% of the time:
    mel = vae_model(real_mel, label)      # replace with VAE reconstruction
```

Why? If the model only sees perfect real mels during training, it panics at inference when VAE outputs are blurry. Mixing in VAE-quality data teaches it: "you'll get blurry inputs — that's normal, just sharpen them."

### Per-batch training loop

```python
# train.py, train_epoch():
for mel, labels in train_loader:              # mel: [B, 1, 64, 552]
    t = torch.randint(0, 1000, (B,))           # random t per sample
    noise = torch.randn_like(mel)              # random noise
    
    x_t = diffusion.q_sample(mel, t, noise)    # x_t = √α·x₀ + √(1-α)·noise
    pred_noise = model(x_t, t, labels)          # U-Net guesses noise
    loss = MSE(pred_noise, noise)               # compare to real noise
    
    loss.backward()                             # compute gradients
    optimizer.step()                            # nudge weights
```

The model sees every t value thousands of times across epochs, on both real and VAE-quality data.

---

## 10. Key Files

| File | Contains |
|------|----------|
| `src/diffusion/config.py` | All hyperparameters |
| `src/diffusion/unet.py` | `SpectrogramUNet`, `ResBlock`, `SinusoidalTimeEmbedding`, `SelfAttention2D` |
| `src/diffusion/diffusion.py` | `DiffusionProcess` — `q_sample()` (add noise), `p_sample()` (remove), DDIM |
| `src/diffusion/train.py` | Dataset (with VAE mix-in), training loop, checkpointing |
| `src/diffusion/inference.py` | `refine_spectrogram()`, `generate_refined()` |

---

## 11. Quick Config

```python
# config.py
timesteps = 1000              # total noise levels
time_emb_dim = 256            # time embedding size
base_channels = 64            # U-Net starting channels
channel_multipliers = (1, 2, 3, 3)   # encoder depths → ~17.8M params
class_emb_dim = 64            # animal type embedding
dropout = 0.1                 # regularization
inference_steps = 50          # DDIM steps (don't need all 1000)

# train.py CONFIG
vae_mix_ratio = 0.3           # 30% VAE reconstructions in training
```

---

## 12. Design Pattern Summary

| Pattern | What It Does | Reusable For |
|---------|-------------|--------------|
| U-Net skeleton | Down-bottleneck-up with skips | Any pixel-level prediction |
| FiLM conditioning | Time → scale+shift per channel in every block | Any conditional generation |
| Sinusoidal embedding | Integer → multi-frequency fingerprints | Any time-step conditioning |
| Skip connections | Bypass compression, preserve detail | Any encoder-decoder |
| Self-attention at bottleneck | Long-range feature relationships | Any architecture with bottleneck |
| EMA shadow model | Smoothed weights for inference | Any training loop |
| VAE data mix-in | Train on target-domain data | Any refinement model |

---

*Built from code in `src/diffusion/`. Read this doc, then read the source — the concepts map 1:1.*
