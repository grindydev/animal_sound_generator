# Phase 7b — Diffusion Refinement

> **One-line summary:** A small U-Net diffusion model sharpens VAE-generated blurry spectrograms before HiFi-GAN converts them to audio.

---

## 1. The Problem Diffusion Solves

Your VAE generates a mel spectrogram. HiFi-GAN converts it to audio. But HiFi-GAN is a **faithful converter** — it converts whatever mel you give it, blur included.

The **blur comes from the VAE decoder.** VAEs average over possibilities. The decoder smooths out detail:

```
Real dog bark spectrogram:           VAE-generated spectrogram:
  sharp edges                          blurred edges
  crisp harmonics                      washed-out harmonics
  clear onsets/offsets                 smeared onsets/offsets

  ██████░░░░                           ▓▓▓▓▓▓░░░░
  ██░░░░░░░░                           ▓▓▓░░░░░░░
  ░░████░░░░                           ░░▓▓▓▓░░░░
```

**Diffusion is a refiner.** It takes the blurry VAE output, adds noise, then iteratively removes it — restoring sharp edges, harmonics, and detail along the way.

```
Full pipeline:
═══════════════════════════════════════════════════════════

"dog" label
    ↓
┌──────────┐
│   VAE    │  ← generates rough spectrogram (blurry)
└────┬─────┘
     ↓
[blurry mel]
     ↓
┌──────────────┐
│  Diffusion   │  ← U-Net denoises / sharpens
│  (Phase 7b)  │    restores edges, harmonics, detail
└────┬─────────┘
     ↓
[sharp mel]
     ↓
┌──────────────┐
│  HiFi-GAN    │  ← converts to crisp waveform
│  (Phase 7a)  │
└────┬─────────┘
     ↓
[crisp AND sharp audio] ✨
```

Without diffusion → crisp-but-blurry. With diffusion → crisp AND sharp.

---

## 2. The Two Processes (Forward + Reverse)

Diffusion is two processes: one that destroys, one that creates.

### Forward Process (q) — Adding Noise

This happens ONLY during training. Take a real spectrogram and gradually add Gaussian noise over T steps.

```
Step 0:  x₀ = real clean mel          ████████░░  (sharp edges)
Step 1:  x₁ = x₀ + tiny noise         ▓▓██████░░  (slightly fuzzy)
Step t:  xₜ = xₜ₋₁ + more noise       ▓▓▓▓▓▓▓░░  (getting worse)
Step T:  x_T = pure random noise       ▓▓▓▓▓▓▓░░  (completely destroyed)
```

**Key formula (DDPM):**
```
x_t = √ᾱ_t · x₀  +  √(1 - ᾱ_t) · ε

Where:
  α_t  = 1 - β_t          (β_t comes from the noise schedule)
  ᾱ_t = ∏ α_i from i=1 to t   (cumulative product — how much signal remains)
  ε    = random Gaussian noise
```

This is a **closed-form** solution — you can jump from x₀ to x_t in one step without iterating. That's why DDPM training is fast.

### Reverse Process (p) — Removing Noise

This happens at inference. Start from a noisy spectrogram and iteratively remove predicted noise.

```
Step T:  x_T = noisy mel (VAE output + noise)   ▓▓▓▓▓▓▓░░
    ↓     U-Net predicts noise ε
Step T-1: remove predicted noise, add tiny fresh noise
    ↓
    ...
Step 0:  x₀ = sharp mel                          ████████░░
```

**p_sample (DDPM, high quality):**
```python
x_{t-1} = 1/√α_t · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t))
          + σ_t · z   # where z ~ N(0, I), σ_t = √β_t
```

The formula has two parts:
1. Remove the predicted noise `ε_θ` (what the U-Net guessed)
2. Add a tiny bit of fresh randomness `z` (stochasticity prevents getting stuck)

**p_sample_ddim (faster, fewer steps):**
```python
x_{t-1} = √ᾱ_{t-1} · (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t  +  √(1-ᾱ_{t-1}) · ε
```

DDIM is deterministic (no random z). You can sample with 50 steps instead of 1000 with minimal quality loss.

### Inference: img2img Refinement

```python
def refine(x_vae, start_t, num_steps):
    # Step 1: Add noise to VAE output (not full noise — partial)
    noise = torch.randn_like(x_vae)
    x = √ā_{start_t} · x_vae  +  √(1-ā_{start_t}) · noise

    # Step 2: Denoise from start_t down to 0
    for t in reversed(range(start_t)):
        ε = unet(x, t, class_label)     # U-Net predicts noise
        x = p_sample_ddim(x, t, ε)       # remove predicted noise

    return x   # sharp spectrogram
```

**`strength` controls how much refinement happens:**

| Strength | start_t (of 1000) | Effect |
|----------|-------------------|--------|
| 0.3 | t=300 | Light polish, preserves VAE structure |
| 0.6 | t=600 | Good balance — **recommended** |
| 0.9 | t=900 | Heavy change, less VAE influence |
| 1.0 | t=1000 | Pure diffusion from scratch |

---

## 3. Noise Schedule

**File:** `src/diffusion/diffusion.py` — `DiffusionProcess.__init__()` and `_cosine_schedule()`

The noise schedule controls how much noise is added at each timestep. Two common schedules:

### Cosine Schedule (used here)

```
β_t follows a cosine curve — smooth, less noise early, more late.
From "Improved DDPM" (Nichol & Dhariwal, 2021).
```

Why cosine over linear:
- **Early steps (t small):** Less noise → U-Net focuses on subtle, high-frequency details
- **Late steps (t large):** More noise → U-Net handles large-scale structure

```
Noise level (1-ā_t) over timesteps:

1.0 ┤                                    ╭───
    │                                ╭───╯    Cosine (smooth)
0.8 ┤                           ╭────╯
    │                      ╭────╯
0.5 ┤                 ╭────╯
    │            ╭────╯
0.3 ┤       ╭────╯
    │  ╭────╯                            ╱
0.0 ┤──╯                                ╱  Linear (straight)
    └──────────────────────────────
   t=0                              t=T
```

**Config values:**
```python
timesteps: int = 1000
beta_start: float = 0.0001      # minimum noise per step
beta_end: float = 0.02          # maximum noise per step
cosine_s: float = 0.008         # smoothness parameter
```

### DDIM Sampling

At inference, we don't use all 1000 steps. We pick N steps (e.g., 50) evenly spaced:

```python
# inference.py — ddpm_sampling()
timesteps_to_use = torch.linspace(start_t, 0, num_steps, dtype=torch.long)
# E.g.: [600, 588, 575, ..., 25, 12, 0] for start_t=600, num_steps=50
```

---

## 4. U-Net Architecture — Spectrogram → Predicted Noise

**File:** `src/diffusion/unet.py`  
**Class:** `SpectrogramUNet`  
**Size:** ~25.8M params (full) or ~17.8M (with `channel_multipliers=(1,2,3,3)`)

### Input / Output

**Input:** `x_t` [B, 1, 64, T] — noisy mel spectrogram at timestep t  
**Conditioning 1:** `t` [B] — timestep index (0–999) → sinusoidal embedding  
**Conditioning 2:** `labels` [B] — animal class (0–7) → learned embedding  
**Output:** predicted noise `ε̂` [B, 1, 64, T] — same shape as input

### Architecture Diagram

```
Input: [B, 1, 64, W] noisy mel
           │
    ┌─────────────────────────────┐
    │ INITIAL CONV                │  Conv2d(1→64, 1×1) — channel expansion
    └─────────────────────────────┘
           │  [B, 64, 64, W]
           ▼
    ╔══════════════════════════════════════════════╗
    ║              ENCODER (down)                  ║
    ║                                              ║
    ║   LEVEL 0: [B,  64, 64, W  ]                ║
    ║     → 2× ResBlock(64→64)  ← skip → decoder  ║
    ║     → Downsample (stride=2)                  ║
    ║                                              ║
    ║   LEVEL 1: [B, 128, 32, W/2]                ║
    ║     → 2× ResBlock(128→128) ← skip → decoder  ║
    ║     → Downsample                            ║
    ║                                              ║
    ║   LEVEL 2: [B, 256, 16, W/4]  🔍 attention  ║
    ║     → Self-Attention(256)                    ║
    ║     → 2× ResBlock(256→256) ← skip → decoder  ║
    ║     → Downsample                            ║
    ║                                              ║
    ║   LEVEL 3: [B, 256,  8, W/8]  🔍 attention  ║
    ║     → Self-Attention(256)                    ║
    ║     → 2× ResBlock(256→256) ← skip → decoder  ║
    ╚══════════════════════════════════════════════╝
           │  [B, 256, 8, W/8]
           ▼
    ┌─────────────────────────────┐
    │ BOTTLENECK                  │  Self-Attention(256)
    │                             │  2× ResBlock(256→256)
    └─────────────────────────────┘
           │  [B, 256, 8, W/8]
           ▼
    ╔══════════════════════════════════════════════╗
    ║              DECODER (up)                   ║
    ║                                              ║
    ║   LEVEL 3: concat(skip[3]) → [B, 512, 8, W/8]║
    ║     → 2× ResBlock(512→256)                   ║
    ║     → Upsample (stride=2)                    ║
    ║                                              ║
    ║   LEVEL 2: concat(skip[2]) → [B, 512,16,W/4]║
    ║     → Self-Attention(256)                    ║
    ║     → 2× ResBlock(512→128)                   ║
    ║     → Upsample                               ║
    ║                                              ║
    ║   LEVEL 1: concat(skip[1]) → [B, 256,32,W/2]║
    ║     → 2× ResBlock(256→128)                   ║
    ║     → Upsample                               ║
    ║                                              ║
    ║   LEVEL 0: concat(skip[0]) → [B, 128,64,W]  ║
    ║     → 2× ResBlock(128→64)                    ║
    ╚══════════════════════════════════════════════╝
           │  [B, 64, 64, W]
           ▼
    ┌─────────────────────────────┐
    │ FINAL CONV                  │  Conv2d(64→1, 1×1)
    └─────────────────────────────┘
           │
           ▼
Output: [B, 1, 64, W] predicted noise ε̂
```

### Why U-Net with Skip Connections?

The encoder loses spatial resolution. Skip connections give the decoder access to **uncompressed detail** from the encoder.

```
Without skip:                     With skip:
  Encoder                          Encoder
    │ [detail lost]                  │ [detail preserved → skip]
    ▼                               ▼
  Bottleneck                       Bottleneck ────┐
    │ [can't recover detail]         │ [has raw detail]
    ▼                               ▼
  Decoder → blurry output          Decoder + skip → crisp output
```

For spectrograms, detail = sharp frequency edges, clear overtones, precise onset timing.

### Time Embedding (Sinusoidal)

```python
# unet.py — SinusoidalTimeEmbedding
def get_timestep_embedding(timesteps, embedding_dim):
    half = embedding_dim // 2
    freqs = exp(-log(10000) * arange(half) / half)         # decaying frequencies
    args = timesteps[:, None].float() * freqs[None, :]     # t × freq
    emb = cat([sin(args), cos(args)], dim=-1)              # [B, embedding_dim]
    return emb
```

Each timestep becomes a unique 256-dim vector. Timestep 0 and 999 have very different embeddings. The U-Net learns that "at t=100, noise is light → focus on fine detail" vs "at t=900, noise is heavy → focus on broad structure."

### Class Embedding (Learned)

```python
# unet.py — __init__
self.class_embedding = nn.Embedding(config.num_classes, config.class_emb_dim)
# num_classes=8 (Dog, Cat, Rooster, Frog, Crow, Insect, Hen, Noise)
# class_emb_dim=64
```

Each animal gets a learned 64-dim vector. At inference, providing the correct class (e.g., "Dog") guides denoising toward dog-like spectral patterns.

### ResBlock with FiLM Conditioning

```python
# unet.py — ResBlock.forward()
def forward(self, x, time_emb, class_emb):
    # Main path
    h = silu(self.norm1(x))
    h = self.conv1(h)

    # FiLM: time + class injected as scale & shift
    # time:  [B, 256] → MLP → [B, channels]
    # class: [B, 64]  → MLP → [B, channels]
    scale, shift = time_mlp(time_emb).chunk(2, dim=1)
    h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    # + class embedding
    class_scale, class_shift = class_mlp(class_emb).chunk(2, dim=1)
    h = h * (1 + class_scale[:, :, None, None]) + class_shift[:, :, None, None]

    # Second conv
    h = silu(self.norm2(h))
    h = self.conv2(h)

    return x + h   # residual
```

FiLM (Feature-wise Linear Modulation) lets time and class **modulate** the features. At t=100, the time embedding says "amplify high-frequency channels." At t=900, it says "amplify low-frequency channels." Different timesteps get different feature emphases.

### Self-Attention

```python
# unet.py — SelfAttention
def forward(self, x):
    B, C, H, W = x.shape
    x = x.view(B, C, H*W).permute(0, 2, 1)   # [B, H*W, C]

    Q = x @ query_weight                      # query: "what am I looking for?"
    K = x @ key_weight                        # key: "what do I have?"
    V = x @ value_weight                      # value: "what do I contribute?"

    attention = softmax(Q @ K.T / √d)         # [B, H*W, H*W]
    out = attention @ V                        # [B, H*W, C]

    return out.permute(0, 2, 1).view(B, C, H, W)
```

In the bottleneck (8×17 or similar), every "pixel" can attend to every other "pixel." This catches long-range relationships: "This high-frequency band should be harmonically related to that low-frequency band."

---

## 5. Training — Standard DDPM on Real Spectrograms

**File:** `src/diffusion/train.py`

### 5.1 The Core Loop

```python
for mel, labels in train_loader:           # mel = [B, 1, 64, T], real spectrogram
    t = torch.randint(0, T, (B,))           # random timesteps for each sample
    noise = torch.randn_like(mel)           # ε ~ N(0, I)

    # Forward: add noise in one step (closed form)
    x_t = diffusion.q_sample(mel, t, noise)  # x_t = √ā_t·mel + √(1-ā_t)·noise

    # U-Net tries to predict the noise we added
    pred_noise = unet(x_t, t, labels)       # ε̂

    # Loss: how close was the prediction?
    loss = MSE(pred_noise, noise)            # smaller = better

    loss.backward()
    optimizer.step()
```

**Why predict NOISE instead of the clean image?**
- Predicting `ε` is a **residual task** — the net only needs to output small changes, not reconstruct the whole spectrogram
- Predict `x₀` directly = harder optimization landscape
- Same reason ResNets learn residuals, not full functions

### 5.2 q_sample (Forward Diffusion, One Step)

**File:** `src/diffusion/diffusion.py`

```python
def q_sample(self, x_start, t, noise):
    """
    x_start: [B, 1, 64, T] clean spectrogram
    t:       [B] timestep indices
    noise:   [B, 1, 64, T] standard normal

    Returns: x_t = √ā_t · x_start + √(1-ā_t) · noise
    """
    sqrt_alpha_cumprod = sqrt(self.alphas_cumprod[t])         # [B, 1, 1, 1]
    sqrt_one_minus_alpha_cumprod = sqrt(1 - self.alphas_cumprod[t])

    return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
```

**What ā_t does:**
```
t=0:    ā = 0.9999     → x ≈ x₀ (barely any noise)
t=500:  ā ≈ 0.5         → x = 0.71·x₀ + 0.71·noise (half signal, half noise)
t=999:  ā ≈ 0.0001      → x ≈ pure noise
```

### 5.3 Dataset — Real Spectrograms

**Class:** `DiffusionDataset` in `train.py`

```python
class DiffusionDataset(Dataset):
    def __getitem__(self, idx):
        wav, sr = load_audio(path)                              # [1, samples]

        # Smart crop on waveform FIRST (matches HiFi-GAN)
        crops = smart_crop(wav, crop_samples=segment_frames * hop_length,
                          threshold_db=-30.0, num_crops=1)
        wav = crops[0]

        # Compute mel on the energy-rich segment
        mel = compute_mel(wav)                                  # [1, 64, T]

        # Normalize to zero-mean unit-variance
        mel = (mel_db - -18.4903) / 19.8031                     # [1, 64, segment_frames]

        return mel, label
```

**Why `smart_crop` before mel?** Without it, random cropping could select silent portions of the audio → mel is all zeros → U-Net learns to predict zero noise on zeros → wasted training.

### 5.4 Class Labels

```python
CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']
```

8 animal classes. The class index is passed to the U-Net's embedding layer at each forward pass. This conditions the denoising — the model learns that "Dog" spectrograms have different patterns from "Cat" spectrograms.

---

## 6. Inference — Refining a VAE Output

**File:** `src/diffusion/inference.py`

### 6.1 `refine_spectrogram()` — The Core API

```python
def refine_spectrogram(mel, class_label, num_steps=50, strength=0.6):
    """
    mel:         [1, 1, 64, 552]  — VAE-generated spectrogram (normalized)
    class_label: int (0-7)        — which animal
    num_steps:   int              — DDIM sampling steps (50 = good quality, 20 = fast)
    strength:    float            — 0.0 = no change, 1.0 = full noise

    Returns: [1, 1, 64, 552] — refined spectrogram
    """
    # Calculate starting timestep
    start_t = int(strength * (T - 1))   # e.g., 0.6 * 999 = 599

    # Add noise to VAE output
    noise = torch.randn_like(mel)
    x = q_sample(mel, start_t, noise)   # x ≈ 0.6×VAE + 0.4×random

    # DDIM denoising: start_t → 0 in num_steps jumps
    timesteps = torch.linspace(start_t, 0, num_steps, dtype=torch.long)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        epsilon = unet(x, t, class_label)        # predict noise
        x = p_sample_ddim(x, t, t_next, epsilon)  # remove predicted noise

    return x  # sharpened spectrogram
```

### 6.2 `generate_refined()` — Full Pipeline

```python
def generate_refined(vae, class_name, num_steps=50, strength=0.6, temperature=0.7):
    """
    Full pipeline: VAE → Diffusion → spectrogram (ready for HiFi-GAN)

    Returns: (mel, mel_refined) — so you can compare with/without
    """
    # Step 1: VAE generates
    label_idx = CLASS_TO_IDX[class_name]
    mel = vae.sample(label_idx, temperature=temperature)      # [1, 1, 64, 552]

    # Step 2: Diffusion refines
    mel_refined = refine_spectrogram(mel, label_idx,
                                     num_steps=num_steps,
                                     strength=strength)

    return mel, mel_refined
```

### 6.3 Model Loading

The U-Net model is loaded once and cached:

```python
_model_cache = None

def _load_model():
    global _model_cache
    if _model_cache is None:
        # Load config
        model = SpectrogramUNet(config)

        # Load checkpoint
        ckpt = torch.load("models/diffusion_unet_train_best.pth", map_location='cpu')
        model.load_state_dict(ckpt["unet"])

        model.eval()
        _model_cache = model
    return _model_cache
```

---

## 7. Training Modes & Config

**File:** `src/diffusion/train.py` — `CONFIG` dict at top

```python
CONFIG = {
    "mode": "train",         # "test" = 5 epoch smoke test | "train" = full
    "device": "auto",        # "auto", "cuda", "mps", or "cpu"

    "data_dir": "data/animal_audio",
    "segment_frames": 552,   # ~5 seconds of mel frames
    "save_interval": 5,      # epochs between checkpoints

    "test": {
        "num_epochs": 5,
        "batch_size": 4,
        "num_workers": 0,
    },

    "train": {
        "num_epochs": 50,
        "batch_size": 1,
        "num_workers": 0,
    },
}
```

### How to Run

```bash
# Edit CONFIG["mode"] to "train" or "test", then:
python src/diffusion/train.py
```

### What to Watch

```
── Epoch   1/50 (45s) ── loss=1.0934 val=0.9821 📉 lr=1.00e-03
── Epoch   5/50 (44s) ── loss=0.3120 val=0.2890 📉 lr=9.60e-04
── Epoch  10/50 (44s) ── loss=0.0891 val=0.0823 📉 lr=9.10e-04
── Epoch  30/50 (45s) ── loss=0.0234 val=0.0251 📉 lr=7.50e-04
── Epoch  50/50 (44s) ── loss=0.0120 val=0.0150 ➡️ lr=6.30e-04
```

- **loss** = training MSE between predicted and actual noise
- **val** = validation MSE
- **📉** = validation improved (new best model saved)
- **➡️** = no improvement

### Interpret the Loss Values

```
Loss > 1.0:      U-Net guessing randomly (early epochs)
Loss 0.3-0.5:    Learning basic spectral patterns
Loss 0.05-0.15:  Good denoising capability
Loss < 0.02:     Excellent — near-perfect noise prediction
```

### Models Saved

| Model | Path | When |
|-------|------|------|
| Best | `models/diffusion_unet_train_best.pth` | When val loss improves |
| Checkpoints | `models/diffusion_checkpoints/train/unet_000005.pth` | Every 5 epochs |
| Final | `models/diffusion_unet_train.pth` | After last epoch |

---

## 8. Critical Hyperparameters

**File:** `src/diffusion/config.py` — `DiffusionConfig`

| Parameter | Default | Controls | If wrong |
|-----------|---------|----------|----------|
| `timesteps` | 1000 | Noise granularity | 2000 = slower training, minimal gain |
| `beta_start` | 0.0001 | Min noise per step | Higher → too much noise early, U-Net can't learn fine detail |
| `beta_end` | 0.02 | Max noise per step | Lower → not enough noise late, U-Net can't learn coarse structure |
| `cosine_s` | 0.008 | Schedule smoothness | Affects noise distribution uniformity |
| `base_channels` | 64 | U-Net starting channels | 32 = too small (underfit); 128 = too large (overfit, slow) |
| `channel_multipliers` | (1,2,3,3) | Encoder/decoder depths | (1,2,4,4) = 25.8M params, (1,2,3,3) = 17.8M |
| `res_blocks_per_level` | 2 | Blocks per U-Net level | 1 = weaker but faster; 2 = standard |
| `attention_levels` | (2,3) | Which levels have self-attention | () = no attention, misses long-range patterns |
| `time_emb_dim` | 256 | Timestep embedding size | 64 = too small; 512 = unnecessary |
| `class_emb_dim` | 64 | Animal class embedding size | 16 = underfit class differences |
| `learning_rate` | 1e-3 | Step size | 1e-4 = too slow; 5e-3 = unstable |
| `lr_decay` | 0.998 | Per-epoch LR multiplier | 0.99 = decays too fast |
| `segment_frames` | 552 | Mel time frames (~5s) | 276 = less context per sample |
| `num_inference_steps` | 50 | DDIM steps at inference | 100 = slower but slightly better; 20 = fast but may artifact |

### Model Size Options

```
channel_multipliers = (1, 2, 4, 4)    → 25.8M params (full)
channel_multipliers = (1, 2, 3, 3)    → 17.8M params (~31% smaller, good balance)
```

Reduce `channel_multipliers` if your GPU runs hot or overfits. The deeper levels (3,4) have the most channels — reducing them saves the most parameters.

---

## 9. Quick File Reference

| What you want | File | Function/Class | Approx line |
|---------------|------|----------------|-------------|
| All hyperparameters | `config.py` | `DiffusionConfig` | ~1 |
| U-Net model | `unet.py` | `SpectrogramUNet` | ~1 |
| U-Net ResBlock | `unet.py` | `ResBlock` | ~70 |
| Time embedding | `unet.py` | `SinusoidalTimeEmbedding` | ~30 |
| Self-attention | `unet.py` | `SelfAttention` | ~145 |
| Diffusion process | `diffusion.py` | `DiffusionProcess` | ~30 |
| Noise schedule | `diffusion.py` | `_cosine_schedule()` | ~80 |
| Forward (q_sample) | `diffusion.py` | `q_sample()` | ~120 |
| DDPM reverse (p_sample) | `diffusion.py` | `p_sample()` | ~160 |
| DDIM reverse | `diffusion.py` | `p_sample_ddim()` | ~200 |
| DDIM sampling loop | `diffusion.py` | `ddpm_sampling()` | ~250 |
| img2img refine | `diffusion.py` | `refine()` | ~300 |
| Dataset (with smart_crop) | `train.py` | `DiffusionDataset` | ~161 |
| Mel computation | `train.py` | `compute_mel()` | ~241 |
| Training one epoch | `train.py` | `train_epoch()` | ~295 |
| Validation | `train.py` | `validate()` | ~348 |
| Full training loop | `train.py` | `training_loop()` | ~383 |
| Config dict (edit this) | `train.py` | `CONFIG = {...}` | ~56 |
| Refine one spectrogram | `inference.py` | `refine_spectrogram()` | ~80 |
| Full pipeline (VAE+Diff) | `inference.py` | `generate_refined()` | ~180 |
| Model loader (cached) | `inference.py` | `_load_model()` | ~50 |

---

## 10. How Diffusion and HiFi-GAN Compare

| Aspect | HiFi-GAN (7a) | Diffusion (7b) |
|--------|--------------|----------------|
| **What it does** | Converts mel → audio | Sharpens blurry mel → sharp mel |
| **Input** | Mel spectrogram [1, 64, T] | Mel spectrogram [1, 64, T] + noise |
| **Output** | Audio waveform [1, T×200] | Denoised mel spectrogram [1, 64, T] |
| **Architecture** | 1D ConvTranspose + MRF blocks | 2D U-Net with attention |
| **Training** | GAN (generator vs discriminator) | DDPM (predict noise, MSE loss) |
| **Training data** | Raw audio → mel | Raw audio → mel (same) |
| **Two-phase?** | Yes (meltrain → GAN) | No (single phase) |
| **Discriminator** | Yes (PeriodDiscriminator) | No |
| **Loss** | Mel L1 + Feature Matching + Adversarial | Simple MSE on noise |
| **Speed (inference)** | ~30-100ms | ~2-3s (50 steps) |
| **Optional?** | No — needed for audio output | Yes — toggle in pipeline |

Both use the same **mel computation** (MelSpectrogram + AmplitudeToDB + same normalization) and **data loading** patterns (smart_crop, class labels).

---

*Built from actual code in `src/diffusion/`. Line numbers approximate — verify against source.*
