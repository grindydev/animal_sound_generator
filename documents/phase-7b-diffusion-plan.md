# Phase 7b — Diffusion Refinement: Implementation Plan

> **One-line summary:** Train a small U-Net diffusion model that takes the VAE's blurry spectrogram and denoises it into a sharp, realistic spectrogram — then send that to HiFi-GAN.

---

## 1. Why Phase 7b AFTER Phase 7a (HiFi-GAN)

### What 7a (HiFi-GAN) fixes:

HiFi-GAN solves the **conversion problem**: it turns a mel spectrogram into a crisp audio waveform. Griffin-Lim sounds robotic; HiFi-GAN sounds natural.

```
Without HiFi-GAN:                    With HiFi-GAN (7a):
┌──────────┐                         ┌──────────┐
│   VAE    │                         │   VAE    │
└────┬─────┘                         └────┬─────┘
     ↓                                   ↓
[blurry mel]                        [blurry mel]
     ↓                                   ↓
Griffin-Lim                          HiFi-GAN
     ↓                                   ↓
"robot underwater"                   "natural but blurry"
     (grainy, metallic)                  (clean but unfocused)
```

### What 7a does NOT fix:

HiFi-GAN is a **faithful converter**. It converts whatever mel you give it. If the mel spectrogram is blurry, the audio is crisp-but-blurry — like a sharp photo of a blurry object.

**The blur comes from the VAE decoder.** VAEs average over possibilities. When you ask for "dog bark," the VAE samples from a distribution and the decoder smooths out the details:

```
Real dog bark spectrogram:           VAE-generated spectrogram:
  sharp edges                          blurred edges
  crisp harmonics                      washed-out harmonics
  clear onsets/offsets                 smeared onsets/offsets
  
  ██████░░░░                           ▓▓▓▓▓▓░░░░
  ██░░░░░░░░                           ▓▓▓░░░░░░░
  ░░████░░░░                           ░░▓▓▓▓░░░░
```

### What 7b (Diffusion) fixes:

Diffusion is a **refiner**. It takes the blurry VAE output and iteratively sharpens it, restoring edges, harmonics, and detail.

```
Full pipeline WITH 7b:
═══════════════════════════════════════════════════════════════

"dog" label
    ↓
┌──────────┐
│   VAE    │  ← generates rough spectrogram (blurry)
└────┬─────┘
     ↓
[blurry mel, 1, 64, 552]
     ↓
┌──────────────┐
│  Diffusion   │  ← U-Net denoises / sharpens
│  (Phase 7b)  │    restores edges, harmonics, detail
└────┬─────────┘
     ↓
[sharp mel, 1, 64, 552]
     ↓
┌──────────────┐
│  HiFi-GAN    │  ← converts to crisp waveform
│  (Phase 7a)  │
└────┬─────────┘
     ↓
[crisp AND sharp audio] ✨

═══════════════════════════════════════════════════════════════
```

**Why after 7a?** Because you need the final converter (HiFi-GAN) working FIRST so you can actually LISTEN to the improvement. Without HiFi-GAN, you'd be comparing Griffin-Lim outputs — which are so grainy you can't tell if diffusion helped. HiFi-GAN isolates the "spectrogram quality" variable.

---

## 2. What You Will Build

### 2.1 New Files

| File | Purpose |
|------|---------|
| `src/diffusion/unet.py` | U-Net architecture for 2D spectrograms |
| `src/diffusion/diffusion.py` | Forward/reverse diffusion process, noise schedule |
| `src/diffusion/train.py` | Training loop (VAE output → diffusion → real mel) |
| `src/diffusion/inference.py` | Sampling loop: take VAE output, add noise, denoise N steps |
| `src/diffusion/config.py` | Hyperparameters (timesteps, beta schedule, model size) |
| `src/diffusion/__init__.py` | Package init |

### 2.2 Modified Files

| File | Change |
|------|--------|
| `src/vae.py` | Add `generate_with_diffusion()` method that chains VAE → Diffusion → output |
| `client/server.py` | Add "Diffusion refinement" toggle + steps slider |
| `documents/phase-7b-diffusion-plan.md` | This document → final docs after build |

---

## 3. Architecture: U-Net for Spectrograms

**Input:** spectrogram `[B, 1, 64, W]` (same shape as VAE output)  
**Output:** predicted noise `[B, 1, 64, W]` (same shape)  
**Conditioning:** animal class label → embedding → injected at each U-Net layer

### Why U-Net?

Same as Stable Diffusion. The U-Net has:
- **Encoder:** downsamples spectrogram, extracts features at multiple scales
- **Bottleneck:** compressed representation
- **Decoder:** upsamples back, with **skip connections** from encoder

Skip connections preserve spatial detail — critical for restoring sharp edges in spectrograms.

```
Input: [B, 1, 64, W] spectrogram at timestep t
           │
           ▼
    ┌─────────────┐
    │ Time embed  │  ← sinusoidal embedding of timestep t
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Class embed │  ← animal class (dog/cat/...) as embedding
    └─────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │           U-NET ENCODER                │
    │                                          │
    │  [64, W] → Conv+ResBlock → [32, W/2]   │
    │       ↓ skip connection                  │
    │  [32, W/2] → Conv+ResBlock → [16, W/4] │
    │       ↓ skip connection                  │
    │  [16, W/4] → Conv+ResBlock → [8, W/8]  │
    └─────────────────────────────────────────┘
           │
           ▼  [8, W/8] bottleneck
           │
    ┌─────────────────────────────────────────┐
    │           U-NET DECODER                │
    │                                          │
    │  [8, W/8] → ConvTranspose+ResBlock →   │
    │           [16, W/4] + skip from encoder │
    │       ↓                                  │
    │  [16, W/4] → ConvTranspose+ResBlock →  │
    │           [32, W/2] + skip from encoder │
    │       ↓                                  │
    │  [32, W/2] → ConvTranspose+ResBlock →  │
    │           [64, W] + skip from encoder   │
    └─────────────────────────────────────────┘
           │
           ▼
Output: [B, 1, 64, W] predicted noise
```

### ResBlock with Class + Time Conditioning

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, class_emb_dim):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Time embedding projected to channel space
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        # Class embedding projected to channel space
        self.class_mlp = nn.Linear(class_emb_dim, out_ch)
        
    def forward(self, x, t_emb, c_emb):
        h = self.conv1(F.silu(x))
        # Add time and class conditioning
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.class_mlp(c_emb)[:, :, None, None]
        h = self.conv2(F.silu(h))
        return x + h   # residual connection
```

### Attention Layers (optional, at bottleneck)

Small self-attention at the bottleneck helps capture long-range frequency relationships (e.g., harmonics that span many frequency bins). For 64×552 spectrograms, even 1-2 attention blocks help.

---

## 4. Training Approach

### 4.1 The Core Idea

Train the U-Net to predict noise. At inference, use it to remove noise from a blurry starting point.

**Training data:** You need pairs of (blurry VAE output, sharp real spectrogram).

**How to get training pairs:**
1. Load a REAL audio clip
2. Compute its REAL mel spectrogram → `x_0` (the clean target)
3. Pass the same audio through the VAE encoder + decoder → `x_vae` (the blurry input)
4. The diffusion model learns: given `x_vae` + noise, predict the noise and recover something close to `x_0`

Wait — actually there's a simpler and more standard approach:

### 4.2 Simpler Approach: Standard DDPM on Real Spectrograms

Instead of training specifically VAE→real, train a standard diffusion model on REAL spectrograms:

```
Training loop:
  1. Load real audio → compute real mel spectrogram → x_0
  2. Pick random timestep t (0 to T-1)
  3. Add noise to x_0: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
  4. U-Net(x_t, t, class_label) → predicted_noise
  5. Loss = MSE(predicted_noise, actual_noise)
  6. Backprop, update U-Net
```

At inference (refinement mode):
```
  1. VAE generates blurry spectrogram → x_vae
  2. Add SMALL amount of noise (e.g., t=0.3*T)
  3. Run reverse diffusion for N steps (e.g., 20-50)
  4. Output: sharpened spectrogram
```

This is called **"img2img"** in Stable Diffusion. You start from an existing image, add noise, and denoise. The amount of initial noise controls how much the output changes.

### 4.3 Why Standard DDPM is Better

- **Simpler training:** No need to run VAE encoder/decoder during training. Just use real spectrograms.
- **More general:** The model learns the manifold of real spectrograms, not just VAE→real mapping.
- **Flexible inference:** You can control how much refinement happens by choosing how many denoising steps.
- **Works with ANY input:** Not just VAE output. Could refine autoencoder outputs, interpolations, etc.

### 4.4 Training Data Pipeline

```python
# In train.py data loader

for audio in train_loader:              # [B, 1, 16384] real audio
    # Compute real mel spectrogram
    real_mel = compute_mel(audio)       # [B, 64, T]
    real_mel = real_mel.unsqueeze(1)    # [B, 1, 64, T]
    
    # Get class labels (from dataset — same as classifier)
    labels = ...                        # [B] — dog=0, cat=1, etc.
    
    # Diffusion forward: add noise
    t = torch.randint(0, T, (B,))       # random timesteps
    noise = torch.randn_like(real_mel)  # Gaussian noise
    x_t = diffusion.q_sample(real_mel, t, noise)  # x_t = sqrt(a)*x_0 + sqrt(1-a)*noise
    
    # U-Net predicts the noise
    pred_noise = unet(x_t, t, labels)
    
    # Loss: how well did we predict the noise?
    loss = F.mse_loss(pred_noise, noise)
    
    loss.backward()
    optimizer.step()
```

### 4.5 Noise Schedule (Beta Schedule)

Use cosine schedule (better than linear for small T):

```python
# config.py
timesteps = 1000
beta_start = 0.0001
beta_end = 0.02

# Cosine schedule (from Improved DDPM)
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

For faster inference, you can train with 1000 steps but sample with 50-100 steps using DDIM.

---

## 5. Inference: Refining a VAE Output

### 5.1 The Refinement Process

```python
def refine_spectrogram(vae_output, class_label, num_steps=50, strength=0.7):
    """
    vae_output: [1, 1, 64, 552] — blurry VAE spectrogram
    class_label: int — which animal (for conditioning)
    num_steps: how many denoising steps (more = sharper but slower)
    strength: how much noise to add initially (0.0 = no change, 1.0 = full noise)
    
    Returns: [1, 1, 64, 552] — sharpened spectrogram
    """
    
    # Step 1: Add noise to VAE output
    # strength=0.7 → start at timestep t = 0.7 * T
    start_t = int(strength * (T - 1))
    noise = torch.randn_like(vae_output)
    x = diffusion.q_sample(vae_output, start_t, noise)
    
    # Step 2: Denoise step by step
    for t in reversed(range(start_t)):
        t_batch = torch.full((1,), t, device=device)
        
        # U-Net predicts noise
        pred_noise = unet(x, t_batch, class_label)
        
        # Remove predicted noise (DDPM sampling step)
        x = diffusion.p_sample(x, t, pred_noise)
    
    return x
```

### 5.2 Full Pipeline at Inference

```python
# Step 1: VAE generates rough spectrogram
label = "dog"
vae_mel = vae.sample(label, num_samples=1, temperature=0.7)   # [1, 1, 64, 552]

# Step 2: Diffusion refines it (OPTIONAL — toggle in UI)
if use_diffusion:
    refined_mel = diffusion.refine(vae_mel, label, num_steps=30, strength=0.6)
    # sharper edges, clearer harmonics
else:
    refined_mel = vae_mel  # skip refinement

# Step 3: HiFi-GAN converts to audio
audio = hifigan(refined_mel.squeeze(1))   # [1, 1, 110400] for 5 sec

# Step 4: Save
import torchaudio
torchaudio.save("dog_bark_refined.wav", audio.squeeze(0), 22050)
```

### 5.3 What `strength` Controls

| Strength | Noise added | Effect | Use case |
|----------|-------------|--------|----------|
| 0.0 | None | No change from VAE output | Off |
| 0.3 | Small | Light polish, preserves structure | Subtle improvement |
| 0.6 | Medium | Good balance of refinement + preservation | **Recommended** |
| 0.9 | Heavy | Major change, less VAE structure | Creative variation |
| 1.0 | Full noise | Pure diffusion from scratch | Not useful for refinement |

---

## 6. Training Config & Mode System

Following the same pattern as `src/hifigan/train.py`, the diffusion training script uses a `CONFIG` dict with **test/train modes**, **device auto-detection**, and **progress tracking**.

### 6.1 CONFIG Dict (in `src/diffusion/train.py`)

```python
CONFIG = {
    "mode": "train",         # "test" | "train"
    "device": "auto",        # "auto", "cuda", "mps", or "cpu"

    "data_dir": "data/animal_audio",
    "mel_dir": "data/animal_mel",
    "model_dir": "models",
    "save_interval": 5,      # epochs between checkpoints

    "test": {
        "num_epochs": 5,
        "batch_size": 4,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 50,
        "batch_size": 8,
        "num_workers": 0,
    },
}
```

### 6.2 Mode Behavior

| Mode | Epochs | Batch | Purpose |
|------|--------|-------|---------|
| `test` | 5 | 4 | Quick smoke test — verify loss drops, no crashes |
| `train` | 50 | 8 | Full training run |

### 6.3 Device Auto-Detection

```python
if CONFIG["device"] == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        is_cuda = True
        print("🚀 Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        is_cuda = False
        print("🍎 Using MPS")
    else:
        device = torch.device("cpu")
        is_cuda = False
        print("⚠️  Using CPU")
```

### 6.4 Progress Tracking (per epoch)

Each epoch prints a one-line summary matching the HiFi-GAN style:

```
── Epoch   3/50 (45s) ── loss=0.0234 val=0.0251 📉 lr=1.85e-04
── Epoch   4/50 (44s) ── loss=0.0198 val=0.0223 📉 lr=1.83e-04
── Epoch   5/50 (45s) ── loss=0.0182 val=0.0210 📉 lr=1.82e-04
```

- 📉 = validation improved (new best)
- ➡️ = no improvement
- `lr` = current learning rate
- `time` = epoch wall-clock time

### 6.5 Checkpoint & Resume

```python
# Save checkpoint (generator state + optimizer state + epoch)
save_checkpoint(unet, optimizer, epoch, CHECKPOINT_DIR)

# Resume from checkpoint
start_epoch = load_checkpoint(unet, optimizer, CHECKPOINT_DIR, device)
```

- Checkpoints saved every `save_interval` epochs
- Best model saved separately when validation loss improves
- Final model saved at end of training

### 6.6 Banner Print

On startup, print a banner with key config info:

```
🔧 Diffusion Refinement → TRAIN MODE
   GPU:    Apple M1 Pro
   Epochs: 50 | Batch: 8 | Workers: 0
   Timesteps: 1000 | Model: ~20M params
   Mixed precision: no
   Best model → models/diffusion_unet_best.pth
```

### 6.7 Data Validation

Before training starts, load one batch and verify:

```python
_check = next(iter(train_loader))
print(f"   📊 First batch: shape={tuple(_check[0].shape)}, "
      f"mel min={_check[0].min():.4f}, max={_check[0].max():.4f}")
if _check[0].abs().max() < 1e-6:
    print("   🚨 ALL ZEROS — mel data not loading correctly.")
    return None
```

This catches silent data loading bugs before wasting training time.

---

## 7. Model Size & Speed Estimates

### U-Net Size

For 64×552 spectrograms, a small U-Net is sufficient:

```
Base channels: 64
Channel multipliers: [1, 2, 4, 4] → [64, 128, 256, 256]
Attention at: bottleneck (16×69) and one encoder level (32×138)
ResBlocks per level: 2

Total parameters: ~15-25M (much smaller than Stable Diffusion's ~800M)
```

### Speed Estimates (MPS / CUDA)

| Step | Time (M1 Pro MPS) | Time (RTX 3060) |
|------|-------------------|-----------------|
| VAE generate | 50ms | 20ms |
| Diffusion (50 steps) | 2-3s | 0.5-1s |
| HiFi-GAN | 100ms | 30ms |
| **Total with diffusion** | **~3s** | **~1s** |
| Total without diffusion | ~150ms | ~50ms |

Diffusion is slower but optional. The UI will have a toggle.

---

## 8. Implementation Checklist

### Week 1: Architecture
- [ ] `src/diffusion/unet.py` — U-Net with time + class conditioning
- [ ] `src/diffusion/diffusion.py` — forward/reverse process, noise schedule
- [ ] `src/diffusion/config.py` — hyperparameters
- [ ] Test: forward diffusion adds correct noise, reverse removes it

### Week 2: Training
- [ ] `src/diffusion/train.py` — training loop on real spectrograms
- [ ] Data loader: reuse `compute_mel()` from HiFi-GAN
- [ ] Class labels: reuse FSD50K metadata
- [ ] Train for 50-100 epochs (small model, fast)
- [ ] Monitor: MSE loss should drop to ~0.01-0.05

### Week 3: Inference & Integration
- [ ] `src/diffusion/inference.py` — DDPM/DDIM sampling, refinement function
- [ ] `src/vae.py` — add `generate_with_diffusion()` helper
- [ ] Test: VAE output → diffusion → HiFi-GAN, compare with/without
- [ ] A/B listening test: does refined audio sound sharper?

### Week 4: UI & Docs
- [ ] `client/server.py` — add diffusion toggle + steps slider + strength slider
- [ ] `documents/phase-7b-diffusion.md` — final documentation
- [ ] Update `roadmap.md` — mark 7b as complete

---

## 9. Key Design Decisions

### Decision 1: Train on real spectrograms, not VAE→real pairs

**Chosen:** Standard DDPM on real spectrograms. At inference, use img2img refinement (add noise to VAE output, denoise).

**Why:** Simpler training pipeline. More general. Works with any input (not just VAE).

**Alternative rejected:** Train specifically VAE→real mapping. More complex, less flexible.

### Decision 2: Operate on spectrograms directly, not latent space

**Chosen:** Pixel-space diffusion on [1, 64, W] spectrograms.

**Why:** Simpler. No need to train a separate VAE encoder/decoder for latent diffusion. Spectrogram dimensions are small enough (64×552 ≈ 35K pixels) that pixel-space is fast.

**Alternative:** Latent diffusion (like Stable Diffusion). Would need another encoder/decoder. Overkill for this project.

### Decision 3: Small U-Net (~20M params), not big

**Chosen:** ~20M parameters, 3-4 resolution levels.

**Why:** Spectrograms are small (64×552). A big model would overfit on ~3K animal clips. Small model trains faster and generalizes better.

### Decision 4: Class conditioning via embedding injection

**Chosen:** Animal class → embedding → added to ResBlocks.

**Why:** Simple, effective. Same pattern as VAE class conditioning. Guides denoising toward the correct animal's spectral characteristics.

**Alternative rejected:** Classifier-free guidance (more complex, needs training with random label dropout). Can add later if needed.

---

## 10. How to Evaluate Success

| Metric | Without Diffusion | With Diffusion | Target |
|--------|-------------------|----------------|--------|
| Fréchet Audio Distance (FAD) | Higher | Lower | FAD ↓ by 10-20% |
| Classification agreement | Lower | Higher | Agreement ↑ by 5-15% |
| Human listening test | "blurry" | "sharper" | 70%+ prefer with diffusion |
| Spectrogram sharpness | smeared edges | crisp edges | Edge clarity ↑ |

**Simplest test:** Generate 10 dog barks with and without diffusion. Listen. The diffused ones should have clearer attack transients and more defined harmonics.

---

## 11. Files to Create (Skeleton)

```
src/diffusion/
├── __init__.py
├── config.py          # timesteps, beta schedule, model size
├── unet.py            # U-Net with time + class conditioning
├── diffusion.py       # Forward/reverse process, q_sample, p_sample
├── train.py           # Training loop
└── inference.py       # Refinement function, DDIM sampling
```

---

## Summary

**What 7b does:** Takes the VAE's blurry spectrogram and sharpens it using a small diffusion U-Net.

**Why after 7a:** HiFi-GAN (7a) gives you a clean audio converter. Now you can isolate and improve the spectrogram quality before conversion.

**The full quality pipeline:**
```
VAE (generates) → Diffusion (sharpens) → HiFi-GAN (converts) → CRISP AUDIO
```

**Effort:** ~2-3 weeks. Small model (~20M params). Standard DDPM. Optional at inference (toggle in UI).
