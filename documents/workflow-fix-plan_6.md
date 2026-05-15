# Workflow Fix Plan v6 — Latent Diffusion (Path B)

> **Date:** May 14, 2026  
> **Status:** Implementing.  
> **Replaces:** Previous v6 (direct diffusion — failed).  
> **Approach:** Industry-standard latent diffusion. Diffuse in compressed 2D latent space instead of raw mel.

---

## 0. Why Direct Diffusion Failed (v4-v6)

2700 samples × 64×552 mel spectrogram = 35,328 values to predict.  
With 7,300 params per sample, the model converges to predicting zero noise — the "safe" strategy.

DDIM amplifies errors 160× at high t (dividing by √α≈0.006).  
DDPM needs accurate noise predictions — can't work when model predicts zero.

## 1. Why Latent Diffusion Works (Industry Standard)

Every major audio model uses this architecture:

```
RAW DATA → Encoder → Compressed Latent → Diffuse → Decoder → Output
                        ↑
                  (small 2D grid)
```

| Model | Latent size | Raw size | Compression |
|-------|:----------:|:--------:|:-----------:|
| Stable Audio | 16×20×86 | 128×344 | 43× |
| AudioLDM 2 | 8×16×256 | 64×512 | 256× |
| **Our Path B** | **16×4×35** | **1×64×552** | **16×** |

## 2. Architecture

### 2.1 Encoder (FROZEN — already trained)

```
mel [1, 64, 552]
    │
    ▼
ImprovedAutoencoder.encode()
    enc1→enc2→enc3→enc4  (149M params, MSE=0.015)
    │
    ▼
Spatial features [512, 4, 35]  ← bottleneck before fc_encode
```

### 2.2 Latent Space (WHERE DIFFUSION HAPPENS)

```
[512, 4, 35]  →  conv_reduce (512→16)  →  [16, 4, 35]
                                               │
                                     2,240 values (16× less than 35,328)
                                               │
                                     ★ Tiny UNet diffuses here ★
                                               │
[16, 4, 35]   ←  conv_expand (16→512) ←  [512, 4, 35]
```

### 2.3 Small Decoder (NEW — 2M params, no skip connections)

```
[512, 4, 35]
    │
    ▼
Block1:  Upsample 2× + Conv(512→256)  →  [256, 8, 70]
Block2:  Upsample 2× + Conv(256→128)  →  [128, 16, 140]
Block3:  Upsample 2× + Conv(128→64)   →  [64, 32, 280]
Block4:  Upsample 2× + Conv(64→32)    →  [32, 64, 560]
    │
    ▼
Conv(32→1) + Interpolate(552)  →  [1, 64, 552]
```

No skip connections. Pure upsampling. By design.

### 2.4 Tiny Diffusion UNet (~3M params)

```
Input: [B, 16, 4, 35] + timestep + class label
3 encoder levels: 16→64→128→256
Attention at bottleneck
3 decoder levels: 256→128→64→16
Output: predicted noise [B, 16, 4, 35]
```

## 3. Training Phases

### Phase 1: Train Decoder + Channel Reducers (~30 min)

Freeze encoder. Train `conv_reduce + decoder + conv_expand` to reconstruct mel.

```python
for mel_batch in data:
    with torch.no_grad():
        features = encoder.encode(mel_batch)  # [B, 512, 4, 35]
    
    latent = conv_reduce(features)       # [B, 16, 4, 35]
    expanded = conv_expand(latent)        # [B, 512, 4, 35]
    output = decoder(expanded)            # [B, 1, 64, 552]
    
    loss = MSE(output, mel_batch)
    loss.backward()
```

Target: MSE ~0.02 (slightly worse than autoencoder's 0.015, acceptable trade-off for generation capability).

### Phase 2: Train Diffusion UNet (~1 hr)

Freeze encoder + decoder + reducers. Train UNet on real latents.

```python
for mel_batch, labels in data:
    with torch.no_grad():
        features = encoder.encode(mel_batch)
        latent = conv_reduce(features)   # [B, 16, 4, 35]
    
    t = random_timestep()
    noise = randn_like(latent)
    x_t = q_sample(latent, t, noise)
    pred_noise = unet(x_t, t, labels)
    
    loss = L2(pred_noise, noise)
    loss.backward()
```

Target: val_loss < 0.05.

## 4. Inference

```python
# Generate Dog sound:
label = 0  # Dog
noise = randn([1, 16, 4, 35])
latent = unet.ddim_sample(noise, label, steps=100)
features = conv_expand(latent)     # [1, 512, 4, 35]
mel = decoder(features)            # [1, 1, 64, 552]
audio = hifigan(mel)               # [1, 110400]
```

## 5. Files Changes

| File | Status | Content |
|------|:------:|---------|
| `src/latent_diff/__init__.py` | NEW | Package init |
| `src/latent_diff/config.py` | NEW | Latent + decoder + UNet config |
| `src/latent_diff/decoder.py` | NEW | Small upsampling decoder (2M) |
| `src/latent_diff/unet.py` | NEW | Tiny UNet for 16ch×4×35 (3M) |
| `src/latent_diff/train_decoder.py` | NEW | Train conv_reduce + decoder |
| `src/latent_diff/train_diff.py` | NEW | Train diffusion UNet on latents |
| `src/latent_diff/generate.py` | NEW | Full generation pipeline |
| `src/latent_diff/dataset.py` | NEW | Dataset that encodes mels to latents |
| `src/diffusion/diffusion.py` | REUSE | q_sample, DDPM, DDIM |
| `src/vae/autoencoder.py` | REUSE | Encoder (frozen) |
| `src/data_loader.py` | REUSE | Audio loading, mel computation |
| `src/hifigan/` | REUSE | HiFi-GAN inference |

## 6. Success Criteria

| Metric | Target |
|--------|:------:|
| Decoder MSE | < 0.03 |
| Diffusion val loss | < 0.05 |
| Generated mel ≠ noise | Peak < 5000 Hz |
| Classes distinguishable | ≥ 3/8 classes |
| Audio RMS | 0.05-0.2 |
