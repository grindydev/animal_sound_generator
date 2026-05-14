# Workflow Fix Plan v4 — Diffusion Direct Generation (Path B)

> **Date:** May 13, 2026  
> **Status:** Implemented. Training pending on Colab.  
> **Previous plans:** v1-v3 all attempted to fix VAE decoder for generation — failed.  
> **New approach:** Cut out VAE entirely. Diffusion UNet generates mel directly from noise.

---

## 0. Why VAE Failed (4 attempts)

| Attempt | Approach | Result |
|---------|----------|--------|
| v1 | skip_dropout=0.5 (original training) | White noise |
| v2 | skip_dropout=1.0 + embed_dim=256 | val_mse exploded |
| v3 | skip_dropout=1.0 + gen_attn (replacement) | val_mse=1500, KL=4M |
| v3b | skip_dropout=1.0 + gen_attn (residual) | val_mse stuck at 4.6 |

**Root cause confirmed:** FiLMDecoderStage architecture requires encoder skip connections. The decoder was designed around receiving `concat(h, enc_skip)` at every level. Without those 2× channels, it cannot produce coherent spectrograms, regardless of training strategy.

---

## 1. Path B Architecture

### 1.1 Pipeline

```
TRAINING                           INFERENCE
========                           =========

Real audio .wav                    "Dog" label
    │                                   │
    ▼                                   ▼
Mel spectrogram [1, 64, 552]       Random noise [1, 64, 552]
(AmplitudeToDB + normalize)             │
    │                                   ▼
    ▼                            ┌──────────────┐
┌──────────────┐                 │ Diffusion UNet │
│ Forward noise │                 │  50 DDIM steps │
│  x_t = α·x₀  │                 │  Class "Dog"   │
│      + β·ε   │                 └──────┬─────────┘
└──────┬───────┘                        │
       │                                ▼
       ▼                         Mel spectrogram
┌──────────────┐                 [1, 64, 552]
│ Diffusion UNet │                     │
│  predict ε     │                     ▼
│  Class "Dog"   │              ┌──────────────┐
└──────┬─────────┘              │   HiFi-GAN    │
       │                        │  mel → audio  │
       ▼                        └──────┬─────────┘
  MSE(pred_ε, ε)                       │
                                       ▼
                                  Audio .wav
```

### 1.2 No VAE Needed

The VAE becomes optional — kept for style transfer experiments, not generation. Models used:

| Model | Role | Retrain? |
|-------|------|:---:|
| **Diffusion UNet** (120M) | Primary generator | ✅ YES |
| **HiFi-GAN** (3M) | Mel → audio converter | ❌ Done |
| Autoencoder (149M) | Compression/analysis | ❌ Done |
| Classifier (457K) | Evaluation | ❌ Done |
| VAE (223M) | Unused | ❌ Optional |

### 1.3 Scaled UNet Architecture

| Parameter | Old (refinement) | New (generation) | Why |
|-----------|:---:|:---:|-----|
| base_channels | 64 | **128** | 2× wider |
| channel_multipliers | (1,2,3,3) | **(1,2,4,4)** | Deeper bottleneck |
| res_blocks_per_level | 2 | **3** | More compute per resolution |
| attention_levels | (2,3) | **(0,1,2,3)** | Attention at ALL levels |
| time_emb_dim | 256 | **512** | Better noise prediction |
| class_emb_dim | 64 | **256** | Stronger conditioning |
| num_heads | 4 | **8** | More attention heads |
| **Total params** | **18M** | **~120M** | 6.7× bigger |
| Training data | VAE mels (blurry) | **Real mels** | Learns true distribution |

### 1.4 Normalization Consistency

All components use the same normalization:
```
norm_mean = -18.4903
norm_std  = 19.8031
```
- `data_loader.py` SimpleNormalize: `(x - mean) / std`
- `diffusion/train.py` _load_mel: `(db - mean) / std`
- `hifigan/config.py`: `norm_mean`, `norm_std`
- `hifigan/inference.py`: `mel * std + mean` (unnormalize)

---

## 2. Files Changed

| File | Change |
|------|--------|
| `src/diffusion/config.py` | Scaled UNet: base_ch=128, mults=(1,2,4,4), attention at all levels |
| `src/diffusion/unet.py` | Added encoder/decoder attention blocks, 8 heads |
| `src/diffusion/train.py` | vae_mix_ratio=0, batch=8, workers=4 |
| `src/diffusion/inference.py` | Added `generate_from_noise()` for pure DDPM generation |
| `src/generate.py` | Added `--from-scratch` flag, imports `generate_from_noise` |

### What was NOT changed
- HiFi-GAN (config, generator, inference)
- data_loader / SimpleNormalize (correct values confirmed)
- Autoencoder (unchanged, MSE=0.015)
- Classifier (unchanged, 95.3%)

---

## 3. Training (Colab L4)

```
Session: Diffusion UNet (~3-4 hours)

Config:
  mode: train
  batch_size: 8 × grad_accum=2 = effective 16
  epochs: 50
  lr: 2e-4 (AdamW)
  data: 5171 train / 1292 val (real mel spectrograms)
  model: 120M params
```

---

## 4. Usage After Training

```bash
# Pure generation from noise
python src/generate.py --label Dog --from-scratch --steps 50

# Multiple samples
python src/generate.py --label Cat --from-scratch --count 5

# Fewer steps = faster
python src/generate.py --label Dog --from-scratch --steps 20
```

---

## 5. Success Criteria

| Test | Target |
|------|--------|
| Diffusion generates non-noise mel | Output ≠ white noise |
| HiFi-GAN converts to audible audio | RMS > 0.05, peak < Nyquist |
| Different classes sound different | Dog ≠ Cat ≠ Insect |
| Training loss decreases | Loss < 0.1 by epoch 50 |
| Generate time < 5 sec | 50 DDIM steps on GPU |
