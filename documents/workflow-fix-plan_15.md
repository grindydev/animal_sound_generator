# v15 — Latent Diffusion with ESC-50

> **Status:** Implementing  
> **Approach:** Fix existing `src/latent_diff/` for ESC-50 + Griffin-Lim  
> **Why v1-v14 failed:** Direct diffusion on 35K mel bins impossible with 640 files

## Architecture

```
ESC-50 mel [1, 64, 552]
       ↓ Frozen Encoder (149M, MSE=0.015)
Spatial latent [256, 4, 35]
       ↓ ChannelReducer
Diffusion input [16, 4, 35] = 2,240 values
       ↓ Tiny UNet (3M) + DDIM + CFG
Denoised latent [16, 4, 35]
       ↓ ChannelExpander
Spatial [256, 4, 35]
       ↓ Small Decoder (2M, no skip connections)
Mel [1, 64, 552]
       ↓ Griffin-Lim
Audio
```

## Changes (5 files)

| File | Change |
|------|--------|
| `src/latent_diff/config.py` | esc50, 7 classes |
| `src/latent_diff/generate.py` | Griffin-Lim, 7 classes |
| `src/latent_diff/dataset.py` | Support escort data |
| `src/latent_diff/train_decoder.py` | esc50 path |
| `src/latent_diff/train_diff.py` | esc50 path |
| `colab/colab_train_v15.ipynb` | Phase 1 + Phase 2 + Generate |
| Delete | v14_vae.py, v14_ldm.py, train_v14.py |

## Training (L4 GPU)

| Phase | Script | Time |
|-------|--------|------|
| 1. Decoder | `train_decoder.py` | ~20 min |
| 2. Diffusion | `train_diff.py` | ~15 min |
| Total | | ~35 min |
