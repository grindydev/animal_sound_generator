# Phase 7b — Diffusion Refinement Plan (Archived)

> **The full learning reference is now in `phase-7b-diffusion.md`.**  
> This file kept for historical implementation checklist only.

## Implementation Checklist (Completed)

- [x] `src/diffusion/config.py` — hyperparameters
- [x] `src/diffusion/unet.py` — U-Net with time + class conditioning
- [x] `src/diffusion/diffusion.py` — forward/reverse process, noise schedule, DDIM
- [x] `src/diffusion/train.py` — training loop with VAE mix-in (30%)
- [x] `src/diffusion/inference.py` — refinement API, full pipeline
- [x] `documents/phase-7b-diffusion.md` — complete learning reference

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Train on | Real mels + 30% VAE reconstructions | Model sees VAE-quality data, not surprised at inference |
| Architecture | 2D U-Net on spectrograms (~17.8M params) | Small enough for ~3K clips, big enough for quality |
| Conditioning | FiLM (time + class → scale+shift in ResBlocks) | Simple, proven, reusable pattern |
| Sampling | DDIM (50 steps) | Fast inference, minimal quality loss vs full 1000 steps |
