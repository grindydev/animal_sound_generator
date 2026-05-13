# Colab Training Guide — Animal Sound Generator

> All source configs are already Colab-optimized. No patching needed.

---

## GPU Comparison

| GPU | VRAM | Free? | Batch Size (AE/VAE) | Est. AE Time |
|-----|------|-------|---------------------|-------------|
| T4 | 16 GB | ✅ Free | 16 / 8 | ~2 hrs |
| **L4** | 24 GB | ❌ Pro | **16 / 8** | **~1.2 hrs** |
| A100 | 40 GB | ❌ Pro+ | 32 / 16 | ~0.8 hrs |

---

## Quick Start (3 steps)

### 1. Upload to Google Drive

Upload your dataset to Drive root:
```
MyDrive/animal_audio.tar.gz
```

### 2. Open Colab notebook

- Go to https://colab.research.google.com
- File → Upload notebook → select `colab/colab_train.ipynb`
- Runtime → Change runtime type → **L4 GPU** (or T4 if free)

### 3. Run cells in order

| Cell | What | Time |
|------|------|------|
| 1 | Clone repo, mount Drive, install deps | 2 min |
| 2 | Extract data from Drive | 1 min |
| 3 | Verify GPU | 10 sec |
| 4 | Train autoencoder | ~1.5 hrs |
| 5 | Save AE to Drive | 30 sec |
| 6 | Train VAE (fine-tune) | ~1.5 hrs |
| 7 | Save VAE to Drive | 30 sec |

---

## Current Config (in source files)

| Model | lr | batch | base_ch | params | workers |
|-------|:--:|:-----:|:-------:|:------:|:-------:|
| Classifier | 1e-3 | 256 | — | 457K | 4 |
| Autoencoder | 1e-3 | 16 | 32 | 149M | 4 |
| VAE finetune | 3e-4 | 8 | 32 | 223M | 4 |
| Diffusion | — | 16 | — | 18M | 4 |
| HiFi-GAN | — | 16 | — | 3.3M | 4 |

---

## Download Models After Training

From Google Drive → `MyDrive/animal_sound_generator/models/`:

```
best_autoencoder_train.pth       (149M params, ~600 MB)
best_vae_finetune_train.pth      (223M params, ~890 MB)
best_audio_cnn_train.pth         (457K params, ~2 MB)
```

Copy to your local `models/` folder, update `base_channels=32` in any inference scripts, done.

---

## Recovery After Timeout

Checkpoints save to Drive every epoch. Restart Colab → run cells 1-2 → training auto-resumes.
