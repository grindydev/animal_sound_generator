# Workflow Fix Plan v10 — Clean Data, Same Architecture

> **Date:** May 16, 2026  
> **Status:** Data ready. Training on Colab.  
> **Root cause (v1-v9):** Old data is 80% silence. All classes identical in mel (cos_sim=0.78).  
> **v10:** ESC-50 — 640 clean animal sound clips. Same v9 code. 0 architecture changes.

---

## 1. Data Comparison

| Metric | Old (animal_audio) | New (ESC-50) |
|--------|:---:|:---:|
| Dog peak freq | 4 Hz (silence) | 852 Hz **(bark)** |
| Cat peak freq | 135 Hz (rumble) | 2,006 Hz **(meow)** |
| Frog peak freq | 313 Hz (rumble) | 2,171 Hz **(croak)** |
| Rooster peak freq | 1,020 Hz (faint) | 1,502 Hz **(crow)** |
| Class mean cos_sim | 0.78 (identical) | 0.15-0.35 **(distinct)** |
| Files total | 3,001 | 640 |
| Silence in clips | ~80% | ~5% |

## 2. Files Per Class

| Class | Files | Source |
|-------|:---:|------|
| Dog | 40 | ESC-50 dog barks |
| Cat | 40 | ESC-50 cat meows |
| Rooster | 40 | ESC-50 rooster crows |
| Frog | 40 | ESC-50 frog croaks |
| Crow | 80 | ESC-50 crow + chirping_birds |
| Insect | 80 | ESC-50 insects + crickets |
| Hen | 40 | ESC-50 hen clucks |
| Noise | 280 | ESC-50 rain, wind, thunderstorm, sea_waves |
| **Total** | **640** | |

## 3. Changes

| File | Change | Reason |
|------|--------|--------|
| `src/diffusion/config.py` | `data_dir="data/esc50"`, epochs=150, gentler augment | Clean data, more epochs |
| `src/scripts/setup_esc50.py` | New file — organizes ESC-50 into class folders | Data prep |
| `colab/colab_train.ipynb` | Added ESC-50 download + setup before training | Colab workflow |

**Same UNet, same GAN, same class balance, same training loop.** Zero model changes.

## 4. Success Criteria

| Metric | Old data (v9) | ESC-50 target |
|--------|:---:|:---:|
| Recognizable classes | 0/8 | 4-7/8 |
| Audio peak | 32Hz (hum) | 200-4000Hz |
| Class-distinct outputs | No | Yes (different frequencies per class) |
| Training time | 70 min | 30 min (640 files) |
