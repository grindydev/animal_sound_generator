# Animal Sound Generator

A deep learning project that generates animal sounds from scratch using variational autoencoders (VAE).

## Results

> Will be filled as phases are completed.

## Dataset

Using **FSD50K** (richer than ESC-50) with animal sound subset.

### One command to download

```bash
python scripts/download_data.py
```

This will:
1. Clone FSD50K repo (if not already)
2. Install Git LFS
3. Pull only the target audio files via LFS
4. Organize into `data/animal_audio/{Class}/` folders
5. Generate `data/animal_audio/metadata.csv`

### Customize what to download

Edit the top of `scripts/download_data.py`:

```python
# Files per class — None = all available, or set a number to cap
FILES_PER_CLASS = None          # e.g. {"Dog": None, "Frog": 50}

# Split ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Which classes (add/remove as needed)
CLASS_MIDS = {
    "Dog":      ["/m/0bt9lr", "/m/05tny_"],   # Dog, Bark
    "Cat":      ["/m/01yrx", "/m/07qrkrw"],    # Cat, Meow
    "Rooster":  ["/m/09b5t"],                   # Chicken_and_rooster
    ...
}
```

Find more label mids in `data/fsd50k_metadata/labels/vocabulary.csv`.

### Data layout after setup

```
data/
├── animal_audio/              # Playable .wav files (open to listen)
│   ├── metadata.csv           # fname, label, split
│   ├── Cat/       (303 files)
│   ├── Crow/      (72 files)
│   ├── Dog/       (750 files)
│   ├── Frog/      (61 files)
│   ├── Hen/       (86 files)
│   ├── Insect/    (371 files)
│   ├── Noise/     (1,222 files)
│   └── Rooster/   (136 files)
│
└── fsd50k_metadata/           # FSD50K repo (labels, metadata, LFS pointers)
    ├── labels/                   ← dev.csv, eval.csv, vocabulary.csv
    ├── metadata/                 ← JSON metadata
    └── clips/dev/                ← LFS cache (pulled files)
```

### Classes (8)

| Class | Train | Val | Test | Total | Source labels |
|-------|-------|-----|------|-------|---------------|
| Dog | 525 | 112 | 113 | 750 | Dog, Bark |
| Cat | 212 | 45 | 46 | 303 | Cat, Meow |
| Noise | 855 | 183 | 184 | 1,222 | Traffic, Rain, Thunder, Wind |
| Insect | 259 | 55 | 57 | 371 | Insect, Cricket |
| Rooster | 95 | 20 | 21 | 136 | Chicken_and_rooster |
| Hen | 60 | 12 | 14 | 86 | Fowl |
| Crow | 50 | 10 | 12 | 72 | Crow |
| Frog | 42 | 9 | 10 | 61 | Frog |
| **Total** | **2,098** | **446** | **457** | **3,001** | |

These are the maximums — set `FILES_PER_CLASS = 50` or `{"Dog": 100, "Frog": None}` to control per-class limits.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Phase 1: Explore audio data
cd src/
python data_loader.py

# Phase 2: Train classifier
python train.py

# Phase 8: Deploy web app (after training generator)
cd ../client/
pip install -r requirements.txt
python start.py
```

Open http://localhost:3000 — click an animal button to generate a unique sound.

## Project Structure

```
├── src/
│   ├── data_loader.py             # Phase 1: Audio loading, spectrograms
│   ├── model.py                   # Phase 2: Audio classifier
│   ├── train.py                   # Phase 2: Training pipeline
│   ├── evaluate.py                # Phase 2: Classifier evaluation
│   ├── autoencoder.py             # Phase 3: Autoencoder
│   ├── vae.py                     # Phase 4: Conditional VAE generator
│   ├── evaluate_gen.py            # Phase 5: Generation quality metrics
│   ├── latent_mixing.py           # Phase 6a: Mix multiple animals
│   ├── sequential_generator.py    # Phase 6b: Longer sounds
│   ├── diffusion_refine.py        # Phase 6c: Diffusion refinement
│   ├── transfer_generator.py      # Phase 7a: Transfer learning
│   ├── tuning.py                  # Phase 7b: Optuna tuning
│   ├── unet_vae.py                # Phase 7c: U-Net skip connections
│   ├── grad_cam_audio.py          # Phase 7d: Grad-CAM on spectrograms
│   ├── optimize.py                # Phase 7e: Pruning + Quantization
│   ├── export_onnx.py             # Phase 8: ONNX export
│   └── helper_utils.py            # Shared utilities
│
├── client/
│   ├── server.py                  # Phase 8: FastAPI backend
│   ├── start.py                   # Phase 8: One-command launcher
│   └── frontend/                  # Phase 8: React frontend
│
├── documents/                     # Learning notes
├── models/                        # Saved checkpoints
├── data/
│   ├── animal_audio/            # Playable wav files (by class)
│   └── fsd50k_metadata/         # FSD50K labels, metadata & LFS pointers
├── roadmap.md                     # Full learning roadmap (8 phases)
└── README.md                      # This file
```

## What I Learned

See [roadmap.md](roadmap.md) for detailed notes on each phase.

## License

Educational project — not intended for production use.
