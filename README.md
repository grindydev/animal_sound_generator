# Animal Sound Generator

A deep learning project that generates animal sounds from scratch using variational autoencoders (VAE).

## Results

> Will be filled as phases are completed.

## Dataset

Download ESC-50 and extract animal sounds:

```bash
mkdir -p data
cd data
wget https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip master.zip
mv ESC-50-master/audio esc50_audio
rm -rf ESC-50-master master.zip
```

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
├── data/                          # ESC-50 dataset
├── roadmap.md                     # Full learning roadmap (8 phases)
└── README.md                      # This file
```

## What I Learned

See [roadmap.md](roadmap.md) for detailed notes on each phase.

## License

Educational project — not intended for production use.
