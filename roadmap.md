# Animal Sound Generator — Learning Roadmap

A hands-on project to learn **audio generation with deep learning** by building a model that generates animal sounds from scratch. Uses the same phased approach as the NSFW Detector — learn concepts, build from scratch, then optimize.

---

## The Problem

**Input:** An animal class label (e.g., "dog", "cat", "bird")  
**Output:** A synthetic audio clip that sounds like that animal

This is a **generative modeling** task — fundamentally different from classification:
- Classification: many inputs → one label (compress information)
- Generation: one label → rich output (expand information)

---

## Dataset

**ESC-50: Dataset for Environmental Sound Classification**

- 2,000 audio recordings, 50 classes, 5 seconds each
- We'll use the animal subset (~600 clips): `dog`, `cat`, `bird`, `chicken`, `pig`, `cow`, `frog`, `cricket`, `insects`, `crow`
- Download: https://github.com/karolpiczak/ESC-50

```bash
mkdir -p data
cd data
wget https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip master.zip
mv ESC-50-master/audio esc50_audio
rm -rf ESC-50-master master.zip
```

Expected structure:
```
data/
└── esc50_audio/
    ├── 1-100032-A-0.wav      ← {fold}-{id}-{take}-{class_id}.wav
    ├── 1-100038-A-14.wav
    └── ...
```

Class mapping in ESC-50 CSV (we'll filter to animal classes):
```
0: dog, 1: rooster, 2: pig, 3: cow, 4: frog,
5: cat, 6: hen, 7: insects, 8: sheep, 9: crow
```

---

## Key Concepts You'll Learn

Audio generation introduces concepts your NSFW project didn't cover:

| Concept | Why it matters | Analogy from NSFW project |
|---------|---------------|--------------------------|
| **Spectrograms** | Audio as 2D images (frequency × time) | Like resizing images — a different representation |
| **1D Convolutions** | Processing raw audio waveforms | Like 2D convolutions but on a single axis |
| **Autoencoders** | Learn compressed representation, then decode | Reverse of classification — output is the data itself |
| **Latent space** | A compressed "zoo" where similar sounds live near each other | Like the 256-dim vector before the final Linear layer |
| **Sampling from distributions** | Generation = sampling random noise + conditioning | New concept — classification has no randomness |
| **Conditional generation** | "Make a dog sound" vs "Make a cat sound" | Like classification in reverse — label → data |
| **Audio evaluation** | How do you measure "sounds like a dog"? | Very different from accuracy/F1 — no single right answer |

---

## Learning Path

```
 Phase 1 — Understand Audio Data
    │   • Load .wav files, visualize waveforms and spectrograms
    │   • Build AudioDataset, compute mel-spectrograms
    │   • Train an audio CLASSIFIER (warm-up, like NSFW Phase 1)
    ▼
 Phase 2 — Audio Classifier (baseline)
    │   • 1D CNN on raw audio + 2D CNN on spectrograms
    │   • Confusion matrix, accuracy, compare approaches
    │   • Course reference: L1-M4 (CNN), L3-M2 (interpreting)
    ▼
 Phase 3 — Autoencoder (learn to reconstruct)
    │   • Encoder: compress spectrogram → latent vector
    │   • Decoder: latent vector → reconstruct spectrogram
    │   • The decoder is the GENERATOR you'll build on
    │   • Course reference: L3-M2 (stable diffusion concepts)
    ▼
 Phase 4 — Conditional Autoencoder (generate by class)
    │   • Add class label to the latent space
    │   • Generate spectrogram → convert to audio
    │   • First moment: "it made a dog sound!"
    │   • Course reference: L3-M3 (self-attention, conditioning)
    ▼
 Phase 5 — Variational Autoencoder (diverse generation)
    │   • VAE: sample from learned distribution
    │   • Each sample produces a slightly different bark/meow
    │   • KL divergence loss: keep latent space organized
    │   • Course reference: L3-M2 (DDPM noise concepts)
    ▼
 Phase 6 — Audio Quality & Evaluation
    │   • FID (Fréchet Distance) for audio quality
    │   • MOS (Mean Opinion Score) — listen and rate
    │   • Compare real vs generated spectrograms
    │   • Course reference: L2-M1 (evaluation metrics)
    ▼
 Phase 7 — Deployment (generate sounds on the web)
        • Export generator, FastAPI + React web app
        • Click animal button → generate unique sound
        • Course reference: L3-M4 (ONNX, deployment)
```

---

## Phase 1 — Understand Audio Data 🔲

**Goal:** Learn to load, visualize, and transform audio files. Build a dataset.

**Build this file:** `data_loader.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Loading .wav files | `data_loader.py` — `torchaudio.load()` | New (not in course) |
| Waveform visualization | `data_loader.py` — matplotlib | — |
| Mel-spectrogram conversion | `data_loader.py` — `torchaudio.transforms.MelSpectrogram` | New — audio's equivalent of image transforms |
| STFT (Short-Time Fourier Transform) | Theory — sliding window FFT | New |
| Log-mel scaling | `data_loader.py` — `AmplitudeToDB()` | Like normalization in NSFW (mean/std) |
| Custom AudioDataset | `data_loader.py` — `__getitem__` returns (spectrogram, label) | L1-M3 `data_management/main.py` |
| Train/val/test split | Same as NSFW project | L1-M3 |

### Key insight: Audio as Images

```
Image:   [3 channels × height × width]    → 2D convolutions
Audio:   [1 channel  × freq   × time ]    → 2D convolutions on spectrogram
         OR
Audio:   [1 channel × samples         ]    → 1D convolutions on raw waveform

Spectrogram = 2D image where:
  Y-axis = frequency (low → high)
  X-axis = time
  Pixel brightness = energy at that frequency & time
```

### After this phase — record observations

```
┌──────────────────────────────────────────────┐
│ AUDIO DATA EXPLORATION                       │
│ Sample rate:     ??? Hz (usually 44100)       │
│ Duration:        5 seconds per clip           │
│ Spectrogram shape: [1, 128 mel bins, ??? frames] │
│ Animal classes:  10                           │
│ Clips per class: ~60                          │
└──────────────────────────────────────────────┘
```

---

## Phase 2 — Audio Classifier 🔲

**Goal:** Build a classifier as a warm-up (same pattern as NSFW Phase 1-2). This teaches you what features matter for animal sounds.

**Build these files:** `model.py`, `train.py`, `evaluate.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| 2D CNN on spectrograms | `model.py` — same CNN architecture as NSFW | L1-M4 `cnn/main.py` |
| 1D CNN on raw waveforms | `model.py` — alternative approach | New |
| Data augmentation for audio | `data_loader.py` — time shift, noise, pitch shift | L1-M3 (augmentation) |
| Confusion matrix | `evaluate.py` | Same as NSFW |
| MLflow tracking | `train.py` | L3-M4 `MLflow/main.py` |

### Approach comparison

```python
# Approach A: 2D CNN on Mel-spectrogram (like an image)
# Pros: well-studied, works like image classification
# Cons: loses phase information

# Approach B: 1D CNN on raw waveform
# Pros: learns directly from audio, no preprocessing bias
# Cons: longer sequences, harder to train

# Approach C (Phase 3+): Use encoder features from autoencoder
```

### After classification — record results

```
┌──────────────────────────────────────────────┐
│ AUDIO CLASSIFIER BASELINE                    │
│ Test Accuracy (2D CNN):  ??%                 │
│ Test Accuracy (1D CNN):  ??%                 │
│ Hardest classes:        ???                  │
│ Easy classes:           ???                  │
│ Spectrogram vs Raw:     ??? is better        │
└──────────────────────────────────────────────┘
```

---

## Phase 3 — Autoencoder (Reconstruct) 🔲

**Goal:** Learn to compress and reconstruct audio. The **decoder** is the foundation of your generator.

**Build this file:** `autoencoder.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Encoder (compress) | `autoencoder.py` — CNN → flatten → latent vector | Like SimpleCNN but reversed output |
| Decoder (expand) | `autoencoder.py` — latent → ConvTranspose2d → spectrogram | New — "deconvolution" / upsampling |
| Latent space | `autoencoder.py` — 64-dim compressed representation | L3-M2 (stable diffusion latent space) |
| Reconstruction loss | `autoencoder.py` — MSE between input & output spectrogram | New — L2 loss instead of cross-entropy |
| ConvTranspose2d | `autoencoder.py` — learnable upsampling | New — reverse of Conv2d + MaxPool |
| Griffin-Lim algorithm | `autoencoder.py` — convert spectrogram back to audio | New — inverse STFT with phase estimation |

### Architecture

```
INPUT: Spectrogram [1, 128, 216]
         │
    ┌────▼────┐
    │ ENCODER  │  ← Same as your NSFW CNN (conv + pool + flatten)
    │ Conv×4   │
    │ → 256-dim│
    └────┬────┘
         │  latent vector (compressed representation)
    ┌────▼────┐
    │ DECODER  │  ← Mirror of encoder (ConvTranspose2d × 4)
    │ 256-dim  │
    │ → [1,128,216] │
    └────┬────┘
         │
OUTPUT: Reconstructed Spectrogram [1, 128, 216]
         │
    ┌────▼────┐
    │ Griffin- │  ← Convert spectrogram back to waveform
    │ Lim      │
    └────┬────┘
         │
OUTPUT: Audio waveform → .wav file
```

### Key insight

```python
# Classification:  input → CNN → [features] → Linear → class label
# Autoencoder:     input → Encoder → [latent] → Decoder → reconstructed input
#                                    ↑
#                          This is what we care about!
#                          The decoder LEARNED to generate spectrograms.
```

---

## Phase 4 — Conditional Generation (by class) 🔲

**Goal:** Generate animal sounds by specifying which animal you want.

**Build this file:** `generator.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Class conditioning | `generator.py` — concatenate class embedding to latent | L3-M3 (decoder conditioning on encoder output) |
| Label embedding | `generator.py` — `nn.Embedding(num_classes, embed_dim)` | L2-M3 `embeddings/main.py` |
| Conditional decoder | `generator.py` — [latent + class_emb] → decoder | L3-M2 (text conditioning in stable diffusion) |
| Audio saving | `generator.py` — `torchaudio.save()` | New |
| Listening tests | Manual — does it sound like the right animal? | New — subjective evaluation |

### Conditioning approaches

```python
# Approach A: Concatenation
class_embedding = self.embed(label)          # [batch, 64]
z_input = torch.cat([z, class_embedding])    # [batch, 256+64]
output = self.decoder(z_input)

# Approach B: Addition (projection to same size)
class_embedding = self.embed_and_project(label)  # [batch, 256]
z_input = z + class_embedding                    # [batch, 256]
output = self.decoder(z_input)

# Approach C: Feature-wise modulation (FiLM)
gamma, beta = self.class_to_params(label)    # scale & shift
z_input = gamma * z + beta                   # per-feature modulation
output = self.decoder(z_input)
```

### After conditional generation

```
┌──────────────────────────────────────────────┐
│ CONDITIONAL GENERATION                       │
│ Can generate dog sounds?    Yes/No           │
│ Can generate cat sounds?    Yes/No           │
│ Classes sound different?    Yes/No           │
│ Audio quality:              ???/10           │
│ Common artifacts:           ???              │
└──────────────────────────────────────────────┘
```

---

## Phase 5 — Variational Autoencoder (Diverse) 🔲

**Goal:** Generate diverse, unique sounds each time. Same class → different barks.

**Build this file:** `vae.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Variational Autoencoder | `vae.py` — encoder outputs μ and σ | L3-M2 (diffusion noise concepts) |
| Reparameterization trick | `vae.py` — `z = μ + σ * ε` where ε ~ N(0,1) | New — key to differentiable sampling |
| KL divergence loss | `vae.py` — keeps latent space organized | New — regularization for the latent space |
| Sampling at inference | `vae.py` — random z → unique generation each time | L3-M2 (noise → denoise → image) |
| Latent space visualization | `vae.py` — t-SNE of z vectors colored by class | L3-M2 (interpreting) |

### VAE vs regular Autoencoder

```
Autoencoder:        x → encoder → z → decoder → x̂
                    z is a FIXED point for each input

VAE:                x → encoder → μ, σ → sample z → decoder → x̂
                    z is SAMPLED from N(μ, σ²) for each input
                    Same input → slightly different z → different output
                    This = DIVERSITY in generation
```

### The two losses

```python
# Loss = Reconstruction + KL Divergence
reconstruction_loss = MSE(output, target)              # Make it sound right
kl_loss = -0.5 * sum(1 + log(σ²) - μ² - σ²)          # Keep latent space normal
total_loss = reconstruction_loss + beta * kl_loss      # beta controls tradeoff
```

### After VAE

```
┌──────────────────────────────────────────────┐
│ VAE GENERATION                               │
│ Same class, different sounds?   Yes/No       │
│ Sound quality vs Phase 4:      Better/Worse  │
│ Latent space is organized?     Yes/No        │
│ Can interpolate between classes?  Yes/No     │
│ Interpolation sounds smooth?   Yes/No        │
└──────────────────────────────────────────────┘
```

---

## Phase 6 — Evaluation & Quality 🔲

**Goal:** Move beyond "sounds okay to me" — quantify generation quality.

**Build this file:** `evaluate.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Fréchet Audio Distance (FAD) | `evaluate.py` — distribution distance | New (like FID for images) |
| Spectrogram comparison | `evaluate.py` — real vs generated side by side | L3-M2 `interpreting/main.py` |
| Audio similarity metrics | `evaluate.py` — multi-scale spectral loss | New |
| Classification-based eval | `evaluate.py` — does classifier agree with intended class? | Uses your Phase 2 classifier! |
| t-SNE visualization | `evaluate.py` — real vs generated clusters | New |

### Evaluation strategies

```python
# Strategy 1: "Turing Test" — can a classifier tell real from fake?
# Train classifier on real, test on generated
# If accuracy is LOW → generated audio looks real (good!)
# If accuracy is HIGH → generated audio has obvious artifacts (bad!)

# Strategy 2: Classification agreement
# Generate "dog" sound → run through classifier → should predict "dog"
# Agreement rate = generation accuracy

# Strategy 3: Diversity metric
# Generate 10 "dog" sounds → pairwise distance in latent space
# High distance = diverse (good)
# Low distance = mode collapse, all same (bad)
```

---

## Phase 7 — Deployment (Web App) 🔲

**Goal:** Deploy a web app where users click an animal button and hear generated sounds.

**Build these files:** `export_onnx.py`, `client/server.py`, `client/frontend/`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| ONNX export (generator only) | `export_onnx.py` | L3-M4 `ONNX/main.py` |
| FastAPI audio endpoint | `client/server.py` — generate + return .wav | L3-M4 (deployment) |
| React audio player | `client/frontend/` — play generated sounds | New |
| Random seed control | `client/server.py` — different sound each click | New |
| Sampling temperature | `client/server.py` — control diversity | L3-M3 (generation temperature) |

### UI concept

```
┌─────────────────────────────────────┐
│   🐾 Animal Sound Generator        │
│   Click an animal to generate sound │
│                                     │
│  🐶 Dog    🐱 Cat    🐔 Rooster    │
│  🐷 Pig    🐄 Cow    🐸 Frog       │
│  🦗 Cricket 🐑 Sheep 🐦 Crow      │
│                                     │
│  ┌───────────────────────────────┐  │
│  │  ▶ [Generated waveform]      │  │
│  │  Duration: 2.0s              │  │
│  │  Temperature: ████░░ 0.7     │  │
│  │                               │  │
│  │  [▶ Play] [⬇ Download]       │  │
│  └───────────────────────────────┘  │
│                                     │
│  History:                           │
│  🐶 Dog #1  🐱 Cat #1  🐶 Dog #2  │
└─────────────────────────────────────┘
```

---

## Project Structure

```
animal_sound_generator/
├── roadmap.md                     ← You are here
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader.py             # Phase 1: Audio loading, spectrograms, dataset
│   ├── model.py                   # Phase 2: Audio classifier (1D + 2D CNN)
│   ├── train.py                   # Phase 2: Training pipeline
│   ├── evaluate.py                # Phase 2+6: Evaluation metrics
│   ├── autoencoder.py             # Phase 3: Autoencoder (encoder + decoder)
│   ├── generator.py               # Phase 4: Conditional generation
│   ├── vae.py                     # Phase 5: Variational autoencoder
│   ├── export_onnx.py             # Phase 7: Export generator to ONNX
│   └── helper_utils.py            # Shared utilities
│
├── client/
│   ├── server.py                  # Phase 7: FastAPI backend
│   ├── start.py                   # Phase 7: One-command launcher
│   └── frontend/                  # Phase 7: React frontend
│
├── documents/                     # Learning notes per phase
├── models/                        # Saved checkpoints
└── data/
    └── esc50_audio/               # ESC-50 dataset
```

---

## Progress Tracker

| Phase | Description | File | Course Reference | Status |
|-------|------------|------|-----------------|--------|
| 1 | Audio data loading & spectrograms | `data_loader.py` | L1-M3 (datasets) | 🔲 |
| 2a | Audio classifier (2D CNN on spectrogram) | `model.py`, `train.py` | L1-M4 (CNN) | 🔲 |
| 2b | Evaluate classifier | `evaluate.py` | L3-M2 (interpreting) | 🔲 |
| 3 | Autoencoder (reconstruct audio) | `autoencoder.py` | L3-M2 (stable diffusion) | 🔲 |
| 4 | Conditional generation (by class) | `generator.py` | L3-M3 (decoder, conditioning) | 🔲 |
| 5 | VAE (diverse generation) | `vae.py` | L3-M2 (noise, sampling) | 🔲 |
| 6 | Audio quality evaluation | `evaluate.py` | L2-M1 (metrics) | 🔲 |
| 7 | Deployment (web app) | `client/` | L3-M4 (ONNX, deployment) | 🔲 |

---

## Dependencies

```
# Core
torch>=2.0
torchaudio>=2.0
torchvision>=0.15
numpy>=1.24
matplotlib>=3.7

# Data
pandas>=2.0          # ESC-50 metadata CSV
scikit-learn>=1.3    # Metrics, t-SNE

# Training
optuna>=3.4          # Hyperparameter tuning (Phase 5)
mlflow>=2.8          # Experiment tracking

# Audio
librosa>=0.10        # Advanced audio processing
soundfile>=0.12      # Save .wav files

# Deployment
fastapi>=0.104
uvicorn>=0.24
onnxruntime>=1.16
python-multipart>=0.0.6
```

---

## New Concepts vs NSFW Detector

| NSFW Detector (Classification) | Animal Sound Generator (Generation) |
|--------------------------------|-------------------------------------|
| Input: image → Output: class label | Input: class label → Output: audio |
| Loss: CrossEntropy (one right answer) | Loss: MSE/MAE (many acceptable answers) |
| Metric: Accuracy, F1 | Metric: FAD, classification agreement, human rating |
| Deterministic output | Stochastic output (random = creative) |
| No decoder needed | Decoder is the core component |
| 2D convolutions on images | 2D conv on spectrograms OR 1D conv on waveforms |
| Last layer: Linear(256, 5) | Last layer: ConvTranspose2d (expand to full spectrogram) |

---

## Key Principle

Same as NSFW project:

```
Listen → measure → improve → listen again

Phase 1: Understand the data (spectrograms, waveforms)
Phase 2: Baseline classifier (what features matter?)
Phase 3: Autoencoder (can we reconstruct?)
Phase 4: Conditional generation (can we generate BY CLASS?)
Phase 5: Diverse generation (can we generate DIFFERENT sounds?)
Phase 6: Quality evaluation (how GOOD are the sounds?)
Phase 7: Deploy (can anyone use it?)
```

Every phase builds on the previous one. The decoder from Phase 3 becomes the generator in Phase 4. The classifier from Phase 2 becomes the evaluator in Phase 6.
