# Audio Generation Workflow

> Complete pipeline: training order, model roles, and how sound is generated.

---

## 1. The Pipeline (End-to-End)

```
"dog" label
    ↓
┌──────────────────┐
│   ImprovedVAE    │  Generates: "what should it look like"
│   (src/vae/)     │  FiLM class conditioning in every decoder layer
└────────┬─────────┘
         ↓
  [1, 64, 552] normalized mel spectrogram
         ↓
┌──────────────────┐
│ Diffusion UNet   │  (OPTIONAL) Refines blurry VAE output
│ (src/diffusion/) │  Only useful if VAE output has noise-like artifacts
└────────┬─────────┘
         ↓
  [1, 64, 552] sharpened mel spectrogram
         ↓
┌──────────────────┐
│    HiFi-GAN      │  Converts mel "thumbnail" → full audio waveform
│ (src/hifigan/)   │  5×5×4×2 = 200 = hop_length
└────────┬─────────┘
         ↓
  [1, 110400] audio waveform @ 22050 Hz  (5 seconds)
         ↓
     🔊 dog_bark.wav
```

---

## 2. Training Order

Train models in this sequence — each depends on the previous:

```
Step 0: Classifier  ← ALREADY TRAINED (91% accuracy)
  python src/train_classifier.py
  → models/best_audio_cnn_train.pth  (457K params, 1.8 MB)
  Role: Classify animal sounds. Used by VAE for class supervision loss.

Step 1: Autoencoder (foundation)
  python src/vae/train_ae.py
  → models/best_autoencoder_train.pth  (149M params, 596 MB)

Step 2: VAE (adds generation capability)
  python src/vae/finetune.py
  → models/best_vae_finetune_train.pth  (223M params, 891 MB)
  Role: Add class conditioning + probabilistic sampling
  Uses: best_autoencoder_train.pth (encoder/decoder weights)
         best_audio_cnn_train.pth (classifier for supervision)

Step 3: HiFi-GAN (mel → audio)       ← ALREADY TRAINED
  src/hifigan/train.py
  → models/hifigan_generator_train.pth
  Role: Convert mel spectrograms to audio waveforms

Step 4: Diffusion (OPTIONAL refinement)
  src/diffusion/train.py
  → models/diffusion_unet_train_best.pth
  Role: Sharpen VAE-generated spectrograms
```

---

## 3. What Each Model Does

### 3.1 Autoencoder (`ImprovedAutoencoder`)

```
Input:  mel spectrogram [B, 1, 64, 552]
           ↓ Encoder (4 ResBlocks, stride=2)
           ↓ 1→32→64→128→256 channels
           ↓ [B, 256, 4, 35]
           ↓ Self-Attention (temporal coherence)
           ↓ Flatten → Linear → z [B, 2048]
           ↓ Linear → Reshape → [B, 256, 4, 35]
           ↓ Decoder (4 stages, skip connections from encoder)
           ↓ 256→128→64→32→16→1 channels
Output: reconstructed [B, 1, 64, 552]

Loss: MSE(reconstructed, input)

Params: 149M (base_channels=32) — fits GTX 1650 4GB
```

**Key improvements over v1:**
- Residual blocks (gradient flow)
- Skip connections encoder→decoder (detail preservation)
- Self-attention at bottleneck (temporal patterns)
- 2048-dim latent (2× larger, less compression)

### 3.2 VAE (`ImprovedVAE`)

Built on top of the autoencoder with two additions:

**Addition 1: Probabilistic bottleneck**
```
Encoder output → fc_mu [B, 2048] + fc_log_var [B, 2048]
z = mu + sigma * random_noise  ← reparameterization trick
```

**Addition 2: FiLM class conditioning**
```
Class "Dog" → Embedding(8, 128) → FiLM in EVERY decoder block
Each decoder block: h = h * (1 + γ) + β
```
The class embedding modulates all 4 decoder stages (not just concatenated to z). This gives the class 4× more influence.

```
Training loss:
  total = MSE(recon, input)
        + beta * KL_divergence     ← organizes latent space
        + 0.5 * CrossEntropy(      ← class supervision
            classifier(recon), label
          )

  beta: 0 → 0.01 (exponential ramp over 20 epochs)
  classifier: frozen SimpleAudioCNN (91% accuracy)

Generation:
  z ~ N(0, temperature * I)  ← sample from prior
  z + class_embed → decoder → new spectrogram
```

**Checkpoints used by VAE finetuning:**
| File | Size | Trained By | Role |
|------|------|------------|------|
| `best_autoencoder_train.pth` | 596 MB | `src/vae/train_ae.py` | Encoder/decoder weights |
| `best_audio_cnn_train.pth` | 1.8 MB | `src/train_classifier.py` | Classifier for supervision loss |

### 3.3 Classifier (`SimpleAudioCNN`)

```bash
# Trained by: python src/train.py
# Output: models/best_audio_cnn_train.pth
```

```
Input:  mel spectrogram [B, 1, 64, T]
           ↓ 4 ConvBlocks (1→32→64→128→256)
           ↓ AdaptiveAvgPool → [B, 256, 1, 1]
           ↓ Flatten → Linear(256, 256)
           ↓ Linear(256, 8)
Output: class logits [B, 8]

Accuracy: 91%
Used by: VAE finetuning (class supervision loss)
         evaluate.py (classifier evaluation)
```

### 3.4 HiFi-GAN

```
Input:  mel spectrogram [B, 64, T]
           ↓ Pre-conv: Conv1d(64→256, k=7)
           ↓ 4 MRF blocks (upsample + multi-kernel resblocks)
           ↓ Total upsample: 5×5×4×2 = 200 = hop_length
           ↓ Post-conv: Conv1d(16→1, k=7)
Output: audio waveform [B, 1, T×200 samples]
```

### 3.5 Diffusion (Optional)

```
Input:  VAE-generated mel + noise
           ↓ DDIM denoising (50 steps)
Output: sharpened mel

Training: U-Net predicts noise added to mel spectrograms
Strength: 0.05-0.10 for safe SNR with VAE output
```

---

## 4. Raw Audio vs Mel Spectrogram

### Raw Audio
```
[0.1, -0.3, 0.7, -0.9, 0.2, ...]  ← 110,400 numbers for 5 seconds @ 22050 Hz
1D — just a wiggly line. Hard for neural nets to "see" patterns.
```

### Mel Spectrogram
```
         Time (552 frames) →
Freq ↓   frame0  frame1  ...  frame551
 bin0  [  0.1     0.2   ...    0.0  ]  ← deep rumble
 bin1  [  0.3     0.5   ...    0.1  ]
  ...  [  ...     ...   ...    ...  ]
bin63  [  0.2     0.1   ...    0.5  ]  ← high whistle

Shape: [64, 552]  ← 64 frequency bins × 552 time frames
2D grid — like a grayscale "photo" of sound, easy for Conv2d to process.
```

### Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `sample_rate` | 22050 | Samples per second |
| `hop_length` | 200 | Audio samples per mel frame |
| `n_mels` | 64 | Frequency bins |
| `n_fft` | 1024 | FFT window size |
| `segment_frames` | 552 | 5 seconds of mel frames |

---

## 5. File Structure

```
src/
  vae/              ← VAE + Autoencoder package
    blocks.py       — ResEncoderBlock, ResDecoderBlock, SelfAttention1D, FiLM
    autoencoder.py  — ImprovedAutoencoder (149M params)
    model.py        — ImprovedVAE (223M params, FiLM conditioning)
    train_ae.py     — Train autoencoder (Ctrl+C resume)
    finetune.py     — Finetune VAE from AE (Ctrl+C resume)

  diffusion/        ← Diffusion package
    config.py
    diffusion.py    — DDPM/DDIM forward & reverse processes
    inference.py    — refine_spectrogram() API
    train.py        — Training with VAE mix-in
    unet.py         — SpectrogramUNet (18M params)

  hifigan/          ← HiFi-GAN package
    config.py
    generator.py    — HiFiGANGenerator
    discriminator.py
    inference.py    — mel_to_waveform() API
    train.py

  model.py          — SimpleAudioCNN (classifier, 457K params)
  data_loader.py    — Shared dataloader (80/20 train/val split)
  train_classifier.py — Train the classifier
  generate.py       — Main entry: VAE → [Diffusion] → HiFi-GAN → audio
  smart_crop.py     — Energy-based audio cropping
  helper_utils.py   — Progress bars, plotting

models/
  best_audio_cnn_train.pth       — Classifier       (457K, 1.8 MB)
  best_autoencoder_train.pth     — Autoencoder v2   (149M, 596 MB)
  best_vae_finetune_train.pth    — VAE v2           (223M, 891 MB)
  hifigan_generator_train.pth    — HiFi-GAN         (3.3M, 13 MB)
  diffusion_unet_train_best.pth  — Diffusion UNet   (17.8M, 71 MB)
```

---

## 6. Running the Pipeline

### Training (one-time)
```bash
# Step 1: Autoencoder (~3 hrs)
python src/vae/train_ae.py

# Step 2: VAE (~3 hrs)
python src/vae/finetune.py

# Step 3 (optional): Diffusion (~5 hrs)
python src/diffusion/train.py
```

### Generation
```bash
# VAE only
python src/generate.py --label Dog --no-diff

# VAE + Diffusion (use low strength!)
python src/generate.py --label Dog --strength 0.07 --diffusion-steps 10

# Multiple samples
python src/generate.py --label Cat --count 5

# All animals
python src/generate.py
```

### Key Generate Flags
| Flag | Default | Effect |
|------|---------|--------|
| `--label` | all | Animal class |
| `--temperature` | 0.7 | Diversity (0.5=consistent, 1.5=wild) |
| `--strength` | 0.6 | Diffusion strength (keep ≤ 0.10) |
| `--diffusion-steps` | 50 | DDIM steps |
| `--no-diff` | — | Skip diffusion |
| `--count` | 1 | Samples per class |
