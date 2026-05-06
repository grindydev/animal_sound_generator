# Phase 7a: HiFi-GAN Neural Vocoder (mel → audio)

> **Why 7a before Diffusion**: Without a vocoder, even perfect spectrograms sound like garbage through GriffinLim. Diffusion upgrades the generator quality — but we can't hear the result until we fix the converter. Vocoder first.

---

## 1. The Problem

```
Current pipe:      VAE → mel_spec → GriffinLim (math) → garbage audio ❌
```

GriffinLim at 64 mel bands throws away 96% of original energy and all phase/harmonic/transient information. The VAE produces mel spectrograms that a CNN classifies at 100% accuracy — the spectrograms are correct. The failure is 100% in the conversion from mel back to audio.

We have been trying to fix this for **multiple sessions** (normalization fixes, sqrt fixes, clamp fixes, n_fft changes) and each time the audio comes out unrecognizable. This is fundamental — you cannot mathematically reconstruct 513 frequency bins from 64 mel values while preserving phase. You need a neural network that has SEEN real audio before.

---

## 2. Solution

```
New pipe:           VAE → mel_spec → HiFi-GAN (neural net) → real audio 🎵
```

HiFi-GAN is trained on `(real_audio → compute_mel) → (mel → predict_audio)` pairs. It learns: "when I see this mel pattern, the real waveform underneath looks like this." Once trained, it converts ANY mel spectrogram — including fake ones from the VAE — into natural-sounding audio.

**Both components are separate, trained independently, no retraining needed:**

| Stage | Model | Trained on | Purpose |
|-------|-------|-----------|---------|
| A | VAE | Real mel specs | Create new animal mel spectrograms |
| B | Classifier | Real mel specs | Evaluate generation quality |
| C | **HiFi-GAN** | Real mel+audio pairs | Convert mel → listenable audio |

---

## 3. Is This Industry Standard?

Yes. Every production TTS system splits generation from vocoding:

| System | Generator | Vocoder |
|--------|-----------|---------|
| Google Tacotron 2 | Tacotron | WaveNet |
| NVIDIA FastPitch | Transformer | **HiFi-GAN** |
| Meta Voicebox | Flow Matching | HiFi-GAN |
| ElevenLabs | Custom LLM | Custom vocoder |
| **Our project** | **VAE** | **HiFi-GAN** |

Two separate jobs. Two separate models. Chained like Lego.

---

## 4. Architecture Overview

### 4.1 Generator (mel → waveform)

```
mel [B, 64, T]
      ↓
Conv1D pre-net  (in=64, hidden, kernel=7)
      ↓
┌─────────────────────────────────────────┐
│  ┌─ Upsample 5× → ResBlock(3,7,11) ─┐  │
│  ┌─ Upsample 5× → ResBlock(3,7,11) ─┤  │  ← repeated
│  ┌─ Upsample 4× → ResBlock(3,7,11) ─┤  │    5×5×4×2 = 200 = hop_length
│  └─ Upsample 2× → ResBlock(3,7,11) ─┘  │
│          MRF (Multi-Receptive Field)    │
└─────────────────────────────────────────┘
      ↓
Conv1D post-net  (kernel=7, tanh output)
      ↓
waveform [B, 1, T × 200]
```

**Why MRF?** Each ResBlock uses 3 kernel sizes (3,7,11) in parallel, summed. This lets the model see both fast transients (small kernel) and sustained tones (large kernel) simultaneously.

**Why those upsample rates?** `5×5×4×2 = 200 = hop_length`. Each mel time step becomes 200 audio samples.

### 4.2 Discriminator (judge real vs fake audio)

HiFi-GAN uses TWO discriminator families to catch different types of artifacts:

**MPD (Multi-Period Discriminator):** 5 discriminators, each checks a specific period (pitch interval):
```
Period 2:  check every 2nd sample   — catches high-frequency artifacts
Period 3:  check every 3rd sample   — catches mid-frequency artifacts  
Period 5:  check every 5th sample
Period 7:  check every 7th sample
Period 11: check every 11th sample  — catches low-frequency wobble
```
Each MPD reshapes audio `[1, L] → [B, period, L/period]` and applies 2D Conv layers. This efficiently checks periodic structure at each pitch level.

**MSD (Multi-Scale Discriminator):** 3 discriminators at different audio resolutions:
```
Scale 1: raw audio            [1, 22050]     — fine details
Scale 2: avg-pooled ×2        [1, 11025]     — medium structure
Scale 3: avg-pooled ×4        [1, 5512]      — overall envelope
```
Each MSD applies 1D Conv layers. Catches wrong overall shape across timescales.

### 4.3 Losses

| Loss | Weight | What it does |
|------|--------|-------------|
| **L_mel** | λ=45 | `|real_mel - mel(fake_audio)|` — forces generated audio to have the right frequency content. Heavy weight because this is the primary goal. |
| **L_adv** | λ=1 | Hinge GAN loss — realism through adversarial training |
| **L_fm** | λ=2 | Feature matching: `Σ|D_k(real) - D_k(fake)|` across ALL discriminator layers. Forces generator to match internal representations, not just fool the final output. |

---

## 5. Training Strategy

### 5.1 Data Format
```
Training pairs from existing dataset (9000 samples, 8 classes):
  Input:  mel spectrogram  [1, 64, T]     ← computed from real audio
  Target: audio waveform   [1, 1, L]       ← original raw audio
```

No synthetic data needed. Trained entirely on real `(mel, audio)` pairs.

### 5.2 Segment Training
Audio vary in length (1-30 seconds). Training on full audio is wasteful and memory-heavy. Instead:

- Use `smart_crop()` from `src/smart_crop.py` (energy-based VAD, already built)
- Find loudest activity region → center a segment window on it
- **Segment size: 32768 samples** (1.49 seconds) — long enough for a bark/meow, short enough for efficient GAN training
- Compute mel from the cropped waveform (same `MelSpectrogram` + `SimpleNormalize` from `data_loader.py`)
- Training pair: `(normalized_mel_segment, raw_audio_segment)`

```python
# Reusing existing code:
from smart_crop import smart_crop
from data_loader import get_transformations

# In HiFi-GAN train loader:
wav = torchaudio.load(path)[0]          # [1, full_samples]
crops = smart_crop(wav, crop_samples=32768, num_crops=1)
segment = crops[0]                        # [1, 32768]
mel = train_tfm(segment.unsqueeze(0))     # [1, 1, 64, ~160]
# Training pair: (mel, segment)
```

**Why not random cropping:** Animal audio has lots of silence between calls. Random cropping lands on silence ~70% of the time → discriminator learns "silent = real" → generator produces silence. `smart_crop` ensures segments contain actual animal sounds.

### 5.3 Training Loop
```
For each epoch:
  For each batch:
    1. Load real_audio [B, 1, 32768] from dataloader (smart_crop already applied)
    2. Compute real_mel [B, 64, ~160] from real_audio (MelSpectrogram + normalize)
    2. Generator: real_mel → fake_audio
    3. Compute fake_mel from fake_audio
    4. Adversarial step:
       a. Discriminators judge real_audio → real scores
       b. Discriminators judge fake_audio → fake scores  
       c. D_loss = hinge_real + hinge_fake
       d. D backward
    5. Generator step:
       a. D judge fake_audio again (after D updated)
       b. G_loss = λ_mel × L_mel + λ_fm × L_fm + λ_adv × L_adv(G)
       c. G backward
    6. Log metrics
```

### 5.4 Config & Hyperparams
```
# Audio
sample_rate:   22050
hop_length:    200
n_mels:        64
n_fft:         1024

# Generator
hidden_dim:            128
resblock_kernels:      [3, 7, 11]
resblock_dilations:    [[1,3,5], [1,3,5], [1,3,5]]
upsample_rates:        [5, 5, 4, 2]
upsample_kernels:      [10, 10, 8, 4]

# Discriminators
mpd_periods:           [2, 3, 5, 7, 11]
msd_scales:            3

# Training
batch_size:            8             # smaller batch for 32768-sample segments
segment_size:          32768         # 1.49s @ 22050Hz — long enough for a vocalization
learning_rate:         2e-4
lr_decay:              0.999 per epoch
num_epochs:            50            # GANs need more steps than VAE (~8400 total)
adam_betas:            [0.8, 0.99]

# Loss weights
lambda_mel:            45
lambda_fm:             2
lambda_adv:            1
```

---

## 6. Files to Create

```
src/hifigan/
├── __init__.py
├── config.py            # All hyperparameters, one place
├── generator.py         # MRF generator
├── discriminator.py     # MPD + MSD discriminators
├── losses.py            # Mel L1, feature matching, hinge GAN
├── train.py             # Training loop with checkpointing
├── inference.py         # Load → mel_to_waveform() single API call
└── utils.py             # init_weights, get_padding, scan_checkpoints
```

### Existing Files Modified (one-line swaps)

| File | Change | 
|------|--------|
| `client/server.py` | `spectrogram_to_waveform()` → `hifigan.inference.mel_to_waveform()` |
| `scripts/diagnose_audio.py` | Same swap |

---

## 7. Acceptance Criteria

| Check | How to verify |
|-------|--------------|
| Mel reconstruction | `mel(fake_audio)` matches input mel closely |
| Audio naturalness | Listen — sounds like a real animal, no hiss/static/wobble |
| VAE integration | Generate VAE mel spec → HiFi-GAN → recognizable animal sound |
| Web app | `python client/start.py` → generate button → hear actual dog bark |
| Speed | <100ms per 5-second sample (GriffinLim takes ~500ms) |

---

## 8. Risk / Mitigation

| Risk | Mitigation |
|------|-----------|
| GAN training instability | Hinge loss (stable), low lr=2e-4, feature matching for gradient |
| Mode collapse (all sounds same) | MPD checks period structure, MRF gives multi-scale generation |
| Long training (100 epochs) | Segment training (32768 samples, smart_crop for active regions), MPS acceleration |
| Overfitting to training animals | Your data has 8 diverse animal classes — good coverage |

---

## 9. Implementation Order (7 sub-steps)

| Step | File | What |
|------|------|------|
| 7a.1 | `config.py` | Constants, hyperparams dict |
| 7a.2 | `utils.py` | `init_weights()`, `get_padding()`, `AttrDict` |
| 7a.3 | `generator.py` | ResBlock, MRF block, HiFiGANGenerator |
| 7a.4 | `discriminator.py` | PeriodDiscriminator, ScaleDiscriminator, MPD, MSD |
| 7a.5 | `losses.py` | MelLoss, FeatureMatchLoss, GeneratorLoss, DiscriminatorLoss |
| 7a.6 | `train.py` | DataLoader, train loop, checkpoint save |
| 7a.7 | `inference.py` | `mel_to_waveform()` function |
| 7a.8 | Swap calls | Update `server.py` and `diagnose_audio.py` |
| 7a.9 | `documents/hifigan_guide.md` | Concept doc with analogies |

---

## 10. Expected Timeline

| Item | Time |
|------|------|
| Implementation (9 files) | ~30 min |
| Training (50 epochs) | ~1.5 hours (MPS, segment-based) |
| Integration + testing | ~15 min |

---

## 11. What Happens After This Works

```
✅ Phase 7a: HiFi-GAN    → mel specs can be heard as real audio
⬜ Phase 7b: Diffusion    → upgrade VAE with diffusion for sharper spectrograms  
⬜ Phase 7c: Sequential   → generate evolving soundscapes
⬜ Phase 7d: Mixing       → blend two animals together
⬜ Phase 7e: UI v2        → polished web interface
```

