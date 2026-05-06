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

**FSD50K: Larger and richer than ESC-50**

- 51,197 audio recordings with human-labeled sound events
- We use an animal sound subset (3,001 clips): `Dog`, `Cat`, `Rooster`, `Frog`, `Crow`, `Insect`, `Hen`, `Noise`
- Variable-length clips (1–30 seconds, no cropping)
- Download script: `python scripts/download_data.py` (pulls only needed files via Git LFS)

```
data/
├── animal_audio/              # Playable .wav files organized by class
│   ├── metadata.csv           # fname, label, split
│   ├── Dog/       (750 files)
│   ├── Cat/       (303 files)
│   ├── Rooster/   (136 files)
│   ├── Frog/       (61 files)
│   ├── Crow/       (72 files)
│   ├── Insect/    (371 files)
│   ├── Hen/        (86 files)
│   └── Noise/    (1222 files)
└── fsd50k_metadata/           # FSD50K repo (labels, LFS cache)
```

Class mapping (AudioSet mids → display name):
```
Dog:      /m/0bt9lr (Dog), /m/05tny_ (Bark)
Cat:      /m/01yrx (Cat), /m/07qrkrw (Meow)
Rooster:  /m/09b5t (Chicken_and_rooster)
Frog:     /m/09ld4 (Frog)
Crow:     /m/04s8yn (Crow)
Insect:   /m/03vt0 (Insect), /m/09xqv (Cricket)
Hen:      /m/025rv6n (Fowl)
Noise:    /m/0btp2 (Traffic), /m/06mb1 (Rain), /m/0ngt1 (Thunder), /m/03m9d0z (Wind)
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
    │   • Crop to 2-second clips
    ▼
 Phase 2 — Audio Classifier (baseline, like NSFW Phase 1-2)
    │   • 2D CNN on spectrograms (your NSFW CNN skills transfer!)
    │   • Confusion matrix, accuracy, per-class analysis
    │   • This classifier becomes your evaluation tool later
    ▼
 Phase 3 — Autoencoder (learn to reconstruct) ✅
    │   • Encoder: compress spectrogram → latent vector
    │   • Decoder: latent vector → reconstruct spectrogram
    │   • The decoder is the GENERATOR you'll build on
    ▼
 Phase 4 — Conditional VAE (generate by class + diverse) ✅
    │   • Add class conditioning: specify which animal to generate
    │   • VAE sampling: same class → different sounds each time
    │   • Latent space exploration, interpolation between animals
    ▼
 Phase 5 — Audio Quality & Evaluation ✅
    │   • Classification agreement, diversity, t-SNE
    │   • Compare real vs generated spectrograms
    │   • DISCOVER MODE COLLAPSE → debug → fix (Phase 5.1)
    ▼
 Phase 6 — Deployment (Web App) 🔜
    │   • FastAPI + React: click button → hear sound
    │   • Temperature control, model comparison
    │   • MLflow model registry
    ▼
  Phase 7 — Audio Quality & Scale 🔲
    │   • 7a: HiFi-GAN vocoder — replace Griffin-Lim with neural converter
    │   • 7b: Diffusion refinement — sharpen blurry VAE outputs
    │   • 7c: Sequential generation — longer sounds, chained animals
    │   • 7d: Latent space mixing — blend animals, interpolate
    │   • 7e: UI v2 with all new controls
    ▼
 Phase 8 — Re-practice All Techniques on the Generator 🔲
    │   • Transfer learning, Optuna, skip connections (U-Net)
    │   • Grad-CAM on spectrograms, pruning, quantization
    │   • All applied to YOUR generative model
    │   • Update UI again with new features
```

---

## Phase 1 — Understand Audio Data ✅

**Goal:** Learn to load, visualize, and transform audio files. Build a dataset.

**Build this file:** `data_loader.py` ✅

### What you practiced

| Concept | Where | Status |
|---------|-------|--------|
| Loading .wav files | `data_loader.py` — `torchaudio.load()` | ✅ |
| Mel-spectrogram conversion | `data_loader.py` — `T.MelSpectrogram()` in `get_transformations()` | ✅ |
| Log-mel scaling | `data_loader.py` — `T.AmplitudeToDB()` in `get_transformations()` | ✅ |
| Custom AudioDataset | `data_loader.py` — `__getitem__` returns (waveform, label) | ✅ |
| Train/val/test split | `get_dataloaders()` — `random_split()` seeded with manual_seed(42) | ✅ |
| Variable-length audio | `collate_fn()` pads waveforms to batch max length (no cropping!) | ✅ |
| Audio augmentation | Time shift, noise, SpecAugment — TODO in `get_transformations()` | 🔲 later |
| Waveform visualization | matplotlib — TODO (optional, for learning) | 🔲 later |

### Architecture Decision: Option B — Pad waveforms, batch transform on GPU

Instead of cropping all clips to a fixed 2 seconds (losing data), we keep the full
audio and pad variable-length waveforms in `collate_fn`. The MelSpectrogram transform
runs on GPU in the training loop for speed.

```
Flow:
  __getitem__()       → raw waveform [1, variable_samples]
      ↓
  collate_fn()        → pad to [batch, 1, max_samples_in_batch]
      ↓
  train.py:
    train_tfm()       → MelSpectrogram + AmplitudeToDB on GPU
                       → [batch, 1, 128, time_frames]
      ↓
  model()             → CNN → AdaptiveAvgPool → classifier
```

### Observations

```
┌──────────────────────────────────────────────┐
│ AUDIO DATA EXPLORATION — RESULTS             │
│ Dataset: FSD50K (not ESC-50 — richer!)       │
│ Sample rate:     44100 Hz                    │
│ Duration:        variable (1–30s, no crop)   │
│ Spectrogram:     [batch, 1, 128, time]       │
│ Value range:     -57.4 to 40.6 dB            │
│ Classes:         8 (Dog, Cat, Rooster,       │
│                  Frog, Crow, Insect,         │
│                  Hen, Noise)                  │
│ Total samples:   3,001                       │
│   Train: 2,100 (70%)                         │
│   Val:     450 (15%)                         │
│   Test:    451 (15%)                         │
│ Per class:  61–1,222 files                   │
│ Download: scripts/download_data.py           │
└──────────────────────────────────────────────┘
```

---

## Phase 2 — Audio Classifier (Baseline) ✅

**Goal:** Build a classifier as a warm-up (same pattern as NSFW Phase 1-2). This teaches you what features matter for animal sounds, AND you'll reuse this classifier in Phase 5 to evaluate your generator.

**Build these files:** `model.py`, `train.py`, `evaluate.py`

### What you practiced

| Concept | Where | Status |
|---------|-------|--------|
| 2D CNN on spectrograms | `model.py` — 4 conv blocks + AdaptiveAvgPool | ✅ |
| Config-driven training | `train.py` — CONFIG dict (test/train mode) | ✅ |
| Confusion matrix + F1 | `evaluate.py` | ✅ |
| Early stopping + Cosine LR | `train.py` | ✅ |
| Smart cropping (energy-based VAD) | `smart_crop.py` — crop loudest activity regions | ✅ |
| Spectrogram normalization | `data_loader.py` — SimpleNormalize | ✅ |
| NestedProgressBar | `helper_utils.py` — epoch + batch progress | ✅ |

### Results

```
┌──────────────────────────────────────────────┐
│ AUDIO CLASSIFIER BASELINE                    │
│ Test Accuracy:  92%                          │
│ Hardest classes: Rooster, Crow               │
│ Easy classes:    Noise, Dog, Insect          │
│ Best model saved for Phase 5 evaluation      │
└──────────────────────────────────────────────┘
```

---

## Phase 3 — Autoencoder (Reconstruct) ✅

**Goal:** Learn to compress and reconstruct audio. The **decoder** is the foundation of your generator.

**Build this file:** `autoencoder.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Encoder (compress) | `autoencoder.py` — CNN → flatten → latent vector | Like SimpleCNN but output is a vector, not a class |
| Decoder (expand) | `autoencoder.py` — latent → ConvTranspose2d → spectrogram | New — "deconvolution" / upsampling |
| Latent space | `autoencoder.py` — 64-256 dim compressed representation | L3-M2 (stable diffusion latent space) |
| Reconstruction loss | `autoencoder.py` — MSE between input & output spectrogram | New — L2 loss instead of cross-entropy |
| ConvTranspose2d | `autoencoder.py` — learnable upsampling | New — reverse of Conv2d + MaxPool |
| Griffin-Lim algorithm | `autoencoder.py` — convert spectrogram back to audio | New — inverse STFT with phase estimation |

### Architecture

```
INPUT: Spectrogram [1, 128, 87]     (128 mel bins × 87 time frames = 2 sec)
         │
    ┌────▼────┐
    │ ENCODER  │  ← Same as your NSFW CNN (conv + pool + flatten)
    │ Conv×4   │
    │ → 128-dim│
    └────┬────┘
         │  latent vector (compressed representation)
    ┌────▼────┐
    │ DECODER  │  ← Mirror of encoder (ConvTranspose2d × 4)
    │ 128-dim  │
    │ → [1,128,87] │
    └────┬────┘
         │
OUTPUT: Reconstructed Spectrogram [1, 128, 87]
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

### After autoencoder — listen and record

```
┌──────────────────────────────────────────────┐
│ AUTOENCODER                                  │
│ Reconstruction loss:  ???                    │
│ Audio quality:        ???/10                 │
│ Does it sound like the original?  Yes/No     │
│ What's lost in compression?      ???         │
└──────────────────────────────────────────────┘
```

---

## Phase 4 — Conditional VAE (Generate by Class) ✅

**Goal:** Generate animal sounds by specifying which animal, with diversity.

**Build these files:** `vae.py`, `train_vae.py`, `finetune_vae.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Variational Autoencoder | `vae.py` — encoder outputs μ and σ | L3-M2 (diffusion noise concepts) |
| Reparameterization trick | `vae.py` — `z = μ + σ * ε` where ε ~ N(0,1) | New — key to differentiable sampling |
| KL divergence loss | `vae.py` — keeps latent space organized | New — regularization for the latent space |
| Class conditioning | `vae.py` — `nn.Embedding(num_classes, embed_dim)` → concat | L2-M3 `embeddings/main.py` |
| Conditional decoder | `vae.py` — `cat([z, class_emb])` → decoder → spectrogram | L3-M2 (text conditioning in stable diffusion) |
| Sampling at inference | `vae.py` — `cat([random_noise, class_emb])` → unique generation | L3-M2 (noise → denoise → image) |
| Latent space interpolation | `vae.py` — z_dog → z_cat, decode each step | L3-M2 (stable diffusion latent arithmetic) |
| Audio saving | `vae.py` — `torchaudio.save()` | New |

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

### Conditioning approaches

We use **CONCATENATION** — the class embedding gets its own dedicated channels
in z that KL divergence cannot dilute:

```python
class_embedding = self.embed(label)              # [batch, 64]
z_input = torch.cat([z, class_embedding], dim=1) # [batch, 1024+64=1088]
output = self.decoder(z_input)

# Why NOT addition?
# z = z + class_embedding — shares the same dimensions
# When KL pushes μ→0, the class signal disappears too
# Concatenation puts a "firewall" between KL and the class lane
```

### After conditional VAE

```
┌──────────────────────────────────────────────┐
│ CONDITIONAL VAE                              │
│ Can generate dog sounds?    Yes/No           │
│ Can generate cat sounds?    Yes/No           │
│ Classes sound different?    Yes/No           │
│ Same class, different each time? Yes/No      │
│ Can interpolate dog → cat?  Yes/No           │
│ Audio quality:              ???/10           │
└──────────────────────────────────────────────┘
```

---

## Phase 5 — Audio Quality & Evaluation ✅

**Goal:** Move beyond "sounds okay to me" — quantify generation quality.

**Build this file:** `evaluate_gen.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Fréchet Audio Distance (FAD) | `evaluate_gen.py` — distribution distance between real & generated | New (like FID for images) |
| Classification agreement | `evaluate_gen.py` — does Phase 2 classifier predict the intended class? | Uses your Phase 2 classifier! |
| t-SNE visualization | `evaluate_gen.py` — real vs generated clusters | New |
| Spectrogram comparison | `evaluate_gen.py` — real vs generated side by side | L3-M2 `interpreting/main.py` |
| Diversity metric | `evaluate_gen.py` — pairwise latent distance | New |
| MLflow logging | Log all metrics | L3-M4 `MLflow/main.py` |

### Evaluation strategies

```python
# Strategy 1: Classification agreement
# Generate "dog" sound → run through Phase 2 classifier → should predict "dog"
# Agreement rate = generation accuracy

# Strategy 2: "Turing Test" — can classifier tell real from fake?
# Train classifier on real, test on generated
# If accuracy is LOW → generated looks real (good!)
# If accuracy is HIGH → generated has obvious artifacts (bad!)

# Strategy 3: Diversity metric
# Generate 10 "dog" sounds → pairwise distance in latent space
# High distance = diverse (good)
# Low distance = mode collapse (bad)
```

### After evaluation — record results

```
┌──────────────────────────────────────────────┐
│ GENERATION EVALUATION                        │
│ Classification agreement:  ??%               │
│ FAD score:                 ???               │
│ Diversity score:           ???               │
│ Real vs generated t-SNE:   ???               │
│ Best class:                ???               │
│ Worst class:               ???               │
└──────────────────────────────────────────────┘
```

### Initial results (before fix)

```
┌──────────────────────────────────────────────┐
│ GENERATION EVALUATION (v1 — addition + β=0.005) │
│ Classification agreement:  15-18% (bad!)      │
│   - Insect: 62-94%, Dog: 0-52%               │
│   - ALL other classes: ~0%                    │
│ Diversity score:            93-97             │
│ MSE:                        ~0.12-0.13        │
│                                              │
│ PROBLEM: MODE COLLAPSE!                      │
│ All generated sounds clustered as Insect,    │
│ regardless of which class was requested.     │
└──────────────────────────────────────────────┘
```

---

## Phase 5.1 — Debug & Fix Mode Collapse ✅

**Goal:** Diagnose and fix the generation quality issues found in Phase 5.

**Problem discovered:** Mode collapse — the VAE output the same "generic animal"
spectrogram for every class. The classifier recognized it as `Insect` 94% of
the time.

### Root cause: Partial Posterior Collapse

Three problems, working together:

```
1. KL pushing μ→0 (β=0.005 too aggressive)
   → All class clouds squished onto the same spot at center
   → Decoder can't tell Dog from Cat from Insect

2. Addition (z = z + class_emb) shares the same dimensions for content & class
   → When KL kills z_content, class signal gets diluted too
   → Decoder was trained to see "content+class", not "class alone"

3. No explicit class signal in the loss
   → MSE only cares about reconstruction accuracy, not class identity
   → "Average animal" has lower MSE than "risky, class-specific" output
```

### Three fixes applied

| Level | Fix | What it does |
|-------|-----|-------------|
| **L1: Architecture** | Addition → Concatenation | Class embedding gets 64 dedicated channels. KL can's touch them. `cat([z_content, class_emb])` = [1024+64=1088] |
| **L2: Loss** | + γ·CrossEntropy(classifier(recon), label) | Frozen Phase 2 classifier grades every output: "Does this look like a Dog?" γ=0.1. Gradient flows backward through classifier into VAE decoder. |
| **L3: Config** | β 0.005→0.002, free bits→0, 15 full-β epochs | Gentler KL = clouds stay spread out. No floor on KL. More refinement time. |

### Files modified

```
vae.py              — concat architecture, removed class_project
                      fc_decode(1088 → flat) instead of (1024 → flat)
train_vae.py        — β=0.002, γ=0.1, free_bits=0, 8+27+15 schedule
                      classifier loading, class_loss in vae_loss()
finetune_vae.py     — same config + class loss changes as train_vae.py
evaluate_gen.py     — unchanged (evaluation framework stays the same)
```

### How to verify the fix worked

```
✅ Classification agreement > 50% (was 15-18%)
✅ Each class gets significant predictions (was Insect domination)
✅ t-SNE shows distinct clusters per class (was one blob)
✅ MSE stays similar (~0.10-0.18), not sacrificed for class signal
✅ Both finetune and scratch models improve similarly
```

### Key lessons

```
1. "Low KL = good training" is WRONG.
   KL=0 means μ=0 everywhere → all classes at same point → mode collapse.
   A healthy VAE needs moderate KL (10-50) for functioning latent space.

2. Addition fails when KL kills content.
   The class embedding was trained as a "helper" (added to content),
   not standalone. When content dies, the helper is useless.
   Concatenation gives class its own independent lane.

3. Classifier as teacher is powerful.
   A frozen, pretrained classifier provides a direct gradient signal
   toward class-recognizable outputs. No guessing — strict grading.

4. β is a "tax rate," not a "quality dial."
   Lower β → lower tax on being different from zero
   → μ can grow to non-zero values → clouds stay distinct
   → but too low → latent space chaotic (no organization)
```

### After fix — record results

```
┌──────────────────────────────────────────────┐
│ GENERATION EVALUATION (v2 — concat + β=0.002 + γ=0.1) │
│ Classification agreement:  ??%               │
│ Best class:                ???               │
│ Worst class:               ???               │
│ MSE (scratch):             ???               │
│ MSE (finetune):            ???               │
│ Diversity score:           ???               │
│ Scratch better or finetune? ???              │
└──────────────────────────────────────────────┘
```

---

## Phase 6 — Deployment (Web App) 🔜

**Goal:** Deploy a web app where users click an animal button and hear generated sounds.
Test your VAE immediately — way more motivating than staring at metrics.

**Build these files:** `client/server.py`, `client/frontend/`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| FastAPI audio endpoint | `client/server.py` — POST /generate → .wav | L3-M4 (deployment) |
| React audio player | `client/frontend/` — play generated sounds | New |
| Random seed control | `client/server.py` — different sound each click | New |
| Sampling temperature slider | `client/frontend/` — control consistent↔wild | L3-M3 (temperature) |
| Model comparison | Side-by-side: finetune vs scratch model | New |
| MLflow model registry | Track best checkpoint, load in production | L3-M4 `MLflow/main.py` |

### UI concept (v1 — basic generation)

```
┌─────────────────────────────────────┐
│   🐾 Animal Sound Generator        │
│   Model: VAE-Concat  β=0.002       │
│                                     │
│  🐶 Dog    🐱 Cat    🐔 Rooster    │
│  🐸 Frog   🐦 Crow  🦗 Insect     │
│  🐔 Hen    🔊 Noise                 │
│                                     │
│  Model: [Finetune ▼] [Scratch ▼]   │
│  Temperature: ████░░ 0.7           │
│                                     │
│  ┌───────────────────────────────┐  │
│  │  ▁▂▃▄▅▆▇ [Waveform 2s]      │  │
│  │  ▶ 0:00 / 2:00               │  │
│  │  [▶ Play] [⬇ Download .wav]  │  │
│  └───────────────────────────────┘  │
│                                     │
│  📜 History:                        │
│  🐶 #1  🐱 #1  🐶 #2  🐔 #1       │
└─────────────────────────────────────┘
```

---

## Phase 7 — Audio Quality & Scale 🔲

**Goal:** Two quality fixes first (diffusion + neural vocoder), then scale up (longer sounds, mixing). You learn BOTH diffusion AND GANs in one project.

**Build these files:** `src/hifigan/`, `diffusion_refine.py`, `sequential_generator.py`, `latent_mixing.py`

---

### 7a — HiFi-GAN Vocoder (replace Griffin-Lim)

> **Full plan:** [`documents/phase-7a-hifigan.md`](documents/phase-7a-hifigan.md)

**Problem it solves:** Griffin-Lim guesses phase from magnitude — it's a mathematical approximation that adds static/noise. HiFi-GAN is a trained neural network that converts mel spectrograms → waveforms with realistic phase and timbre.

**What you'll learn: a GAN (Generative Adversarial Network)**

| Concept | Where | Course reference |
|---------|-------|-----------------|
| GAN Generator | ConvTranspose upsampling — mel bands → raw waveform | New (GAN from scratch) |
| GAN Discriminator | "Is this a real audio clip or a fake one?" | New |
| Multi-scale discriminator | Check realism at different time resolutions | New |
| Adversarial loss | Generator tries to fool discriminator | L3-M2 (GAN basics) |
| Feature matching loss | Match intermediate discriminator features | New |
| Mel-spectrogram loss | Generated audio's mel spec must match input mel spec | New |

```
mel spectrogram → ConvTranspose × 4 → waveform [1, T]
                        ↓
              Discriminator: real or fake?

Generator and discriminator play cat-and-mouse → generator learns
what REAL audio waveforms look like, not just mathematical approximations.
```

**Why this matters:** HiFi-GAN doesn't fix the spectrogram — it fixes the CONVERSION. Even a perfect VAE spectrogram sounds mediocre through Griffin-Lim. HiFi-GAN turns a good spectrogram into crisp, natural audio.

**VAE vs Diffusion vs HiFi-GAN (what each fixes):**
```
  Real waveform
       ↓
  [Mel spectrogram]        ← VAE generates this (blurred)
       ↓                        ↓
  Griffin-Lim (lossy)      Diffusion refiner (sharpens VAE output)
       ↓                        ↓
  Grainy audio              Sharp spectrogram
                                 ↓
                            HiFi-GAN (neural conversion)
                                 ↓
                            ✨ Crisp audio ✨
```

---

### 7b — Diffusion Refinement (sharpen the spectrogram)

**Problem it solves:** VAE outputs are blurry — the decoder averages possibilities together. Diffusion removes this blur by iterative denoising.

**What you'll learn: a DIFFUSION model**

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Forward diffusion (add noise) | Gradually corrupt spectrogram | L3-M2 `stable_diffusion` (forward process) |
| Reverse diffusion (denoise) | U-Net learns to remove noise step-by-step | L3-M2 (reverse process) |
| Noise schedule (β schedule) | How fast noise is added — controls quality vs speed | L3-M2 |
| Conditional diffusion | Guide denoising toward target animal class | L3-M2 (text conditioning) |
| DDPM sampling loop | 50–1000 denoising steps → sharp output | L3-M2 (DDPM pipeline) |
| U-Net architecture | Encoder-decoder with skip connections | L3-M1 (resnet), Phase 8c |

```
VAE output (blurry) → add small noise → U-Net denoise → sharper spectrogram → audio

Same principle as Stable Diffusion, but on spectrograms instead of images.
```

**Why this is fascinating:** The model learns to predict "remove 1% of noise," which is a much easier, more precise task than "generate the whole thing at once." That's why diffusion beats raw VAE quality.

---

### 7c — Sequential Generation (longer sounds, chained animals)

**Problem it solves:** VAE generates fixed 5-second clips. Real animal sounds happen in sequences: "bark → pause → bark → pause → growl."

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Autoregressive generation | Predict next audio chunk from previous | L3-M3 `decoder_block` (Shakespeare generator) |
| Overlap-add stitching | Crossfade between generated chunks | New |
| Sequence planning | "dog bark → pause → cat meow" as a sequence | L3-M3 `translation` (seq2seq) |
| Temperature control | Higher = more random, lower = more consistent | L3-M3 (generation temperature) |
| Causal masking | Can only attend to past chunks (not future) | L3-M3 `decoder_block` (causal mask) |

---

### 7d — Latent Space Mixing (blending animals)

**Problem it solves:** Generate hybrids — what does 70% dog + 30% cat sound like?

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Latent space arithmetic | `latent_mixing.py` — vector math in z-space | L3-M2 `stable_diffusion` (latent space) |
| Interpolation | z_dog → z_cat, decode each step → smooth transition | L3-M2 (latent arithmetic) |
| Weighted mixing | 0.7×dog + 0.3×cat → hybrid sound | L3-M2 (prompt weighting) |

```python
# Smooth interpolation: 0% dog → 100% cat
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    z_mix = (1 - alpha) * z_dog + alpha * z_cat
    audio = decoder(z_mix)  # hear the morphing!

# Three-way mix: "dog-cat-bird"
z_mix = 0.5 * z_dog + 0.3 * z_cat + 0.2 * z_bird
```

---

### 7e — Update UI v2 (all new controls)

**Goal:** Add controls for all Phase 7 features to the web app.

| Control | What it does |
|---------|-------------|
| Diffusion toggle + steps | ON = VAE + diffusion polish. 10-50 timesteps |
| Vocoder selector | Griffin-Lim vs HiFi-GAN — hear the difference! |
| Duration slider (2-30s) | How many seconds of audio to generate |
| Sequence editor | Drag-and-drop: "Dog 3s → pause 1s → Cat 2s → Rooster 1s" |
| Mix mode (2 classes) | Slider: 70% Dog + 30% Cat → hybrid sound |
| Waveform view | Longer waveform scroll, play/pause with seek |

```
┌─────────────────────────────────────────┐
│   🐾 Animal Sound Generator v2         │
│   Model: VAE + Diffusion  Size: ??? MB │
│                                         │
│  Duration: ████████░░░ 12 sec          │
│                                         │
│  Sequence:                              │
│  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │ 🐶 3s│→│ ⏸ 1s│→│ 🐱 2s│  [+Add]  │
│  └──────┘  └──────┘  └──────┘         │
│                                         │
│  🔀 Mix: 🐶 70% + 🐱 30%  [Generate]  │
│                                         │
│  ⚙️ Diffusion: [ON]  Steps: ████░ 30  │
│  🎤 Vocoder: [HiFi-GAN ▼]             │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  ▁▂▃▄▅▆▇  [Waveform 12s]      │    │
│  │  ▶ 0:03 / 12:00                │    │
│  │  [▶ Play] [⏸ Pause] [⬇ WAV]   │    │
│  └─────────────────────────────────┘    │
│                                         │
│  📜 History:                            │
│  🐶 3s  🐶🐱 Mix 5s  🐱🐔🐶 Seq 12s  │
└─────────────────────────────────────────┘
```

### After Phase 7 — record results

```
┌──────────────────────────────────────────────┐
│ AUDIO QUALITY & SCALE                        │
│ Diffusion quality improvement:     +???%      │
│ HiFi-GAN vs Griffin-Lim:          ??? vs ???  │
│ Best pipeline: VAE+Diff+HiFiGAN     ???/10    │
│ Longest sequence generated:        ??? sec    │
│ Dog→Cat interpolation smooth?      Yes/No     │
│ 3-way mix sounds natural?          Yes/No     │
│ Best approach overall:             ???         │
└──────────────────────────────────────────────┘
```

---

## Phase 8 — Re-practice All Techniques on the Generator 🔲

**Goal:** Apply every technique from your NSFW project (and PyTorch course) to the generative model. Same concepts, different domain — this solidifies your understanding.

Each sub-phase takes your working VAE generator and improves it with a technique you've already learned:

### 8a — Transfer Learning for the Generator

**You learned:** ResNet18 freeze/fine-tune/full retrain on images (NSFW Phase 4, L2-M2)  
**Now apply:** Use a pretrained audio encoder (PANNs) as the encoder part of your autoencoder

| Strategy | What you do | Expected result |
|----------|------------|----------------|
| Freeze pretrained encoder | Only train decoder | Fast, decent quality (encoder already knows audio features) |
| Fine-tune later layers | Train last encoder layers + full decoder | Best balance — adapts features for reconstruction |
| Full retrain | Train everything with small LR | Might overfit on 600 clips (same lesson as NSFW!) |

```python
# PANNs (Pretrained Audio Neural Networks)
# Trained on AudioSet (2M YouTube audio clips, 527 classes)
# Download: CNN14 weights from https://github.com/qiuqiangkong/audioset_tagging_cnn

# Replace your encoder:
pretrained_encoder = load_panns_encoder("cnn14.pth")
# Freeze all
for param in pretrained_encoder.parameters():
    param.requires_grad = False
# Your decoder stays the same — learns to decode from PANNs features
model = ConditionalVAE(encoder=pretrained_encoder, decoder=your_decoder)
```

**Compare:** Did PANNs encoder + your decoder beat your from-scratch autoencoder?

---

### 8b — Optuna Tuning for the Generator

**You learned:** Search CNN architecture, lr, dropout for classifier (NSFW Phase 3, L2-M1)  
**Now apply:** Search VAE hyperparameters for best generation quality

| Parameter | Search space | What it controls |
|-----------|-------------|-----------------|
| Latent dim | 32, 64, 128, 256 | How compressed the sound representation is |
| Encoder depth | 2-6 conv layers | How many features extracted |
| Decoder depth | 2-6 conv layers | How detailed the reconstruction |
| Learning rate | 1e-5 to 1e-2 | Training speed |
| KL loss weight (β) | 0.001 to 10.0 | Reconstruction quality vs latent organization |
| Dropout | 0.1-0.5 | Regularization |
| Augmentation intensity | SpecAugment params | Data robustness |

```python
# Optuna objective for GENERATION (not classification)
def objective(trial):
    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256])
    kl_weight = trial.suggest_float('kl_weight', 0.001, 10.0, log=True)
    # ... build VAE with these params ...
    # Metric: reconstruction loss on val set (lower = better)
    # OR: FAD score (lower = more realistic audio)
    return val_reconstruction_loss
```

**New insight:** β (KL weight) is a hyperparameter unique to VAEs — Optuna helps find the sweet spot between sharp but mode-collapsed (β too low) vs diverse but blurry (β too high).

---

### 8c — Skip Connections (U-Net Architecture)

**You learned:** ResidualTunedCNN skip connections `F(x) + x` (NSFW Phase 5, L3-M1)  
**Now apply:** U-Net skip connections from encoder → decoder (different from ResNet!)

```python
# ResNet skip:   output = F(x) + x          ← same spatial size
# U-Net skip:    encoder features → concat → decoder  ← across different levels

# Your autoencoder loses fine details through pooling.
# U-Net copies encoder features to decoder via skip connections:

# Encoder:  conv1 → pool → conv2 → pool → conv3 → pool → latent
#              ↓ skip         ↓ skip         ↓ skip
# Decoder:  up  ← concat ← up  ← concat ← up   ← latent
```

📄 **Detailed implementation guide:** [`documents/autoencoder_improvement_plan.md`](documents/autoencoder_improvement_plan.md)

This document covers 5 improvement steps for the autoencoder (applied here in Phase 8c but also relevant to Phase 3 understanding):

| Step | Change | Why | Expected Impact |
|------|--------|-----|----------------|
| 1 | **Skip connections (U-Net)** | Fine detail lost through encoder can't be recovered — skips bypass the bottleneck | **Highest** — MSE from ~0.065 to ~0.02–0.04 |
| 2 | Multi-layer bottleneck | Single linear layer compression is too harsh — add hidden layers for gradual compression | Medium |
| 3 | Remove decoder BatchNorm | BN constrains output range and adds batch-dependent noise to reconstruction | Low–Medium |
| 4 | Fix `F.interpolate` with input padding | Bilinear interpolation blurs output — pad input to multiple of 16 instead | Low (cleaner) |
| 5 | MSE + L1 combined loss | MSE produces blurry output — L1 preserves sharp edges and fine detail | Medium |

**Compare:** U-Net reconstruction quality vs basic autoencoder. Same lesson as NSFW — skip connections preserve information that would otherwise be lost.

---

### 8d — Grad-CAM on Spectrograms

**You learned:** Grad-CAM on images, what pixels matter for classification (NSFW Phase 6, L3-M2)  
**Now apply:** Grad-CAM on spectrograms — what frequencies and time regions matter

```python
# Same Grad-CAM code, but overlay on spectrogram instead of image
# For classifier:  "Why did the model predict dog?"
#   → Grad-CAM highlights LOW frequencies (bark is low-pitched)

# For VAE decoder: "Why did the decoder generate this frequency?"
#   → Grad-CAM shows which latent dimensions activate which frequencies
#   → Debug: if decoder ignores high frequencies, that's why bird sounds fail
```

**What you'll discover:**
```
Dog bark:  Grad-CAM focuses on LOW frequencies + sharp temporal onset
Cat meow:  Grad-CAM focuses on MID frequencies + sustained tone
Bird:      Grad-CAM focuses on HIGH frequencies + rapid patterns
Frog:      Grad-CAM focuses on narrow frequency band + repetitive

On your GENERATOR:
If dog generation has weak low frequencies → decoder didn't learn dog's spectral profile
→ Fix: add frequency-weighted loss or more data augmentation
```

---

### 8e — Pruning + Quantization on the Generator

**You learned:** Pruning removes small weights, quantization shrinks to INT8 (NSFW Phase 8b/8c, L3-M4)  
**Now apply:** Shrink your generator for real-time sound generation

```python
# Prune the decoder (the part that runs at inference time)
for module in generator.decoder.modules():
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        prune.l1_unstructured(module, name="weight", amount=0.3)

# Fine-tune to recover quality (same as NSFW)
# 3-5 epochs, small LR

# Quantize for INT8 inference
quantized = torch.quantization.quantize_dynamic(
    generator, {nn.Linear}, dtype=torch.qint8
)
```

**Measure:**
```
┌──────────────────────────────────────────────┐
│ GENERATOR OPTIMIZATION                       │
│ Original:    ??? ms per sound                │
│ Pruned 30%:  ??? ms  Quality drop: ???%      │
│ Quantized:   ??? ms  Quality drop: ???%      │
│ Real-time capable? (< 100ms per sound)       │
│                                              │
│ Same lesson as NSFW: small models don't      │
│ benefit much from pruning. Big models do.    │
└──────────────────────────────────────────────┘
```

---

### Phase 8 Summary

After completing all sub-phases, compare everything:

```
┌──────────────────────────────────────────────────────────┐
│ GENERATOR COMPARISON (Audio Quality / 10)                │
│                                                          │
│ Phase 4:  Basic VAE (from scratch)               ???/10  │
│ Phase 8a: VAE + PANNs transfer encoder          ???/10  │
│ Phase 8b: VAE + Optuna-tuned architecture        ???/10  │
│ Phase 8c: U-Net VAE + skip connections           ???/10  │
│ Phase 8d: (diagnostic — not a model change)         —    │
│ Phase 8e: Pruned + Quantized                     ???/10  │
│                                                          │
│ Best model:  ???                                         │
│ Best approach: ???                                       │
│ Same lessons as NSFW:                                    │
│   - Transfer learning helps but may overfit              │
│   - Skip connections improve generation quality          │
│   - Small models don't benefit much from pruning         │
│   - Optuna finds non-obvious architectures               │
└──────────────────────────────────────────────────────────┘
```

### 8f — Update UI v3 (Re-practice Features)

**Goal:** After applying all Phase 8 techniques, update the web app with:

| Feature | From Phase | What it enables |
|---------|-----------|----------------|
| ONNX export + inference | 8e | Faster generation, smaller deployment |
| U-Net model toggle | 8c | Compare basic VAE vs U-Net VAE quality |
| Transfer model selector | 8a | Compare from-scratch vs PANNs-pretrained |
| Optuna best params display | 8b | Show which hyperparameters won the search |
| Grad-CAM overlay on spectrogram | 8d | "Why did it generate this frequency?" |
| INT8 speed benchmark | 8e | Show ms-per-generation before/after quantization |

```
┌──────────────────────────────────────────┐
│   🐾 Animal Sound Generator v3          │
│                                          │
│  ⚙️ Settings:                           │
│  Architecture: [VAE-Concat ▼] [U-Net ▼] │
│  Weights: [Scratch ▼] [PANNs ▼] [Best▼] │
│  Precision: [FP32 ▼] [INT8 ▼]           │
│                                          │
│  📊 Optuna Best Config:                 │
│  latent=128, β=0.0018, lr=0.0008        │
│  [Load Config]                          │
│                                          │
│  🔍 Grad-CAM: [Show Overlay]           │
│  ┌──────────────────────────────────┐   │
│  │  Spectrogram + heatmap overlay   │   │
│  │  ████▓▓▓▓░░░░  Red = important  │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ⚡ Bench: FP32=45ms  INT8=12ms (3.75×) │
└──────────────────────────────────────────┘
```

---

## Project Structure

```
animal_sound_generator/
├── roadmap.md                     ← You are here
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader.py             # Phase 1: ✅ Audio loading, spectrograms, variable-length padding, dataset
│   ├── model.py                   # Phase 2: Audio classifier (2D CNN)
│   ├── train.py                   # Phase 2: Training pipeline
│   ├── evaluate.py                # Phase 2: Classifier evaluation
│   ├── train_autoencoder.py       # Phase 3: Autoencoder training loop
│   ├── vae.py                     # Phase 4: Conditional VAE (generator)
│   ├── train_vae.py               # Phase 4: VAE from-scratch training
│   ├── finetune_vae.py            # Phase 4: VAE finetune from AE
│   ├── evaluate_gen.py            # Phase 5: Generation quality metrics
│   ├── diffusion_refine.py        # Phase 7b: Diffusion refinement
│   ├── hifigan/                     # Phase 7a: HiFi-GAN neural vocoder
│   │   ├── config.py
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   ├── losses.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── utils.py
│   ├── sequential_generator.py    # Phase 7c: Longer + sequential sounds
│   ├── latent_mixing.py           # Phase 7d: Latent space mixing
│   ├── transfer_generator.py      # Phase 8a: Transfer learning on generator
│   ├── tuning.py                  # Phase 8b: Optuna for generator
│   ├── unet_vae.py                # Phase 8c: U-Net skip connections
│   ├── grad_cam_audio.py          # Phase 8d: Grad-CAM on spectrograms
│   ├── optimize.py                # Phase 8e: Pruning + Quantization
│   ├── export_onnx.py             # Phase 6: Export to ONNX
│   └── helper_utils.py            # Shared utilities
│
├── client/
│   ├── server.py                  # Phase 6: FastAPI backend
│   ├── start.py                   # Phase 6: One-command launcher
│   └── frontend/                  # Phase 6: React frontend
│
├── documents/                     # Learning notes per phase
├── models/                        # Saved checkpoints
└── data/
    ├── animal_audio/            # FSD50K animal clips (3,001 files)
    └── fsd50k_metadata/         # FSD50K labels & LFS cache
```

---

## Progress Tracker

| Phase | Description | File | Course Reference | Status |
|-------|------------|------|-----------------|--------|
| 1 | Audio data loading & spectrograms | `data_loader.py` | L1-M3 (datasets) | ✅ |
| 2 | Audio classifier baseline | `model.py`, `train.py`, `evaluate.py`, `smart_crop.py` | L1-M4 (CNN) | ✅ |
| 3 | Autoencoder (reconstruct) | `autoencoder.py` | L3-M2 (stable_diffusion) | ✅ |
| 4 | Conditional VAE (generate by class) | `vae.py`, `train_vae.py`, `finetune_vae.py` | L2-M3 (embeddings), L3-M2 (conditioning) | ✅ |
| 5a | Audio quality evaluation | `evaluate_gen.py` | L2-M1 (metrics), L3-M2 (interpreting) | ✅ |
| 5b | Debug & fix mode collapse | `vae.py`, `train_vae.py`, `finetune_vae.py` | — | ✅ |
| 6 | Deployment (web app) | `client/`, `export_onnx.py` | L3-M4 (ONNX, MLflow, deployment) | 🔜 |
| 7a | HiFi-GAN vocoder | `src/hifigan/` | New (GAN from scratch) | 🔲 |
| 7b | Diffusion refinement | `diffusion_refine.py` | L3-M2 (DDPM pipeline) | 🔲 |
| 7c | Sequential generation | `sequential_generator.py` | L3-M3 (decoder_block, translation) | 🔲 |
| 7d | Latent space mixing | `latent_mixing.py` | L3-M2 (stable diffusion latent space) | 🔲 |
| 7e | Update UI v2 (new controls) | `client/frontend/` | — | 🔲 |
| 8a | Transfer learning (PANNs encoder) | `transfer_generator.py` | L2-M2 (transfer_learning) | 🔲 |
| 8b | Optuna tuning (generator architecture) | `tuning.py` | L2-M1 (optuna) | 🔲 |
| 8c | U-Net skip connections | `unet_vae.py` | L3-M1 (resnet) | 🔲 |
| 8d | Grad-CAM on spectrograms | `grad_cam_audio.py` | L3-M2 (saliency_and_class_activation_map) | 🔲 |
| 8e | Pruning + Quantization | `optimize.py` | L3-M4 (pruning, quantization) | 🔲 |
| 8f | Update UI v3 (new features) | `client/frontend/` | — | 🔲 |

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
pandas>=2.0          # data handling
scikit-learn>=1.3    # Metrics, t-SNE

# Training
optuna>=3.4          # Phase 8b: Hyperparameter tuning
mlflow>=2.8          # All phases: Experiment tracking

# Audio
librosa>=0.10        # Advanced audio processing
soundfile>=0.12      # Save .wav files

# Transfer learning
panns-inference>=0.1 # Phase 8a: PANNs pretrained models

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
| ResNet18 transfer learning | PANNs transfer learning (Phase 8a) |
| ResidualTunedCNN skip connections | U-Net skip connections (Phase 8c) |
| Grad-CAM on images | Grad-CAM on spectrograms (Phase 8d) |
| Optuna for classifier | Optuna for generator (Phase 8b) |
| Pruning/Quantization on classifier | Pruning/Quantization on generator (Phase 8e) |

---

## Key Principle

Same as NSFW project:

```
Listen → measure → improve → listen again

Phase 1:  Understand the data (spectrograms, waveforms)
Phase 2:  Baseline classifier (what features matter? reusable evaluator)
Phase 3:  Autoencoder (can we compress and reconstruct?)
Phase 4:  Conditional VAE (can we generate BY CLASS? diverse?)
Phase 5:  Evaluation (how GOOD are they?)
Phase 5.1: Debug & fix (mode collapse → concat + β + class loss)
Phase 6:  Deploy (can anyone use it? click → hear!) 🔜
Phase 7:  Vocoder first (HiFi-GAN), then quality refinement (diffusion), then scale (longer, mixing) 🔲
Phase 8:  Re-practice ALL techniques on the generator
```

Every phase builds on the previous one. The classifier from Phase 2 becomes the evaluator in Phase 5. The autoencoder from Phase 3 becomes the generator in Phase 4. Phase 7 applies every technique from your NSFW project to the new domain — same concepts, deeper understanding.
