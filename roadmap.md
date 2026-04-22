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
- We'll use the animal subset (~600 clips): `dog`, `rooster`, `pig`, `cow`, `frog`, `cat`, `hen`, `insects`, `sheep`, `crow`
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

**Important:** Crop all clips to 2 seconds (loudest part) for faster training.

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
    │   • 1D CNN on raw audio + 2D CNN on spectrograms
    │   • Confusion matrix, accuracy, compare approaches
    │   • Course reference: L1-M4 (CNN), L3-M2 (interpreting)
    ▼
 Phase 3 — Transfer Learning for Audio
    │   • Use pretrained audio models (PANNs, VGGish)
    │   • Same 3 strategies as NSFW: freeze / fine-tune / full retrain
    │   • Course reference: L2-M2 (transfer_learning)
    ▼
 Phase 4 — Optuna Hyperparameter Tuning
    │   • Search best classifier architecture, learning rate, augmentation
    │   • Course reference: L2-M1 (optuna)
    ▼
 Phase 5 — Grad-CAM for Audio (What frequencies matter?)
    │   • Saliency maps on spectrograms — visual + audio debugging
    │   • Course reference: L3-M2 (saliency_and_class_activation_map)
    ▼
 Phase 6 — Autoencoder (learn to reconstruct)
    │   • Encoder: compress spectrogram → latent vector
    │   • Decoder: latent vector → reconstruct spectrogram
    │   • The decoder is the GENERATOR you'll build on
    │   • Course reference: L3-M2 (stable diffusion concepts)
    ▼
 Phase 7 — U-Net Autoencoder with Skip Connections
    │   • Add skip connections from encoder → decoder
    │   • Better reconstruction = better generation later
    │   • Course reference: L3-M1 (resnet)
    ▼
 Phase 8 — Conditional Generation (generate by class)
    │   • Add class label to the latent space
    │   • Generate spectrogram → convert to audio
    │   • First moment: "it made a dog sound!"
    │   • Course reference: L3-M3 (self-attention, conditioning)
    ▼
 Phase 9 — Variational Autoencoder (diverse generation)
    │   • VAE: sample from learned distribution
    │   • Each sample produces a slightly different bark/meow
    │   • KL divergence loss: keep latent space organized
    │   • Course reference: L3-M2 (DDPM noise concepts)
    ▼
 Phase 10 — Audio Quality & Evaluation
    │   • FAD (Fréchet Audio Distance), classification agreement
    │   • Compare real vs generated spectrograms
    │   • Course reference: L2-M1 (evaluation metrics)
    ▼
 Phase 11 — Latent Space Mixing (multiple animals)
    │   • z_dog + z_cat = hybrid sound
    │   • Interpolation between animals
    │   • Course reference: L3-M2 (stable diffusion latent space)
    ▼
 Phase 12 — Longer & Sequential Sounds
    │   • Autoregressive: predict next audio chunk
    │   • Stitch chunks into longer sequences
    │   • Course reference: L3-M3 (decoder_block, translation)
    ▼
 Phase 13 — Diffusion Refinement (highest quality)
    │   • VAE output → diffusion model cleans it up
    │   • Same as DDPM but on spectrograms
    │   • Course reference: L3-M2 (DDPM pipeline)
    ▼
 Phase 14 — Pruning + Quantization
    │   • Shrink generator for real-time inference
    │   • Course reference: L3-M4 (pruning, quantization)
    ▼
 Phase 15 — Deployment (Web App)
        • Export generator, FastAPI + React web app
        • Click animal button → generate unique sound
        • MLflow tracking throughout all phases
        • Course reference: L3-M4 (ONNX, MLflow, deployment)
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
| Crop to 2 seconds | Energy-based windowing (loudest 2s) | New |

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
│ Duration:        2 seconds (cropped)          │
│ Spectrogram shape: [1, 128 mel bins, ??? frames] │
│ Animal classes:  5-10                         │
│ Clips per class: ~60                          │
└──────────────────────────────────────────────┘
```

---

## Phase 2 — Audio Classifier (Baseline) 🔲

**Goal:** Build a classifier as a warm-up (same pattern as NSFW Phase 1-2). This teaches you what features matter for animal sounds.

**Build these files:** `model.py`, `train.py`, `evaluate.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| 2D CNN on spectrograms | `model.py` — same CNN architecture as NSFW | L1-M4 `cnn/main.py` |
| 1D CNN on raw waveforms | `model.py` — alternative approach | New |
| Data augmentation for audio | `data_loader.py` — time shift, noise, pitch shift, SpecAugment | L1-M3 (augmentation) |
| Confusion matrix | `evaluate.py` | Same as NSFW |
| MLflow tracking | `train.py` — log params, metrics, spectrograms | L3-M4 `MLflow/main.py` |
| Config-driven training | `train.py` — CONFIG dict | Same as NSFW |

### Approach comparison

```python
# Approach A: 2D CNN on Mel-spectrogram (like an image)
# Pros: well-studied, works like image classification
# Cons: loses phase information

# Approach B: 1D CNN on raw waveform
# Pros: learns directly from audio, no preprocessing bias
# Cons: longer sequences, harder to train
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

## Phase 3 — Transfer Learning for Audio 🔲

**Goal:** Use a pretrained audio model — biggest accuracy jump (just like NSFW Phase 4).

**Build these files:** `transfer_audio.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Pretrained audio models | `transfer_audio.py` — PANNs (Pretrained Audio Neural Networks) or VGGish | L2-M2 `transfer_learning/main.py` |
| Strategy 1: Freeze all, train head | Same as NSFW ResNet18 Strategy 1 | L2-M2 (freeze all) |
| Strategy 2: Fine-tune later layers | Same as NSFW ResNet18 Strategy 2 | L2-M2 (fine-tune) |
| Strategy 3: Full retrain | Same as NSFW ResNet18 Strategy 3 | L2-M2 (full retrain) |
| Pretrained preprocessing | PANNs expects 64 mel bins, specific sample rate | Like ImageNet normalization in NSFW |
| Compare with Phase 2 baseline | Did pretrained help? | Same comparison pattern as NSFW |

### Pretrained audio models

```python
# Option A: PANNs (CNN14) — trained on AudioSet (2M YouTube videos)
#   Input: 64 mel bins, 2 seconds
#   Very similar to ResNet18 for images

# Option B: VGGish — trained on YouTube-8M
#   Input: 64 mel bins, 0.96 seconds
#   Older but simpler

# Option C: torchvggish (Google's VGGish ported to PyTorch)
#   pip install torchvggish
```

### After transfer learning — record results

```
┌──────────────────────────────────────────────┐
│ TRANSFER LEARNING COMPARISON                 │
│ Strategy 1 (freeze all):   ???%              │
│ Strategy 2 (fine-tune):    ???%              │
│ Strategy 3 (full retrain): ???%              │
│ Phase 2 baseline:          ???%              │
│ Best approach:             ???               │
│ Overfitting?               Yes/No            │
└──────────────────────────────────────────────┘
```

---

## Phase 4 — Optuna Hyperparameter Tuning 🔲

**Goal:** Push the best model from Phase 2-3 to its limit.

**Build this file:** `tuning.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Flexible CNN architecture | `tuning.py` — variable layers, filters, kernel sizes | L2-M1 `optuna/main.py` |
| Search space design | `tuning.py` — lr, dropout, augmentation params | L2-M1 (optuna) |
| Train fraction for fast trials | `data_loader.py` — 50% data for optuna trials | Same as NSFW |
| Retrain with best params | `tuning.py` — full data, more epochs | Same as NSFW |
| Compare with baseline | Phase 2 vs Phase 3 vs Phase 4 | Same comparison pattern as NSFW |

### Search space

| Parameter | Search space | Why |
|-----------|-------------|-----|
| Learning rate | 1e-5 to 1e-2 | Always search this |
| Num conv layers | 2-6 | Deeper = more capacity |
| Filters per layer | 16-256 (powers of 2) | GPU-friendly |
| Dropout | 0.1-0.5 | Regularization |
| Mel bins | 64 or 128 | Input resolution |
| SpecAugment intensity | time_mask / freq_mask width | How much augmentation |
| Batch size | 16, 32, 64 | Memory vs speed |

### After Optuna — record results

```
┌──────────────────────────────────────────────┐
│ OPTUNA TUNING                                │
│ Best accuracy:    ???%                       │
│ Best params:      (see study)               │
│ Improvement:      +X.X% vs baseline         │
│ N trials:         20 × 10 epochs            │
└──────────────────────────────────────────────┘
```

---

## Phase 5 — Grad-CAM for Audio 🔲

**Goal:** See what frequency and time regions the model focuses on for each animal.

**Build this file:** `grad_cam_audio.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Grad-CAM on spectrograms | `grad_cam_audio.py` — same Grad-CAM, but overlay on spectrogram | L3-M2 `saliency_and_class_activation_map/main.py` |
| Frequency importance | Which mel bins matter for dog vs cat? | New — unique to audio |
| Temporal importance | Which time steps matter? | New — when does the bark happen? |
| Per-class comparison | Dog focuses on low freq, bird on high freq? | Same pattern as NSFW per-class comparison |
| Saliency maps | Pixel-level gradient importance | L3-M2 (saliency) |

### What you'll discover

```
Dog bark:  Grad-CAM focuses on LOW frequencies + sharp temporal onset
Cat meow:  Grad-CAM focuses on MID frequencies + sustained tone
Bird:      Grad-CAM focuses on HIGH frequencies + rapid temporal patterns
Frog:      Grad-CAM focuses on narrow frequency band + repetitive pattern

This tells you what the model learned — and what to improve
```

---

## Phase 6 — Autoencoder (Reconstruct) 🔲

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

---

## Phase 7 — U-Net Autoencoder with Skip Connections 🔲

**Goal:** Better reconstruction using skip connections — same idea as ResNet but for the encoder-decoder path.

**Build this file:** `unet_autoencoder.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Skip connections (encoder → decoder) | `unet_autoencoder.py` — copy encoder features to decoder | L3-M1 `resnet/main.py` |
| U-Net architecture | `unet_autoencoder.py` — the standard architecture for generation tasks | L3-M1 (residual connections) |
| Why skips help generation | Encoder preserves fine details that decoder would lose | Same concept as Phase 5 in NSFW |
| Compare with Phase 6 | U-Net vs basic autoencoder reconstruction quality | Same comparison pattern as NSFW |

### Why skip connections matter for generation

```
Basic autoencoder:
  Encoder: fine details → pool → gone forever
  Decoder: tries to reconstruct → blurry, missing details

U-Net with skip connections:
  Encoder: conv_block1 → pool → conv_block2 → pool → ...
                    ↓                         ↓
  Decoder: ... ← up ← conv_block2' ← up ← conv_block1'
                    ↑                         ↑
                  skip                      skip
                  (fine details preserved!)
```

### After U-Net — record results

```
┌──────────────────────────────────────────────┐
│ AUTOENCODER COMPARISON                       │
│ Basic autoencoder reconstruction loss:  ???  │
│ U-Net autoencoder reconstruction loss:  ???  │
│ Audio quality (basic):                  ???/10│
│ Audio quality (U-Net):                  ???/10│
│ Skip connections helped?                Yes/No│
└──────────────────────────────────────────────┘
```

---

## Phase 8 — Conditional Generation (by Class) 🔲

**Goal:** Generate animal sounds by specifying which animal you want.

**Build this file:** `generator.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Class conditioning | `generator.py` — concatenate class embedding to latent | L3-M3 `decoder_block` (conditioning) |
| Label embedding | `generator.py` — `nn.Embedding(num_classes, embed_dim)` | L2-M3 `embeddings/main.py` |
| Conditional decoder | `generator.py` — [latent + class_emb] → decoder | L3-M2 (text conditioning in stable diffusion) |
| Audio saving | `generator.py` — `torchaudio.save()` | New |
| Listening tests | Manual — does it sound like the right animal? | New — subjective evaluation |

### Conditioning approaches

```python
# Approach A: Concatenation
class_embedding = self.embed(label)          # [batch, 64]
z_input = torch.cat([z, class_embedding])    # [batch, 128+64]
output = self.decoder(z_input)

# Approach B: Addition (projection to same size)
class_embedding = self.embed_and_project(label)  # [batch, 128]
z_input = z + class_embedding                    # [batch, 128]
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

## Phase 9 — Variational Autoencoder (Diverse Generation) 🔲

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
| Interpolation between classes | `vae.py` — z_dog → z_cat, decode each step | L3-M2 (stable diffusion latent arithmetic) |

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
│ Sound quality vs Phase 8:      Better/Worse  │
│ Latent space is organized?     Yes/No        │
│ Can interpolate between classes?  Yes/No     │
│ Interpolation sounds smooth?   Yes/No        │
└──────────────────────────────────────────────┘
```

---

## Phase 10 — Audio Quality & Evaluation 🔲

**Goal:** Move beyond "sounds okay to me" — quantify generation quality.

**Build this file:** `evaluate_gen.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Fréchet Audio Distance (FAD) | `evaluate_gen.py` — distribution distance between real & generated | New (like FID for images) |
| Spectrogram comparison | `evaluate_gen.py` — real vs generated side by side | L3-M2 `interpreting/main.py` |
| Classification-based eval | `evaluate_gen.py` — does Phase 2 classifier agree with intended class? | Uses your Phase 2 classifier! |
| t-SNE visualization | `evaluate_gen.py` — real vs generated clusters | New |
| Diversity metric | `evaluate_gen.py` — pairwise distance in latent space | New |
| MLflow logging | Log all metrics for comparison | L3-M4 `MLflow/main.py` |

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

## Phase 11 — Latent Space Mixing (Multiple Animals) 🔲

**Goal:** Mix multiple animal sounds by combining their latent representations.

**Build this file:** `latent_mixing.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Latent space arithmetic | `latent_mixing.py` — vector math in z-space | L3-M2 `stable_diffusion` (latent space concepts) |
| Interpolation | z_dog → z_cat, decode each step → hear smooth transition | L3-M2 (stable diffusion latent arithmetic) |
| Weighted mixing | 0.6×dog + 0.4×cat → hybrid sound | L3-M2 (prompt weighting concept) |
| Style transfer | Keep content, change "style" (pitch, energy) | L3-M2 (guidance_scale, negative_prompt) |
| Guided generation | Control generation by manipulating latent dimensions | L3-M2 (guidance_scale controls) |

### What you'll create

```python
# Mix two animals
z_dog = encoder(dog_sound)
z_cat = encoder(cat_sound)

# Smooth interpolation: 0% dog → 100% cat
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    z_mix = (1 - alpha) * z_dog + alpha * z_cat
    audio = decoder(z_mix)
    # Listen to the morphing!

# Weighted mix: "mostly dog with a hint of cat"
z_mix = 0.7 * z_dog + 0.3 * z_cat

# Three-way mix: "dog-cat-bird"
z_mix = 0.5 * z_dog + 0.3 * z_cat + 0.2 * z_bird
```

---

## Phase 12 — Longer & Sequential Sounds 🔲

**Goal:** Generate sounds longer than 2 seconds by predicting the next audio chunk.

**Build this file:** `sequential_generator.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Autoregressive generation | Predict next spectrogram frame given previous frames | L3-M3 `decoder_block` (Shakespeare generator) |
| Overlap-add stitching | Generate overlapping 2-sec chunks, crossfade | New |
| Sequence-to-sequence | "dog bark → pause → cat meow" as a sequence plan | L3-M3 `translation` (encoder-decoder) |
| Temperature control | Higher = more random/diverse, lower = more consistent | L3-M3 (generation temperature) |
| Causal masking | Can only attend to past audio chunks (not future) | L3-M3 `decoder_block` (causal mask) |

### Approaches

```python
# Approach A: Overlap-Add (simplest)
# Generate 2-sec chunks with 0.5s overlap, crossfade between them
chunk1 = generate(label, seed=1)  # [0.0 - 2.0s]
chunk2 = generate(label, seed=2)  # [1.5 - 3.5s]  ← overlap 0.5s
# Crossfade in overlap region → seamless 3.5s audio

# Approach B: Autoregressive (predict next chunk)
# Like Shakespeare generator but for audio
# Input: previous spectrogram → Output: next spectrogram frame
# Course reference: decoder_block/main.py

# Approach C: Sequence planner (different animals in order)
# Input: [dog, pause, cat, pause, cow] → Output: full audio scene
# Course reference: translation/main.py (English → French, but Animal → Audio)
```

---

## Phase 13 — Diffusion Refinement (Highest Quality) 🔲

**Goal:** Use a diffusion model to clean up VAE output — the same architecture as Stable Diffusion but on spectrograms.

**Build this file:** `diffusion_refine.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Forward diffusion (add noise) | `diffusion_refine.py` — gradually corrupt spectrogram | L3-M2 `stable_diffusion` (forward process) |
| Reverse diffusion (denoise) | `diffusion_refine.py` — U-Net learns to remove noise | L3-M2 (reverse process) |
| Conditional diffusion | Guide denoising toward target animal class | L3-M2 (text conditioning) |
| DDPM on spectrograms | Same as DDPM bedroom model, but for audio spectrograms | L3-M2 (DDPM pipeline) |
| Noise schedule | Linear vs cosine schedule — controls quality/speed tradeoff | L3-M2 (inference steps) |

### Pipeline (exactly like Stable Diffusion!)

```
VAE generates rough spectrogram
         │
    ┌────▼────┐
    │ ADD NOISE│  ← Forward diffusion (t steps)
    └────┬────┘
         │  noisy spectrogram
    ┌────▼────┐
    │  U-NET  │  ← Learns to predict & remove noise
    │  DENOISE│     conditioned on animal class
    └────┬────┘
         │  refined spectrogram (cleaner than VAE output)
    ┌────▼────┐
    │ GRIFFIN │  ← Convert back to audio
    │ -LIM    │
    └────┬────┘
         │
OUTPUT: High-quality animal sound
```

---

## Phase 14 — Pruning + Quantization 🔲

**Goal:** Shrink the generator for real-time inference on edge devices.

**Build this file:** `optimize.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| L1 unstructured pruning | `optimize.py` — remove smallest weights from generator | L3-M4 `pruning/main.py` |
| Fine-tune after pruning | `optimize.py` — recover quality with 3-5 epochs | L3-M4 (pruning) |
| Dynamic quantization | `optimize.py` — FP32 → INT8 for faster inference | L3-M4 `quantization/main.py` |
| ONNX export | `optimize.py` — export optimized generator | L3-M4 `ONNX/main.py` |
| Benchmark speed vs quality | `optimize.py` — measure latency and audio quality tradeoff | L3-M4 `metro_city/main.py` |

### Optimization comparison (same as NSFW)

```
┌──────────────────────────────────────────────┐
│ OPTIMIZATION PIPELINE                        │
│                                              │
│ Original generator:  ??? ms per sound        │
│ Pruned 30%:          ??? ms (faster?)        │
│ Quantized INT8:      ??? ms (faster?)        │
│ ONNX:                ??? ms (faster?)        │
│                                              │
│ Audio quality drop:  ???%                    │
│ Real-time capable?   Yes/No (<100ms)         │
└──────────────────────────────────────────────┘
```

---

## Phase 15 — Deployment (Web App) 🔲

**Goal:** Deploy a web app where users click an animal button and hear generated sounds.

**Build these files:** `export_onnx.py`, `client/server.py`, `client/frontend/`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| ONNX export (generator only) | `export_onnx.py` | L3-M4 `ONNX/main.py` |
| FastAPI audio endpoint | `client/server.py` — generate + return .wav | L3-M4 (deployment) |
| React audio player | `client/frontend/` — play generated sounds | New |
| Random seed control | `client/server.py` — different sound each click | New |
| Sampling temperature slider | `client/frontend/` — control diversity | L3-M3 (temperature) |
| Model info on UI | `client/frontend/` — show model type, size | Same as NSFW |
| MLflow model registry | Track best model, load in production | L3-M4 `MLflow/main.py` |

### UI concept

```
┌─────────────────────────────────────┐
│   🐾 Animal Sound Generator        │
│   Model: VAE-UNet  Size: 2.1 MB    │
│                                     │
│  🐶 Dog    🐱 Cat    🐔 Rooster    │
│  🐷 Pig    🐄 Cow    🐸 Frog       │
│  🦗 Cricket 🐑 Sheep 🐦 Crow      │
│                                     │
│  Temperature: ████░░ 0.7           │
│  Duration:    ██░░░░ 2.0s          │
│                                     │
│  ┌───────────────────────────────┐  │
│  │  ▶ [Generated waveform]      │  │
│  │  Duration: 2.0s              │  │
│  │                               │  │
│  │  [▶ Play] [⬇ Download]       │  │
│  └───────────────────────────────┘  │
│                                     │
│  🔀 Mix Mode:                      │
│  🐶 70% + 🐱 30%  → [Generate]    │
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
│   ├── train.py                   # Phase 2: Training pipeline (config-driven)
│   ├── evaluate.py                # Phase 2: Classifier evaluation
│   ├── transfer_audio.py          # Phase 3: Transfer learning (PANNs, VGGish)
│   ├── tuning.py                  # Phase 4: Optuna hyperparameter search
│   ├── grad_cam_audio.py          # Phase 5: Grad-CAM on spectrograms
│   ├── autoencoder.py             # Phase 6: Basic autoencoder
│   ├── unet_autoencoder.py        # Phase 7: U-Net with skip connections
│   ├── generator.py               # Phase 8: Conditional generation
│   ├── vae.py                     # Phase 9: Variational autoencoder
│   ├── evaluate_gen.py            # Phase 10: Generation quality metrics
│   ├── latent_mixing.py           # Phase 11: Latent space mixing
│   ├── sequential_generator.py    # Phase 12: Longer + sequential sounds
│   ├── diffusion_refine.py        # Phase 13: Diffusion refinement
│   ├── optimize.py                # Phase 14: Pruning + Quantization
│   ├── export_onnx.py             # Phase 15: Export to ONNX
│   └── helper_utils.py            # Shared utilities
│
├── client/
│   ├── server.py                  # Phase 15: FastAPI backend
│   ├── start.py                   # Phase 15: One-command launcher
│   └── frontend/                  # Phase 15: React frontend
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
| 2a | Audio classifier (2D CNN) | `model.py`, `train.py` | L1-M4 (CNN) | 🔲 |
| 2b | Evaluate classifier | `evaluate.py` | L3-M2 (interpreting) | 🔲 |
| 3 | Transfer learning (PANNs/VGGish) | `transfer_audio.py` | L2-M2 (transfer_learning) | 🔲 |
| 4 | Optuna hyperparameter tuning | `tuning.py` | L2-M1 (optuna) | 🔲 |
| 5 | Grad-CAM for audio | `grad_cam_audio.py` | L3-M2 (saliency_and_class_activation_map) | 🔲 |
| 6 | Autoencoder (reconstruct) | `autoencoder.py` | L3-M2 (stable_diffusion) | 🔲 |
| 7 | U-Net with skip connections | `unet_autoencoder.py` | L3-M1 (resnet) | 🔲 |
| 8 | Conditional generation | `generator.py` | L3-M3 (decoder, conditioning) | 🔲 |
| 9 | VAE (diverse generation) | `vae.py` | L3-M2 (noise, sampling) | 🔲 |
| 10 | Audio quality evaluation | `evaluate_gen.py` | L2-M1 (metrics) | 🔲 |
| 11 | Latent space mixing | `latent_mixing.py` | L3-M2 (stable diffusion) | 🔲 |
| 12 | Longer & sequential sounds | `sequential_generator.py` | L3-M3 (decoder_block, translation) | 🔲 |
| 13 | Diffusion refinement | `diffusion_refine.py` | L3-M2 (DDPM pipeline) | 🔲 |
| 14 | Pruning + Quantization | `optimize.py` | L3-M4 (pruning, quantization) | 🔲 |
| 15 | Deployment (web app) | `client/` | L3-M4 (ONNX, MLflow, deployment) | 🔲 |

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
optuna>=3.4          # Hyperparameter tuning (Phase 4)
mlflow>=2.8          # Experiment tracking (all phases)

# Audio
librosa>=0.10        # Advanced audio processing
soundfile>=0.12      # Save .wav files

# Transfer learning
panns-inference>=0.1 # PANNs pretrained models (Phase 3)

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
| ResNet18 transfer learning | PANNs/VGGish transfer learning |
| Residual connections (ResidualTunedCNN) | U-Net skip connections (encoder → decoder) |
| Grad-CAM on images | Grad-CAM on spectrograms |
| Pruning/Quantization on classifier | Pruning/Quantization on generator |

---

## Key Principle

Same as NSFW project:

```
Listen → measure → improve → listen again

Phase 1:  Understand the data (spectrograms, waveforms)
Phase 2:  Baseline classifier (what features matter?)
Phase 3:  Transfer learning (biggest accuracy jump?)
Phase 4:  Optuna tuning (push architecture to its limit)
Phase 5:  Grad-CAM (what does the model focus on?)
Phase 6:  Autoencoder (can we compress and reconstruct?)
Phase 7:  U-Net (do skip connections help?)
Phase 8:  Conditional generation (can we generate BY CLASS?)
Phase 9:  VAE (can we generate DIFFERENT sounds?)
Phase 10: Quality evaluation (how GOOD are they?)
Phase 11: Mixing (can we blend animals?)
Phase 12: Sequences (can we make longer sounds?)
Phase 13: Diffusion (can we make them even better?)
Phase 14: Optimize (can we make generation faster?)
Phase 15: Deploy (can anyone use it?)
```

Every phase builds on the previous one. The classifier from Phase 2 becomes the evaluator in Phase 10. The autoencoder from Phase 6 becomes the generator in Phase 8. The U-Net from Phase 7 becomes the denoiser in Phase 13.
