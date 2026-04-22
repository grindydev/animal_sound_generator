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
    │   • 2D CNN on spectrograms (your NSFW CNN skills transfer!)
    │   • Confusion matrix, accuracy, per-class analysis
    │   • This classifier becomes your evaluation tool later
    ▼
 Phase 3 — Autoencoder (learn to reconstruct)
    │   • Encoder: compress spectrogram → latent vector
    │   • Decoder: latent vector → reconstruct spectrogram
    │   • The decoder is the GENERATOR you'll build on
    ▼
 Phase 4 — Conditional VAE (generate by class + diverse)
    │   • Add class conditioning: specify which animal to generate
    │   • VAE sampling: same class → different sounds each time
    │   • Latent space exploration, interpolation between animals
    ▼
 Phase 5 — Audio Quality & Evaluation
    │   • Fréchet Audio Distance, classification agreement
    │   • Latent space visualization (t-SNE)
    │   • Compare real vs generated spectrograms
    ▼
 Phase 6 — Advanced Generation (longer, mixed, refined)
    │   • Mix multiple animals in latent space
    │   • Generate longer sequences (autoregressive / overlap-add)
    │   • Diffusion refinement for higher quality
    ▼
 Phase 7 — Re-practice All Techniques on the Generator
    │   • Transfer learning, Optuna, skip connections (U-Net)
    │   • Grad-CAM on spectrograms, pruning, quantization
    │   • All applied to YOUR generative model
    ▼
 Phase 8 — Deployment (Web App)
        • Export generator, FastAPI + React web app
        • Click animal button → generate unique sound
        • Mix mode, temperature control, history
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
| Audio augmentation | Time shift, noise, pitch shift, SpecAugment | L1-M3 (augmentation) |

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

**Goal:** Build a classifier as a warm-up (same pattern as NSFW Phase 1-2). This teaches you what features matter for animal sounds, AND you'll reuse this classifier in Phase 5 to evaluate your generator.

**Build these files:** `model.py`, `train.py`, `evaluate.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| 2D CNN on spectrograms | `model.py` — same CNN architecture as NSFW | L1-M4 `cnn/main.py` |
| Config-driven training | `train.py` — CONFIG dict | Same as NSFW |
| Confusion matrix + F1 | `evaluate.py` | Same as NSFW |
| MLflow tracking | `train.py` — log params, metrics | L3-M4 `MLflow/main.py` |
| Early stopping + LR scheduler | `train.py` | L2-M1 `scheduler/main.py` |

### After classification — record results

```
┌──────────────────────────────────────────────┐
│ AUDIO CLASSIFIER BASELINE                    │
│ Test Accuracy:  ??%                          │
│ Hardest classes: ???                         │
│ Easy classes:    ???                         │
│ Best model saved for Phase 5 evaluation      │
└──────────────────────────────────────────────┘
```

---

## Phase 3 — Autoencoder (Reconstruct) 🔲

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

## Phase 4 — Conditional VAE (Generate by Class) 🔲

**Goal:** Generate animal sounds by specifying which animal, with diversity.

**Build this file:** `vae.py`

### What you'll practice

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Variational Autoencoder | `vae.py` — encoder outputs μ and σ | L3-M2 (diffusion noise concepts) |
| Reparameterization trick | `vae.py` — `z = μ + σ * ε` where ε ~ N(0,1) | New — key to differentiable sampling |
| KL divergence loss | `vae.py` — keeps latent space organized | New — regularization for the latent space |
| Class conditioning | `vae.py` — `nn.Embedding(num_classes, embed_dim)` | L2-M3 `embeddings/main.py` |
| Conditional decoder | `vae.py` — [z + class_emb] → decoder → spectrogram | L3-M2 (text conditioning in stable diffusion) |
| Sampling at inference | `vae.py` — random z → unique generation each time | L3-M2 (noise → denoise → image) |
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

```python
# Approach A: Concatenation
class_embedding = self.embed(label)          # [batch, 64]
z_input = torch.cat([z, class_embedding])    # [batch, 128+64]
output = self.decoder(z_input)

# Approach B: Addition (projection to same size)
class_embedding = self.embed_and_project(label)  # [batch, 128]
z_input = z + class_embedding                    # [batch, 128]
output = self.decoder(z_input)
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

## Phase 5 — Audio Quality & Evaluation 🔲

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

---

## Phase 6 — Advanced Generation 🔲

**Goal:** Push beyond basic 2-second single-animal generation.

**Build these files:** `latent_mixing.py`, `sequential_generator.py`, `diffusion_refine.py`

### 6a — Latent Space Mixing (Multiple Animals)

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

### 6b — Longer & Sequential Sounds

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Autoregressive generation | Predict next audio chunk from previous | L3-M3 `decoder_block` (Shakespeare generator) |
| Overlap-add stitching | Crossfade between generated chunks | New |
| Sequence planning | "dog bark → pause → cat meow" as a sequence | L3-M3 `translation` (seq2seq) |
| Temperature control | Higher = more random, lower = more consistent | L3-M3 (generation temperature) |
| Causal masking | Can only attend to past chunks (not future) | L3-M3 `decoder_block` (causal mask) |

### 6c — Diffusion Refinement

| Concept | Where | Course reference |
|---------|-------|-----------------|
| Forward diffusion (add noise) | Gradually corrupt spectrogram | L3-M2 `stable_diffusion` (forward process) |
| Reverse diffusion (denoise) | U-Net learns to remove noise | L3-M2 (reverse process) |
| Conditional diffusion | Guide denoising toward target animal class | L3-M2 (text conditioning) |
| DDPM on spectrograms | Same pipeline as DDPM bedroom model | L3-M2 (DDPM pipeline) |

```
VAE output (rough) → add noise → U-Net denoise → clean spectrogram → audio

Exactly what Stable Diffusion does, but on spectrograms instead of images!
```

### After advanced generation — record results

```
┌──────────────────────────────────────────────┐
│ ADVANCED GENERATION                          │
│ Dog→Cat interpolation smooth?   Yes/No       │
│ 3-way mix sounds natural?       Yes/No       │
│ Longest sequence generated:     ??? seconds   │
│ Diffusion quality improvement:  +???%        │
│ Best approach overall:          ???           │
└──────────────────────────────────────────────┘
```

---

## Phase 7 — Re-practice All Techniques on the Generator 🔲

**Goal:** Apply every technique from your NSFW project (and PyTorch course) to the generative model. Same concepts, different domain — this solidifies your understanding.

Each sub-phase takes your working VAE generator and improves it with a technique you've already learned:

### 7a — Transfer Learning for the Generator

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

### 7b — Optuna Tuning for the Generator

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

### 7c — Skip Connections (U-Net Architecture)

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

**Compare:** U-Net reconstruction quality vs basic autoencoder. Same lesson as NSFW — skip connections preserve information that would otherwise be lost.

---

### 7d — Grad-CAM on Spectrograms

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

### 7e — Pruning + Quantization on the Generator

**You learned:** Pruning removes small weights, quantization shrinks to INT8 (NSFW Phase 7b/7c, L3-M4)  
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

### Phase 7 Summary

After completing all sub-phases, compare everything:

```
┌──────────────────────────────────────────────────────────┐
│ GENERATOR COMPARISON (Audio Quality / 10)                │
│                                                          │
│ Phase 4:  Basic VAE (from scratch)               ???/10  │
│ Phase 7a: VAE + PANNs transfer encoder          ???/10  │
│ Phase 7b: VAE + Optuna-tuned architecture        ???/10  │
│ Phase 7c: U-Net VAE + skip connections           ???/10  │
│ Phase 7d: (diagnostic — not a model change)         —    │
│ Phase 7e: Pruned + Quantized                     ???/10  │
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

---

## Phase 8 — Deployment (Web App) 🔲

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
│   Model: VAE-UNet  Size: ??? MB    │
│                                     │
│  🐶 Dog    🐱 Cat    🐔 Rooster    │
│  🐷 Pig    🐄 Cow    🐸 Frog       │
│  🦗 Cricket 🐑 Sheep 🐦 Crow      │
│                                     │
│  Temperature: ████░░ 0.7           │
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
│   ├── model.py                   # Phase 2: Audio classifier (2D CNN)
│   ├── train.py                   # Phase 2: Training pipeline
│   ├── evaluate.py                # Phase 2: Classifier evaluation
│   ├── autoencoder.py             # Phase 3: Basic autoencoder
│   ├── vae.py                     # Phase 4: Conditional VAE (generator)
│   ├── evaluate_gen.py            # Phase 5: Generation quality metrics
│   ├── latent_mixing.py           # Phase 6a: Latent space mixing
│   ├── sequential_generator.py    # Phase 6b: Longer + sequential sounds
│   ├── diffusion_refine.py        # Phase 6c: Diffusion refinement
│   ├── transfer_generator.py      # Phase 7a: Transfer learning on generator
│   ├── tuning.py                  # Phase 7b: Optuna for generator
│   ├── unet_vae.py                # Phase 7c: U-Net skip connections
│   ├── grad_cam_audio.py          # Phase 7d: Grad-CAM on spectrograms
│   ├── optimize.py                # Phase 7e: Pruning + Quantization
│   ├── export_onnx.py             # Phase 8: Export to ONNX
│   └── helper_utils.py            # Shared utilities
│
├── client/
│   ├── server.py                  # Phase 8: FastAPI backend
│   ├── start.py                   # Phase 8: One-command launcher
│   └── frontend/                  # Phase 8: React frontend
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
| 2 | Audio classifier baseline | `model.py`, `train.py`, `evaluate.py` | L1-M4 (CNN) | 🔲 |
| 3 | Autoencoder (reconstruct) | `autoencoder.py` | L3-M2 (stable_diffusion) | 🔲 |
| 4 | Conditional VAE (generate by class) | `vae.py` | L2-M3 (embeddings), L3-M2 (conditioning) | 🔲 |
| 5 | Audio quality evaluation | `evaluate_gen.py` | L2-M1 (metrics), L3-M2 (interpreting) | 🔲 |
| 6a | Latent space mixing | `latent_mixing.py` | L3-M2 (stable diffusion latent space) | 🔲 |
| 6b | Longer & sequential sounds | `sequential_generator.py` | L3-M3 (decoder_block, translation) | 🔲 |
| 6c | Diffusion refinement | `diffusion_refine.py` | L3-M2 (DDPM pipeline) | 🔲 |
| 7a | Transfer learning (PANNs encoder) | `transfer_generator.py` | L2-M2 (transfer_learning) | 🔲 |
| 7b | Optuna tuning (generator architecture) | `tuning.py` | L2-M1 (optuna) | 🔲 |
| 7c | U-Net skip connections | `unet_vae.py` | L3-M1 (resnet) | 🔲 |
| 7d | Grad-CAM on spectrograms | `grad_cam_audio.py` | L3-M2 (saliency_and_class_activation_map) | 🔲 |
| 7e | Pruning + Quantization | `optimize.py` | L3-M4 (pruning, quantization) | 🔲 |
| 8 | Deployment (web app) | `client/` | L3-M4 (ONNX, MLflow, deployment) | 🔲 |

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
optuna>=3.4          # Phase 7b: Hyperparameter tuning
mlflow>=2.8          # All phases: Experiment tracking

# Audio
librosa>=0.10        # Advanced audio processing
soundfile>=0.12      # Save .wav files

# Transfer learning
panns-inference>=0.1 # Phase 7a: PANNs pretrained models

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
| ResNet18 transfer learning | PANNs transfer learning (Phase 7a) |
| ResidualTunedCNN skip connections | U-Net skip connections (Phase 7c) |
| Grad-CAM on images | Grad-CAM on spectrograms (Phase 7d) |
| Optuna for classifier | Optuna for generator (Phase 7b) |
| Pruning/Quantization on classifier | Pruning/Quantization on generator (Phase 7e) |

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
Phase 6:  Advanced (mixing, longer, diffusion)
Phase 7:  Re-practice ALL techniques on the generator
Phase 8:  Deploy (can anyone use it?)
```

Every phase builds on the previous one. The classifier from Phase 2 becomes the evaluator in Phase 5. The autoencoder from Phase 3 becomes the generator in Phase 4. Phase 7 applies every technique from your NSFW project to the new domain — same concepts, deeper understanding.
