# Phase 7a — HiFi-GAN Neural Vocoder

> **Learn this after VAE (Phase 4)** — HiFi-GAN fixes what VAEs can't: converting spectrograms into crisp, natural audio.

---

## The Problem: Why Griffin-Lim Isn't Enough

```
Your VAE generates:    [mel spectrogram]  (blurred, but semantically correct)
Then you convert to:   [waveform]
```

Two ways to convert mel → waveform:

| Method | How it works | Problem |
|--------|-------------|---------|
| **Griffin-Lim** | Math: iteratively guess phase from magnitude | Adds static/noise. Sounds like "robot voice underwater." |
| **HiFi-GAN** | Neural network trained to generate phase | Learns what REAL audio sounds like. Crisp and natural. |

```
VAE mel ── Griffin-Lim ──▶ Grainy, metallic audio
       │
       └─ HiFi-GAN ────▶ Crisp, realistic audio ✨
```

**HiFi-GAN doesn't fix the spectrogram.** It fixes the **conversion**. Even a perfect VAE mel sounds mediocre through Griffin-Lim.

---

## What You'll Learn

| Concept | New or familiar? | Analogy |
|---------|----------------|---------|
| **GAN (Generative Adversarial Network)** | New | Like a forger vs. detective playing cat-and-mouse |
| **Discriminator** | New | "Is this audio real or fake?" — a classifier |
| **Adversarial loss** | New | Generator gets rewarded for fooling the discriminator |
| **Feature matching loss** | New | Match the discriminator's internal features, not just its output |
| **Mel spectrogram loss** | Familiar (like VAE reconstruction) | But now on WAVEFORMS, not spectrograms |
| **Multi-scale + Multi-period discriminators** | New | Check realism at different time resolutions |

---

## GAN Intuition: The Forger vs. The Detective

```
┌──────────────┐         fake audio         ┌──────────────┐
│  GENERATOR   │ ──────────────────────────▶ │DISCRIMINATOR │
│  (forger)    │                             │(detective)   │
│              │                             │              │
│  mel → audio │     ┌───────────────┐       │  real or     │
│              │     │               │       │  fake?       │
└──────────────┘     │   Feedback    │       │              │
      ▲              │   "this sounds│       └──────┬───────┘
      │              │   fake because│              │
      └── GRADIENT ──│   it lacks... │◀─────────────┘
                     │               │
                     └───────────────┘

They train together:
  Generator: "I'll make audio that looks real to you"
  Discriminator: "I'll learn to tell real from fake"

Over time → Generator produces audio indistinguishable from real.
```

---

## Architecture: Multi-Receptive Field Fusion (MRF)

HiFi-GAN's generator is a **1D neural vocoder** — it upsamples from mel frames to audio samples:

```
Input:  mel spectrogram [B, 64, T]    ← 64 frequency bins × time frames

              ↓  pre-net Conv1d (7)
         [B, 256, T]                   ← expand to 256 channels

              ↓  ┌─ MRF Block 1 ─┐
              ↓  │ upsample × 5  │ ← 200 samples per mel frame
              ↓  │ ResBlock(3)   │ ← fine details (transients, attack)
              ↓  │ ResBlock(7)   │ ← medium patterns (timbre)
              ↓  │ ResBlock(11)  │ ← slow modulation (pitch contour)
              ↓  │ SUM outputs   │ ← fuse all receptive fields
              ↓  └───────────────┘

              ↓  ┌─ MRF Block 2 ─┐
              ↓  │ upsample × 5  │
              ↓  │ ResBlock(3,7,11)│
              ↓  └───────────────┘

              ↓  ┌─ MRF Block 3 ─┐
              ↓  │ upsample × 4  │
              ↓  │ ResBlock(3,7,11)│
              ↓  └───────────────┘

              ↓  ┌─ MRF Block 4 ─┐
              ↓  │ upsample × 2  │
              ↓  │ ResBlock(3,7,11)│
              ↓  └───────────────┘

              ↓  post-net Conv1d (7)
              ↓  (no tanh — bounded by training loss)

Output: waveform [B, 1, T × 200]  ← 22050 Hz audio
```

### Why 3 parallel ResBlocks per upsampling stage?

```
Kernel=3:  catches FAST transients (bark attack, click)
           ┌───┐
           │   │  ← narrow window, sees sharp edges

Kernel=7:  catches MEDIUM patterns (vocal timbre, formants)
           ┌───────┐
           │       │  ← wider window, sees tone

Kernel=11: catches SLOW modulation (pitch contour, vibrato)
           ┌───────────┐
           │           │  ← widest window, sees melody

Their outputs are SUMMED → generator sees ALL time scales simultaneously.
```

### Upsampling math

```
Total upsampling factor: 5 × 5 × 4 × 2 = 200
This matches hop_length (200)

Each mel frame → 200 audio samples
44100 Hz audio → 220.5 mel frames per second
```

---

## The Discriminators: Multi-Scale + Multi-Period

HiFi-GAN uses **TWO types** of discriminators working together:

### 1. Multi-Scale Discriminator (MSD) — checks overall structure

```
Scale 0: raw audio [B, 1, T]           ← full resolution
Scale 1: avg-pool ×2 → [B, 1, T/2]     ← downsampled
Scale 2: avg-pool ×4 → [B, 1, T/4]     ← more downsampled

Each scale runs through its own CNN:
  Conv1d(1,16) → Conv1d(16,64) → Conv1d(64,128) → ... → score

Why? Audio has structure at multiple time scales.
  Scale 0: catches fine details (transients, noise)
  Scale 1: catches medium patterns (syllables)
  Scale 2: catches long patterns (pitch, rhythm)
```

### 2. Multi-Period Discriminator (MPD) — catches periodicity

```
Period=2: reshape audio as [B, 2, T/2]  ← checks 2-sample periodicity
Period=3: reshape audio as [B, 3, T/3]  ← checks 3-sample periodicity
Period=5: reshape audio as [B, 5, T/5]  ← catches vocal pitch
Period=7: reshape audio as [B, 7, T/7]  ← catches harmonics
Period=11: reshape as [B, 11, T/11]     ← catches longer patterns

Then: 2D Conv → captures cross-channel (period) patterns

Why? Speech and animal sounds are PERIODIC (pitch = repeating waveform).
MPD catches if the waveform period looks natural.
```

### Combined: 8 discriminators total

```
3 MSD scales + 5 MPD periods = 8 discriminators
Each gives: score (real/fake) + intermediate features

Generator loss = average over ALL discriminators
```

---

## The Three Losses (This is where GANs get tricky)

```python
total_generator_loss = λ_mel × L_mel + λ_fm × L_fm + λ_adv × L_adv
```

### 1. Mel Spectrogram Loss (λ_mel = 45) — THE ANCHOR

```python
# "Does the generated audio have the right frequency content?"
fake_mel = MelSpectrogram(generator(mel_input))
real_mel = MelSpectrogram(real_audio)
L_mel = L1(fake_mel, real_mel)
```

**Purpose:** Forces generator to match the INPUT mel spectrogram.
**Weight: HEAVY (45×)** — this is the PRIMARY objective.
**Without it:** generator invents any "realistic" audio (could be speech, music, noise).

> 🚨 **Critical lesson learned:** With `lambda_mel=1.0`, the generator produced
> human speech instead of dog barks. The adversarial loss dominated, so "sounds
> like real audio" mattered more than "sounds like the INPUT mel."

### 2. Feature Matching Loss (λ_fm = 2) — THE GUIDE

```python
# "Do the generator's features look like real audio features?"
real_feats = discriminator_features(real_audio)    # list of tensors
fake_feats = discriminator_features(fake_audio)    # same structure
L_fm = sum(L1(f, r.detach()) for f, r in zip(fake_feats, real_feats))
```

**Purpose:** Matches the discriminator's INTERNAL features, not just its output.
**Why it helps:** The discriminator's intermediate layers capture "what makes
audio sound real" at multiple abstraction levels. Matching these guides the
generator toward realistic waveforms.

### 3. Adversarial Loss (λ_adv = 1) — THE POLISH

```python
# "Can the discriminator tell this is fake?"
fake_scores = discriminator(fake_audio)
L_adv = -mean(fake_scores)   # generator wants scores to be HIGH (real)
```

**Purpose:** Pushes generator toward producing audio that fools the discriminator.
**Weight: SMALL (1×)** — adversarial loss is powerful and can destabilize training
if too strong. It's the polish, not the main objective.

---

## Training Loop: Generator vs. Discriminator Dance

```
Each batch:
  1. Generate fake audio from mel
  2. Train discriminator:
     - real audio → should score HIGH
     - fake audio → should score LOW
     - Update D weights
  3. Train generator:
     - fake audio → should score HIGH (fool D)
     - mel loss → should match input mel
     - feature matching → should match D's internal features
     - Update G weights
  4. Repeat
```

```python
# Discriminator step
opt_d.zero_grad()
r_score = discriminator(real_audio)
f_score = discriminator(fake.detach())  # detach — don't train G here
d_loss = hinge_loss(r_score, f_score)
d_loss.backward()
opt_d.step()

# Generator step
opt_g.zero_grad()
f_score = discriminator(fake)           # NOW train G through D
g_loss = mel_loss + fm_loss + adv_loss
g_loss.backward()
opt_g.step()
```

**Key insight:** The generator and discriminator are updated **alternately** on
the same batch. They're literally playing a game — each one's update makes the
other one's job harder.

---

## Critical Hyperparameters (Learned the Hard Way)

| Parameter | Value | Why | What happens if wrong |
|-----------|-------|-----|----------------------|
| `segment_size` | **16384** (0.74s) | Long enough for full bark/meow | 8192 (0.37s) → generator sees too little context, misses full patterns |
| `lambda_mel` | **45** | Anchors generator to input mel | 1.0 → generator drifts to speech/wrong content |
| `lambda_fm` | **2** | Guides waveform realism | 10 → overwhelms mel loss, similar drift problem |
| `lambda_adv` | **1** | Small adversarial polish | 2+ → discriminator dominates, generator ignores mel |
| `learning_rate` | **2e-4** | Standard for HiFi-GAN | Higher → unstable GAN; lower → painfully slow |
| `batch_size` | **8+** | Enough diversity per batch | < 4 → discriminator memorizes, mode collapse |
| `beta1` (Adam) | **0.8** | Less momentum for GANs | 0.9 (default) → discriminator overshoots |

### The Mel Loss Lesson

```
WRONG: lambda_mel=1.0, lambda_adv=2.0
→ Generator: "I'll make realistic audio" (ignores input mel)
→ Result: Human speech instead of dog barks

CORRECT: lambda_mel=45.0, lambda_adv=1.0
→ Generator: "I must match THIS mel spectrogram"
→ Discriminator: "But make it sound natural"
→ Result: Correct content + natural waveform
```

**Rule of thumb:** Mel loss should be the HEAVIEST weight. The discriminator
polishes what the mel loss constrains.

---

## Training Strategy: Meltrain → Full GAN

### Step 1: Mel-Only Pretraining (`mode="meltrain"`)

```
Train generator WITHOUT discriminator.
Loss = L_mel + L_time (reconstruction only)

Why?
  - Generator learns "mel → audio" mapping from scratch
  - No adversarial instability in early training
  - Converges to a reasonable baseline in ~15-30 epochs
  - If mel=0.0000 → data loading bug (not a model issue)
```

### Step 2: Full GAN Training (`mode="train"`)

```
Load meltrain checkpoint (optional).
Train generator + discriminator together.
Loss = 45×L_mel + 2×L_fm + 1×L_adv

Why?
  - Generator already knows mel → audio
  - Discriminator refines waveform quality
  - Feature matching adds naturalness
  - ~20-30 epochs for convergence
```

### What to watch during training

```
Healthy:
  G mel loss: drops steadily (6→3→2→1.5→1.0...)
  D loss: hovers around 1-2 (hinge loss equilibrium)
  Val mel: similar to train mel (no overfitting)

Unhealthy:
  D loss → 0: Discriminator is too strong → generator can't learn
  D loss → very high: Discriminator collapsed → adversarial loss meaningless
  G mel loss stops dropping: Generator plateaued → need more epochs or LR warmup
  D loss oscillates wildly: Learning rate too high
```

---

## HiFi-GAN vs. Your VAE: Different Problems, Same Goal

```
Your VAE:
  Input: class label (e.g. "dog")
  Output: mel spectrogram (blurred, needs conversion)
  Trained with: MSE reconstruction + KL + classifier loss
  Problem it solves: "What should a dog sound look like?"

HiFi-GAN:
  Input: mel spectrogram (from VAE or real)
  Output: audio waveform (crisp, natural)
  Trained with: mel loss + feature matching + adversarial loss
  Problem it solves: "How do I convert this mel into audio?"

Together:
  "dog" → VAE → mel → HiFi-GAN → waveform ✨
```

**VAE and HiFi-GAN are complementary.** VAE generates the spectrogram (the
"what"). HiFi-GAN converts it to audio (the "how").

---

## Why This is Industry Standard

HiFi-GAN was published by Kong, Kim, and Bae (2020) at NeurIPS. It's used in:

- **TTS systems:** VITS, YourTTS, Coqui — all use HiFi-GAN as the vocoder
- **Voice cloning:** Matcha-TTS, Bark — mel → waveform via HiFi-GAN
- **Music generation:** AudioCraft (Meta) — similar GAN vocoder approach
- **Speech enhancement:** SEGAN variants — GAN-based audio restoration

The MRF architecture and multi-discriminator design became the **standard**
because it produces high-quality audio with efficient inference.

---

## Key Lessons from Our Implementation

### 1. Data loading bugs are the #1 cause of "loss=0"

If training shows `mel=0.0000` from epoch 1, the audio files aren't loading.
Our fix: absolute paths, soundfile fallback, data validation before training.

```python
# Before training, always verify:
_check = next(iter(train_loader))
assert _check.abs().max() > 1e-6, "🚨 ALL ZEROS — check data loading"
```

### 2. torchaudio 2.11+ requires FFmpeg shared libraries

On systems without FFmpeg (like some Linux servers), `torchaudio.load()` fails.
**Fix:** Fall back to `soundfile` which reads WAV natively without FFmpeg.

```python
def load_audio(path):
    try: return torchaudio.load(path)
    except: return soundfile.read(path)  # no FFmpeg needed
```

### 3. Multiprocessing workers can silently fail on CUDA

With `num_workers>0`, relative paths may break in spawned processes.
**Fix for meltrain:** Use `num_workers=0` (data loading is fast enough).
**Fix for GAN:** Use absolute paths + soundfile fallback.

### 4. The discriminator is a powerful teacher but easily overpowers

With strong adversarial loss, the generator produces "realistic" but
**wrong** content. The mel loss must anchor the generator to the input.

**Loss weight ordering matters:**
```
lambda_mel (45) > lambda_fm (2) > lambda_adv (1)
     ↑                  ↑               ↑
   PRIMARY            GUIDE          POLISH
```

### 5. Segment size must match the signal structure

```
8192  samples (0.37s) → too short for full bark
16384 samples (0.74s) → captures most bark patterns
44100 samples (2.00s) → ideal for long sounds, slower training
```

---

## Next Steps After HiFi-GAN

Once HiFi-GAN training is stable and audio sounds good:

| Phase | What | Builds on |
|-------|------|-----------|
| **7b: Diffusion** | Sharpen the VAE's blurry mel output | HiFi-GAN (vocoder still needed) |
| **7c: Sequential** | Generate long sounds, chain animals | HiFi-GAN (handles any length) |
| **7d: Latent mixing** | Blend animals in z-space | HiFi-GAN (converts any mel) |
| **7e: UI v2** | Add vocoder selector to web app | All of Phase 7 |

The full pipeline:
```
"dog" → VAE → [blurry mel] → Diffusion → [sharp mel] → HiFi-GAN → [crisp audio]
```

---

## References

- **Kong, Kim, Bae (2020):** "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" — [Paper](https://arxiv.org/abs/2010.05646)
- **Original repo:** [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- **Kazuki's explanation:** [Understanding HiFi-GAN](https://kazemnejad.com/blog/2024-03-19-hifi-gan/)
- **GAN basics:** [Ian Goodfellow's original paper (2014)](https://arxiv.org/abs/1406.2661)
- **Feature matching:** [Salimans et al. (2016) Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
