# Phase 7a — HiFi-GAN Neural Vocoder

> **Learn this after VAE (Phase 4)** — HiFi-GAN fixes what VAEs can't: converting spectrograms into crisp, natural audio.

---

## The Problem: Why Griffin-Lim Isn't Enough

Your VAE generates a **mel spectrogram** (a 2D image of frequencies over time).
To hear it, you need to convert back to **waveform** (a 1D signal of air pressure over time).

Two ways to convert mel → waveform:

| Method | How it works | Sound quality | Training needed? |
|--------|-------------|---------------|-----------------|
| **Griffin-Lim** | Math formula: iteratively guess phase from magnitude | "Robot underwater" — grainy, metallic | No — just a function call |
| **HiFi-GAN** | Neural network that learns what real audio looks like | Crisp, natural, realistic | Yes — 30-60 epochs of training |

```
Your VAE mel ── Griffin-Lim ──▶ Grainy, metallic audio (usable for evaluation)
            │
            └─ HiFi-GAN ────▶ Crisp, natural audio ✨ (production quality)
```

**HiFi-GAN doesn't fix the spectrogram.** It fixes the **conversion from spectrogram to audio**.
Even a perfect VAE mel sounds mediocre through Griffin-Lim.

---

## What You'll Learn

| Concept | Familiar or new? | What it means |
|---------|-----------------|---------------|
| **GAN (Generative Adversarial Network)** | New | Two networks play against each other |
| **Generator** | Familiar (your VAE decoder!) | Converts mel spectrogram → waveform |
| **Discriminator** | New (like a classifier) | Judges: "Is this audio real or fake?" |
| **Adversarial loss** | New | Generator gets rewarded for fooling the discriminator |
| **Feature matching loss** | New | Match the discriminator's internal features, not just its output |
| **Mel spectrogram loss** | Familiar (like VAE reconstruction) | Forces generator to match the input mel |

---

## GAN Intuition: The Forger vs. The Detective

```
┌──────────────┐     fake audio        ┌──────────────┐
│  GENERATOR   │ ─────────────────────▶ │DISCRIMINATOR │
│  (the forger)│                        │(the detective)│
│              │                        │              │
│  "I'll make  │                        │  "This audio │
│  audio that  │                        │   looks REAL │
│  sounds real │                        │   to me!"    │
│  to you!"    │                        │              │
└──────────────┘                        └──────┬───────┘
       ▲                                       │
       │  "You fooled me this time..."         │
       │  "...but next time I'll catch you!"   │
       │  (feedback = gradient)                │
       │                                       │
       └───────────────────────────────────────┘

They train together in a loop:

  Round 1: Generator makes bad fake → Discriminator easily catches it
  Round 2: Generator gets better → Discriminator struggles more
  Round 3: Discriminator improves → catches new tricks
  ...
  Round N: Generator produces audio indistinguishable from real!
```

**Real-world analogy:** Think of a forger painting fake art and an art expert
learning to detect fakes. Each round, the forger gets better at faking, and the
expert gets better at detecting. Eventually, the forger's work is so good that
even experts can't tell the difference.

---

## Architecture: Multi-Receptive Field Fusion (MRF)

HiFi-GAN's generator is a **1D neural network** that upsamples from mel frames
to audio samples. Here's the full pipeline with actual code references:

### The Big Picture

```
Input:  mel spectrogram [B, 64, T]
                     (64 frequency bins × T time frames)

              ↓  Step 1: Pre-net (expand channels)
              │    Code: generator.py → HiFiGANGenerator.__init__()
              │    Layer: self.pre_conv = Conv1d(64 → 256, kernel=7)
         [B, 256, T]

              ↓  Step 2: MRF Block 1 (upsample × 5)
              │    Code: generator.py → MRFBlock.__init__()
              │    Layer: ConvTranspose1d(256 → 128, stride=5)
              │    Then: 3 parallel ResBlocks (kernel 3, 7, 11)
              │    Then: SUM all 3 outputs
         [B, 128, T × 5]

              ↓  Step 3: MRF Block 2 (upsample × 5)
              │    ConvTranspose1d(128 → 64, stride=5)
              │    3 parallel ResBlocks (kernel 3, 7, 11)
              │    SUM outputs
         [B, 64, T × 5 × 5]

              ↓  Step 4: MRF Block 3 (upsample × 4)
              │    ConvTranspose1d(64 → 32, stride=4)
              │    3 parallel ResBlocks (kernel 3, 7, 11)
              │    SUM outputs
         [B, 32, T × 5 × 5 × 4]

              ↓  Step 5: MRF Block 4 (upsample × 2)
              │    ConvTranspose1d(32 → 16, stride=2)
              │    3 parallel ResBlocks (kernel 3, 7, 11)
              │    SUM outputs
         [B, 16, T × 5 × 5 × 4 × 2]
         [B, 16, T × 200]   ← 200 = hop_length

              ↓  Step 6: Post-net (compress to 1 channel = audio)
              │    Code: generator.py → HiFiGANGenerator.__init__()
              │    Layer: self.post_conv = Conv1d(16 → 1, kernel=7)

Output: waveform [B, 1, T × 200]   ← mono audio at 22,050 Hz
```

### Math notation guide (for people who hate math)

Whenever you see tensor shapes like `[B, 64, T]`, here's what each letter means:

| Symbol | What it means | Example |
|--------|--------------|---------|
| **B** | Batch size — how many audio clips processed at once | B=8 means 8 clips |
| **64** | Number of mel frequency bins (n_mels) | Always 64 in this project |
| **T** | Time frames — how many mel frames in the clip | Varies by clip length |
| **200** | Upsampling factor — each mel frame becomes 200 audio samples | = hop_length |

So `[B, 64, T] → [B, 1, T × 200]` means:
- **Input:** B audio clips, each with 64 frequency bins and T time frames
- **Output:** B audio clips, each with 1 channel (mono) and T×200 audio samples

---

### Deep Dive: ResBlock (the building block)

**File:** `generator.py` — class `ResBlock`

```python
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        self.convs = nn.ModuleList()
        for d in dilations:  # dilations = (1, 3, 5)
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=1),
            ))
```

**What it does:**

```
Input x ────────────┐
                    │ (this is the "residual" — skip connection)
                    │
   x → LeakyReLU ──▶ Conv(dilation=d) ──▶ LeakyReLU ──▶ Conv(dilation=1)
                                                    │
                                                    ▼
                                            x + output (add them)
                                                    │
                                            new x → next ResBlock
```

**What is "dilation"?**

Think of dilation as **spacing between kernel elements**:

```
kernel_size = 3

Normal conv (dilation=1):     ● ● ●    ← looks at 3 adjacent samples
                              ─────
                                ↑

Dilated conv (dilation=2):    ●   ●   ●  ← looks at samples 2 apart
                              ────────
                                ↑

Dilated conv (dilation=3):    ●     ●     ●  ← looks at samples 3 apart
                              ──────────
                                ↑

Why dilation? It lets a small kernel "see" a wider area without more parameters.
```

**Each ResBlock has 3 parallel conv paths with dilation = 1, 3, 5:**

```
Dilation 1:  ● ● ●          ← catches fine details (transients, clicks)
Dilation 3:  ●   ●   ●      ← catches medium patterns (vocal timbre)
Dilation 5:  ●     ●     ●  ← catches slower patterns (pitch contour)
             │               │              │
             └───── SUM these outputs together ─────┘
```

**Code reference:** `generator.py` line 62-68
```python
for conv in self.convs:
    residual = x          # save the original x
    x = conv(x)           # pass through the conv path
    x = x + residual      # add original x back (residual connection)
```

The residual connection (`x = x + residual`) is important: it prevents the signal
from degrading through many layers. Think of it like a "safety net" — if the
convolution messes up, the original signal is still there.

---

### Deep Dive: MRFBlock (one upsample stage)

**File:** `generator.py` — class `MRFBlock`

```python
class MRFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ...):
        # Step A: Upsample + reduce channels
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_rate,    # this controls the upsampling factor
            padding=(upsample_kernel_size - upsample_rate) // 2,
        )

        # Step B: 3 parallel ResBlock paths (different kernel sizes)
        self.resblocks = nn.ModuleList()
        for ks, ds in zip(kernel_sizes, dilations):
            # ks = 3, 7, 11
            # ds = (1,3,5), (1,3,5), (1,3,5)
            self.resblocks.append(ResBlock(out_channels, ks, ds))
```

**What it does:**

```
Input [B, in_channels, T_in]
              │
              ▼
    ┌─────────────────┐
    │ ConvTranspose1d │  ← UP-SAMPLE: stretches time axis
    │  stride=5       │     e.g., T_in=41 → T_out=205
    │  channels halve │     e.g., 256 → 128
    └────────┬────────┘
             │
             ▼
         [B, out_channels, T_out]
             │
        ┌────┴────┬────┐
        │         │    │        3 PARALLEL PATHS:
        ▼         ▼    ▼
    ┌──────┐ ┌──────┐ ┌──────┐
    │ResBlk│ │ResBlk│ │ResBlk│   Each has a DIFFERENT kernel size:
    │ksize=│ │ksize=│ │ksize=│
    │   3  │ │   7  │ │  11  │
    └──┬───┘ └──┬───┘ └──┬───┘
       │        │        │
       ▼        ▼        ▼
       └────────┴────────┘
                │
                ▼       SUM all 3 outputs together

Output: [B, out_channels, T_out]
```

**Why 3 different kernel sizes in parallel?**

```
Kernel=3 (small window):
  ●●●
  Catches FAST, sharp changes:
  → bark attack, click, pop
  → very fine details

Kernel=7 (medium window):
  ●●●●●●●
  Captures MEDIUM patterns:
  → vocal timbre (how a voice "sounds")
  → formants (resonant frequencies)

Kernel=11 (large window):
  ●●●●●●●●●●●
  Captures SLOW patterns:
  → pitch contour (rising/falling tone)
  → vibrato, modulation

All 3 run on the SAME signal simultaneously.
Then their outputs are SUMMED.
The generator gets ALL time scales at once!
```

**Code reference:** `generator.py` line 113-118
```python
def forward(self, x):
    x = self.upsample(x)          # upsample + reduce channels
    outputs = []
    for resblock in self.resblocks:
        outputs.append(resblock(x))  # 3 paths, 3 outputs
    return sum(outputs)             # merge them
```

---

### The Full Generator Forward Pass

**File:** `generator.py` — class `HiFiGANGenerator`, method `forward()`

```python
def forward(self, mel, target_length=None):
    if mel.dim() == 4:
        mel = mel.squeeze(1)    # [B, 1, 64, T] → [B, 64, T]

    x = self.pre_conv(mel)      # Step 1: expand channels 64 → 256
    for mrf in self.mrf_blocks: # Step 2: 4 MRF blocks (each upsamples)
        x = mrf(x)
    x = self.post_conv(x)       # Step 3: compress to 1 channel = audio

    # Trim or pad to exact length if requested
    if target_length is not None and x.shape[-1] != target_length:
        if x.shape[-1] > target_length:
            x = x[..., :target_length]  # chop the end
        else:
            x = F.pad(x, (0, target_length - x.shape[-1]))  # pad with zeros

    return x
```

**Shape evolution (concrete example):**

```
Input mel:  [8, 64, 41]     ← 8 clips, 64 freq bins, 41 time frames (~0.37s)

pre_conv:   [8, 256, 41]    ← expanded to 256 channels

MRF block 1: [8, 128, 205]  ← upsample × 5, channels halve (256→128)
MRF block 2: [8, 64, 1025]  ← upsample × 5, channels halve (128→64)
MRF block 3: [8, 32, 4100]  ← upsample × 4, channels halve (64→32)
MRF block 4: [8, 16, 8200]  ← upsample × 2, channels halve (32→16)

post_conv:  [8, 1, 8200]    ← compressed to 1 channel = audio waveform

Output: 8 audio clips, each 8200 samples = 0.37 seconds at 22,050 Hz
```

**Why does the output length = input frames × 200?**

```
Total upsampling = 5 × 5 × 4 × 2 = 200

This number is NOT random — it equals hop_length from config.py:
  hop_length = 200

Why? When you compute a mel spectrogram from audio:
  audio_samples / hop_length = mel_frames

So to go back:
  mel_frames × hop_length = audio_samples

41 mel frames × 200 = 8,200 audio samples ✓
```

---

## The Discriminators: Multi-Scale + Multi-Period

HiFi-GAN uses **TWO types** of discriminators working together.

### 1. Multi-Period Discriminator (MPD) — checks waveform structure

**File:** `discriminator.py` — class `MultiPeriodDiscriminator`, `PeriodDiscriminator`

**What it does:** It folds the audio waveform into a 2D grid at different periods,
then uses 2D convolutions to check if each fold looks "natural."

**How folding works:**

```
Audio: [B, 1, 10]   ← 10 audio samples

Period=2 (fold into 2 columns):
  [1, 2]    [3, 4]    [5, 6]    [7, 8]    [9, 10]
  → reshape to [B, 1, 2, 5]
  → becomes a 2D image with 2 rows and 5 columns

Period=5 (fold into 5 columns):
  [1, 2, 3, 4, 5]    [6, 7, 8, 9, 10]
  → reshape to [B, 1, 5, 2]
  → becomes a 2D image with 5 rows and 2 columns
```

**Why fold?** Audio has repeating patterns (pitch = repeating waveform). When you
fold at the right period, repeating patterns line up in columns and 2D convolutions
can easily detect if the repetition looks natural or artificial.

**5 periods used:**

| Period | What it catches | Frequency range |
|--------|----------------|-----------------|
| 2 | Very fast oscillations | ~11 kHz |
| 3 | Fast oscillations | ~7 kHz |
| 5 | Mid oscillations | ~4.4 kHz |
| 7 | Medium oscillations | ~3 kHz |
| 11 | Slow oscillations | ~2 kHz |

**Code reference:** `discriminator.py` line 44-50
```python
def forward(self, x):
    B, C, L = x.shape

    # Pad so length is divisible by period
    if L % self.period != 0:
        n_pad = self.period - (L % self.period)
        x = F.pad(x, (0, n_pad), mode="reflect")

    # Reshape: [B, 1, L] → [B×period, 1, 1, L/period]
    x = x.view(B, C, self.period, L // self.period)
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(B * self.period, 1, L // self.period).unsqueeze(2)
```

---

### 2. Multi-Scale Discriminator (MSD) — checks overall structure

**File:** `discriminator.py` — class `MultiScaleDiscriminator`, `ScaleDiscriminator`

**What it does:** It downsamples the audio to different resolutions, then runs
1D convolutions on each scale.

**How it works:**

```
Scale 0: raw audio [B, 1, L]           ← full resolution, catches fine details
Scale 1: average pool × 2 → [B, 1, L/2]  ← downsampled, catches medium patterns
Scale 2: average pool × 4 → [B, 1, L/4]  ← more downsampled, catches overall shape
```

**Why multiple scales?**

```
Scale 0 (full resolution):
  ● ● ● ● ● ● ● ●
  Catches tiny glitches: clicks, pops, noise artifacts

Scale 1 (half resolution):
  ●   ●   ●   ●
  Catches syllable structure: consonant → vowel transitions

Scale 2 (quarter resolution):
  ●       ●
  Catches overall envelope: loud → quiet patterns
```

**Our implementation uses MPD only:**

```python
# discriminator.py — class Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        self.mpd = MultiPeriodDiscriminator()

    def forward(self, x):
        return self.mpd(x)  # 5 period discriminators, no MSD
```

**Combined:** 5 discriminators total (one for each period). Each gives:
- **score** → a number saying "real or fake"
- **features** → intermediate layer outputs for feature matching loss

---

## The Three Losses (This Is Where GANs Get Tricky)

```python
total_generator_loss = λ_mel × L_mel + λ_fm × L_fm + λ_adv × L_adv
```

**Math notation guide for loss equations:**

| Symbol | What it means | Example |
|--------|--------------|---------|
| **λ** (lambda) | A weight — how important this loss is | λ_mel=45 means mel loss is 45× stronger |
| **L_mel** | Mel spectrogram loss (a number) | Lower = generated audio matches input mel better |
| **L_fm** | Feature matching loss (a number) | Lower = generated audio features look more like real audio |
| **L_adv** | Adversarial loss (a number) | Lower = discriminator thinks the audio is real |
| **×** | Multiplication | λ_mel × L_mel means: multiply the loss by the weight |

### Loss 1: Mel Spectrogram Loss (λ_mel = 45) — THE ANCHOR

**File:** `losses.py` — class `MelL1Loss`

```python
class MelL1Loss(nn.Module):
    def forward(self, fake_audio, real_audio):
        fake_mel = self.mel_transform(fake_audio.squeeze(1))
        real_mel = self.mel_transform(real_audio.squeeze(1))
        return F.l1_loss(fake_mel, real_mel)
```

**What it does:**

```
1. Take generated audio → compute its mel spectrogram
2. Take real audio → compute its mel spectrogram
3. Compare: how different are they? (L1 = absolute difference)
4. Lower difference = better
```

**In plain English:** "Does the generated audio have the SAME frequency content
as the real audio?"

**Why the weight is 45 (VERY high):**

This is the MOST important loss. It forces the generator to match the **input
mel spectrogram**. Without a strong mel loss, the generator can invent ANY
"realistic" audio — it might generate human speech, music, or noise instead
of a dog bark.

> 🚨 **What we learned the hard way:** With λ_mel=1.0, the generator produced
> human speech instead of dog barks. The adversarial loss dominated, so
> "sounds like real audio" mattered more than "sounds like the INPUT mel."
>
> **Fix:** Set λ_mel=45 so mel matching is the PRIMARY objective.

---

### Loss 2: Feature Matching Loss (λ_fm = 2) — THE GUIDE

**File:** `losses.py` — function `feature_matching_loss()`

```python
def feature_matching_loss(real_features, fake_features):
    loss = 0.0
    count = 0
    for real_group, fake_group in zip(real_features, fake_features):
        for r, f in zip(real_group, fake_group):
            loss += F.l1_loss(f, r.detach())  # compare feature by feature
            count += 1
    return loss / max(count, 1)
```

**What it does:**

```
Discriminator has multiple internal layers.
Each layer produces "features" — intermediate representations of the audio.

For REAL audio:  discriminator produces features = [f1_real, f2_real, f3_real, ...]
For FAKE audio:  discriminator produces features = [f1_fake, f2_fake, f3_fake, ...]

Feature matching loss = how different are these features?
  |f1_fake - f1_real| + |f2_fake - f2_real| + |f3_fake - f3_real| + ...
```

**In plain English:** "Do the internal features of fake audio look like the
internal features of real audio?"

**Why it helps:** The discriminator's intermediate layers capture "what makes
audio sound real" at different levels of abstraction. Matching these features
guides the generator toward producing realistic waveforms — even before the
discriminator's final output score says "this is real."

**Analogy:** Instead of just saying "this painting is fake," the art expert
says "the brush strokes look wrong, the color mixing is off, the shading is
unnatural." Each specific feedback helps the forger improve.

---

### Loss 3: Adversarial Loss (λ_adv = 1) — THE POLISH

**File:** `losses.py` — function `generator_loss()`

```python
# Adversarial (generator wants discriminator to say "real")
loss_adv = 0.0
count = 0
for scores in fake_scores:
    loss_adv += -scores.mean()  # negative because generator wants HIGH scores
    count += 1
loss_adv = loss_adv / max(count, 1)
```

**What it does:**

```
Discriminator scores fake audio:
  score = 10 → "this looks REAL"
  score = -10 → "this looks FAKE"

Generator wants: score to be HIGH (looks real)
So generator loss = -score  (minimize negative = maximize score)

If discriminator says "fake" (score=-5):
  generator loss = -(-5) = 5  → HIGH loss → generator will update

If discriminator says "real" (score=5):
  generator loss = -(5) = -5  → LOW loss → generator is doing well
```

**In plain English:** "Can the discriminator tell this is fake? If yes,
generate different audio."

**Why the weight is 1 (VERY low):** The adversarial loss is powerful and can
destabilize training if too strong. It's the final polish, not the main objective.

---

### How the Three Losses Work Together

```
Generator receives 3 signals each update:

  1. Mel loss (weight 45):  "Your audio must match THIS mel spectrogram!"
     → Controls CONTENT (dog vs cat vs bird)

  2. Feature matching (weight 2):  "Your audio features must look natural!"
     → Controls REALISM (does it sound like real audio?)

  3. Adversarial (weight 1):  "Can the discriminator tell it's fake?"
     → Controls QUALITY (crisp vs grainy)

Without mel loss:    generator makes "realistic" audio of WRONG content
Without FM loss:     generator matches mel but sounds artificial
Without adversarial: generator matches mel but lacks fine details
```

---

## Training Loop: Generator vs. Discriminator Dance

**File:** `train.py` — function `train_epoch()`

### Step-by-Step for Each Batch

```
Batch: real_audio [B, 1, 8192]

Step A: Generate fake audio
  real_mel = compute_mel(real_audio)
  fake_audio = generator(real_mel)

Step B: Train Discriminator
  real_score = discriminator(real_audio)    # should be HIGH
  fake_score = discriminator(fake.detach()) # should be LOW (detach = don't update generator)
  d_loss = hinge_loss(real_score, fake_score)
  d_loss.backward()
  optimizer_d.step()

Step C: Train Generator
  fake_score = discriminator(fake_audio)    # NOW update generator through discriminator
  mel_loss = MelL1Loss(fake_audio, real_audio)
  fm_loss = feature_matching(real_features, fake_features)
  adv_loss = adversarial(fake_score)
  g_loss = 45×mel_loss + 2×fm_loss + 1×adv_loss
  g_loss.backward()
  optimizer_g.step()
```

**Key insight:** Step B and Step C are **alternating updates** on the same batch.
First the discriminator learns to catch fakes, then the generator learns to
fool the updated discriminator. They're literally playing a game — each one's
update makes the other one's job harder.

**Code reference:** `train.py` line 293-340
```python
# ── Discriminator ──
opt_d.zero_grad()
r_score, r_feat = discriminator(d_real)
f_score_d, _ = discriminator(d_fake)
d_loss, d_dict = discriminator_loss(r_score, f_score_d)
d_loss.backward()
opt_d.step()

# ── Generator ──
opt_g.zero_grad()
f_score_g, f_feat_g = discriminator(fake)
g_loss, g_dict = generator_loss(
    fake, real_trim, f_score_g, f_feat_g, r_feat, mel_loss_fn,
    lambda_mel=cfg.lambda_mel,  # 45
    lambda_fm=cfg.lambda_fm,     # 2
    lambda_adv=cfg.lambda_adv,   # 1
)
g_loss.backward()
opt_g.step()
```

---

## Training Strategy: Meltrain → Full GAN

### Phase 1: Mel-Only Pretraining (`mode="meltrain"`)

```python
# train.py — MODE == "meltrain"
# Generator trains WITHOUT discriminator
# Loss = mel_loss + time_loss
```

**What happens:**
- Generator learns "mel → audio" mapping from scratch
- No adversarial instability in early training
- Converges to a reasonable baseline in 15-30 epochs

**What to watch:**
```
Epoch 1: mel=6.6  ← generator starts blind
Epoch 2: mel=2.5  ← huge improvement (first gradient updates are massive)
Epoch 5: mel=1.8  ← steady learning
Epoch 15: mel=1.0 ← plateau approaching
```

**If mel=0.0000 from epoch 1:** Data loading bug — audio files not loading.
Check data paths and file formats.

### Phase 2: Full GAN Training (`mode="train"`)

```python
# train.py — MODE == "train"
# Generator + discriminator train together
# Loss = 45×mel_loss + 2×fm_loss + 1×adv_loss
```

**What happens:**
- Generator already knows mel → audio from Phase 1
- Discriminator refines waveform quality
- Feature matching adds naturalness
- ~20-30 epochs for convergence

**What to watch:**
```
Healthy training:
  G mel loss: drops steadily (6→3→2→1.5→1.0...)
  D loss: hovers around 1-2 (hinge loss equilibrium)
  Val mel: similar to train mel (no overfitting)

Unhealthy training:
  D loss → 0:       Discriminator too strong → generator can't learn
  D loss → very high: Discriminator collapsed → adversarial loss meaningless
  G mel loss stops:  Generator plateaued → need more epochs or LR warmup
  D loss oscillates wildly: Learning rate too high
```

---

## Critical Hyperparameters (Learned the Hard Way)

| Parameter | Value | What it controls | What happens if wrong |
|-----------|-------|-----------------|----------------------|
| `segment_size` | **16384** (0.74s) | How much audio context generator sees per sample | 8192 (0.37s) → too short for full bark, generator sees incomplete patterns |
| `lambda_mel` | **45** | How hard generator must match input mel | 1.0 → generator drifts to speech/wrong content |
| `lambda_fm` | **2** | How much feature matching guides realism | 10 → overwhelms mel loss, similar drift |
| `lambda_adv` | **1** | How much adversarial loss polishes | 2+ → discriminator dominates, generator ignores mel |
| `learning_rate` | **2e-4** | How big each training step is | Higher → unstable GAN; lower → painfully slow |
| `batch_size` | **8** | How many samples per update | < 4 → discriminator memorizes, mode collapse |
| `beta1` (Adam) | **0.8** | Momentum for optimizer | 0.9 (default) → discriminator overshoots |
| `num_workers` | **0** | Multiprocessing for data loading | > 0 → silent data bugs on CUDA (relative paths break) |

### The Mel Loss Lesson (Most Important)

```
WRONG: lambda_mel=1.0, lambda_adv=2.0
→ Generator: "I'll make realistic audio" (ignores input mel)
→ Result: Human speech instead of dog barks ❌

CORRECT: lambda_mel=45.0, lambda_adv=1.0
→ Generator: "I must match THIS mel spectrogram"
→ Discriminator: "But make it sound natural"
→ Result: Correct content + natural waveform ✅
```

**Rule of thumb:** Mel loss should be the HEAVIEST weight (45×).
The discriminator only polishes what the mel loss constrains.

---

## Why This is Industry Standard

HiFi-GAN was published by Kong, Kim, and Bae (2020) at NeurIPS. It's used in:

| Company/Product | Uses HiFi-GAN for |
|----------------|-------------------|
| VITS (TTS system) | Text → speech vocoder |
| YourTTS (Coqui) | Voice cloning vocoder |
| Matcha-TTS | Text → speech vocoder |
| Bark (Suno AI) | Text → speech vocoder (similar architecture) |
| AudioCraft (Meta) | Music generation vocoder |

The MRF architecture and multi-discriminator design became the **standard**
because it produces high-quality audio with fast inference (real-time on GPU).

---

## Key Lessons from Our Implementation

### 1. Data loading bugs are the #1 cause of "loss = 0"

If training shows `mel=0.0000` from epoch 1, audio files aren't loading.
The generator outputs zeros, real audio is zeros, loss = |0-0| = 0.

**Fix:** Always validate data before training:
```python
# train.py — added data validation
_check = next(iter(train_loader))
_check_max = _check.abs().max().item()
if _check_max < 1e-6:
    print("🚨 ALL ZEROS — audio files not loading correctly.")
    return None  # stop training immediately
```

### 2. torchaudio 2.11+ requires FFmpeg shared libraries

On systems without FFmpeg, `torchaudio.load()` fails with:
`Could not load libtorchcodec`

**Fix:** Fall back to `soundfile` (reads WAV natively, no FFmpeg needed):
```python
# train.py — _load_audio()
def _load_audio(path):
    try:
        return torchaudio.load(path)  # fast, native
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(data)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav, sr
    except Exception as e:
        warnings.warn(f"⚠️ Audio load FAILED: {path} → {e}")
        return None, None
```

### 3. Relative paths break in multiprocessing workers

With `num_workers > 0`, spawned worker processes may have a different working
directory. `data/animal_audio/Dog/file.wav` becomes an invalid path.

**Fix:** Use absolute paths + `num_workers=0`:
```python
# dataset stores absolute paths
self.files.append(os.path.abspath(os.path.join(cls_dir, fname)))

# config sets workers to 0
"num_workers": 0,
```

### 4. The discriminator easily overpowers the generator

With strong adversarial loss (λ_adv=2), the generator produces "realistic"
but **wrong** content. Human speech sounds "real" but isn't a dog bark.

**Fix:** Mel loss must dominate:
```
λ_mel (45) >> λ_fm (2) > λ_adv (1)
  ↑                  ↑           ↑
PRIMARY            GUIDE       POLISH
```

### 5. Segment size must match the signal

```
8192 samples (0.37s) → too short, misses full bark pattern
16384 samples (0.74s) → captures most animal sounds
44100 samples (2.00s) → ideal but slower training
```

---

## HiFi-GAN vs. Your VAE: Different Problems, Same Goal

```
VAE (Phase 4):
  Input:  class label (e.g., "dog")
  Output: mel spectrogram [1, 64, T]
  Goal:   "What should a dog sound look like?"
  Quality: Blurred (averages over possibilities)
  Conversion to audio: Griffin-Lim (grainy)

HiFi-GAN (Phase 7a):
  Input:  mel spectrogram [1, 64, T]
  Output: waveform [1, 1, T×200]
  Goal:   "How do I convert THIS mel into audio?"
  Quality: Crisp (learned phase + waveform)
  Conversion to audio: Neural network (natural)

Together:
  "dog" → VAE → [mel spectrogram] → HiFi-GAN → [crisp audio ✨]
```

**VAE and HiFi-GAN are complementary.** The VAE generates the spectrogram
(the **what**). HiFi-GAN converts it to audio (the **how**).

---

## Next Steps After HiFi-GAN

| Phase | What | Builds on HiFi-GAN |
|-------|------|-------------------|
| **7b: Diffusion** | Sharpen the VAE's blurry mel output | HiFi-GAN still needed as vocoder |
| **7c: Sequential** | Generate long sounds, chain animals | HiFi-GAN handles any length via chunking |
| **7d: Latent mixing** | Blend animals in z-space | HiFi-GAN converts any mel, even mixed ones |
| **7e: UI v2** | Add vocoder selector to web app | Users can hear Griffin-Lim vs HiFi-GAN |

The full quality pipeline:
```
"dog" → VAE → [blurry mel] → Diffusion → [sharp mel] → HiFi-GAN → [crisp audio ✨]
```

---

## Appendix: Layer-by-Layer Architecture Diagrams

These diagrams trace exact tensor shapes (batch=1, T=41 frames ≈ 0.37s) through your actual code.

### Generator — mel → waveform

```
[1, 64, 41]          ← mel spectrogram (64 freq bins × 41 time frames)
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  PRE-CONV                                           │
│  Conv1d(64 → 256, kernel=7, padding=3)              │
│  No activation                                      │
└─────────────────────────────────────────────────────┘
     │
     ▼
[1, 256, 41]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MRF BLOCK 1  — upsample ×5, channels: 256→128                                  │
│                                                                                  │
│   [1, 256, 41]                                                                   │
│        │                                                                         │
│        ▼                                                                         │
│   ┌──────────────────────────┐                                                   │
│   │ ConvTranspose1d          │  stride=5, kernel=10, pad=2                       │
│   │ 256 → 128                │  Time stretches: 41 × 5 = 205                     │
│   └──────────────────────────┘                                                   │
│        │                                                                         │
│        ▼                                                                         │
│   [1, 128, 205]                                                                  │
│        │                                                                         │
│        ├──┬──────────────────────────────────────────────────┬──┐                │
│        │  │                                                  │  │                │
│        │  ▼                                                  ▼  ▼                │
│        │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │                │
│        │ │ ResBlock    │  │ ResBlock    │  │ ResBlock    │  │  │                │
│        │ │ kernel=3    │  │ kernel=7    │  │ kernel=11   │  │  │                │
│        │ │ dil=1,3,5   │  │ dil=1,3,5   │  │ dil=1,3,5   │  │  │                │
│        │ │             │  │             │  │             │  │  │                │
│        │ │ catch FAST  │  │ catch MID   │  │ catch SLOW  │  │  │                │
│        │ │ transients  │  │ timbre      │  │ pitch       │  │  │                │
│        │ │ [128,205]   │  │ [128,205]   │  │ [128,205]   │  │  │                │
│        │ └─────────────┘  └─────────────┘  └─────────────┘  │  │                │
│        │         │                │                │         │  │                │
│        │         └────────────────┴────────────────┘         │  │                │
│        │                      SUM (add together)             │  │                │
│        │                           │                         │  │                │
│        └───────────────────────────┘                         │  │                │
│                                    ▼                         │  │                │
│                             [1, 128, 205]                    │  │                │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[1, 128, 205]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MRF BLOCK 2  — upsample ×5, channels: 128→64                                   │
│  ConvTranspose1d(128→64, stride=5, kernel=10)                                   │
│  3×ResBlock(3,7,11) → SUM                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[1, 64, 1025]        ← 205 × 5 = 1025
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MRF BLOCK 3  — upsample ×4, channels: 64→32                                    │
│  ConvTranspose1d(64→32, stride=4, kernel=8)                                     │
│  3×ResBlock(3,7,11) → SUM                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[1, 32, 4100]        ← 1025 × 4 = 4100
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MRF BLOCK 4  — upsample ×2, channels: 32→16                                    │
│  ConvTranspose1d(32→16, stride=2, kernel=4)                                     │
│  3×ResBlock(3,7,11) → SUM                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[1, 16, 8200]        ← 4100 × 2 = 8200  (also 41 × 200)
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  POST-CONV                                          │
│  Conv1d(16 → 1, kernel=7, padding=3)                │
│  No activation (learned amplitude)                  │
└─────────────────────────────────────────────────────┘
     │
     ▼
[1, 1, 8200]         ← AUDIO WAVEFORM 🎵
```

**Shape evolution table:**

| Stage | Operation | Shape In | Shape Out | Time dim |
|-------|-----------|----------|-----------|----------|
| Pre | Conv1d | `[1, 64, 41]` | `[1, 256, 41]` | 41 |
| MRF1 | ConvTranspose + 3×ResBlock | `[1, 256, 41]` | `[1, 128, 205]` | 41×5 |
| MRF2 | ConvTranspose + 3×ResBlock | `[1, 128, 205]` | `[1, 64, 1025]` | 205×5 |
| MRF3 | ConvTranspose + 3×ResBlock | `[1, 64, 1025]` | `[1, 32, 4100]` | 1025×4 |
| MRF4 | ConvTranspose + 3×ResBlock | `[1, 32, 4100]` | `[1, 16, 8200]` | 4100×2 |
| Post | Conv1d | `[1, 16, 8200]` | `[1, 1, 8200]` | 8200 |

---

### Zoom: Inside One ResBlock

```
Input x  ───────────────────────────┐
  [128, 205]                        │
                                    │   ← residual / skip connection
                                    │
     x → LeakyReLU(0.1) ────────────┤
                │                   │
                ▼                   │
        Conv1d(128→128, k=3,       │
                dilation=1,         │
                padding=1)          │
                │                   │
                ▼                   │
           LeakyReLU(0.1) ──────────┤
                │                   │
                ▼                   │
        Conv1d(128→128, k=3,       │
                dilation=3,         │
                padding=3)          │
                │                   │
                ▼                   │
           LeakyReLU(0.1) ──────────┤
                │                   │
                ▼                   │
        Conv1d(128→128, k=3,       │
                dilation=5,         │
                padding=5)          │
                │                   │
                ▼                   │
              output                │
                │                   │
                └──────────(+)──────┘   ← x + output (element-wise)
                            │
                            ▼
                         new x
                         [128, 205]
```

**Each ResBlock repeats this 3 times** (for dilations 1, 3, 5). The residual connection means if the convolutions output garbage, the original signal is still preserved.

---

### Discriminator — "Is this audio real?"

Your code uses **MPD only** (5 period discriminators). Here's how ONE works:

#### Period Discriminator (period = 5)

```
[1, 1, 8200]         ← waveform
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  FOLD into 2D grid at period=5                                              │
│                                                                             │
│  Pad to multiple of 5: 8200 already divisible                               │
│                                                                             │
│  Reshape: [1, 1, 8200] → [1, 1, 5, 1640]    (5 rows, 1640 columns)        │
│  Permute: [1, 5, 1, 1640]                                                   │
│  View:    [5, 1, 1, 1640]    ← batch now 5!                               │
│                                                                             │
│  Visual: samples [1,2,3,4,5,6,7,8...] folded into grid:                   │
│     row 0:  1, 6, 11, 16...   ← every 5th, offset 0                        │
│     row 1:  2, 7, 12, 17...   ← every 5th, offset 1                        │
│     row 2:  3, 8, 13, 18...                                                │
│     row 3:  4, 9, 14, 19...                                                │
│     row 4:  5, 10, 15, 20...                                               │
│                                                                             │
│  A 4.4kHz tone (period ≈ 5 samples) becomes a VERTICAL stripe              │
│  that 2D convolution detects instantly!                                     │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[5, 1, 1, 1640]      ← B=5, C=1, H=1, W=1640
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1                                                                    │
│  Conv2d(1 → 16, kernel=(5,5), stride=(3,3), padding=(2,2))                  │
│  LeakyReLU(0.1)                                                             │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[5, 16, 1, 547]      ← H stays 1, W ≈ 1640/3
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2                                                                    │
│  Conv2d(16 → 64, kernel=(5,5), stride=(3,3), padding=(2,2))                 │
│  LeakyReLU(0.1)                                                             │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[5, 64, 1, 183]
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3                                                                    │
│  Conv2d(64 → 128, kernel=(5,5), stride=(3,3), padding=(2,2))                │
│  LeakyReLU(0.1)                                                             │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[5, 128, 1, 61]
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  POST-CONV                                                                  │
│  Conv2d(128 → 1, kernel=(3,1), padding=(1,0))                               │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[5, 1, 1, 61]
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  UNFOLD back: [5, 1, 1, 61] → [1, 5, 61]                                   │
│                                                                             │
│  Score: [1, 5, 61]  ← each value = "how real is this chunk?"               │
│  (higher = discriminator thinks it's real)                                  │
│                                                                             │
│  Features saved at every layer for feature-matching loss!                   │
└────────────────────────────────────────────────────────────────────────────┘
```

#### Full Multi-Period Discriminator (all 5 periods)

```
                         [1, 1, 8200]  waveform
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
  ┌──────────┐          ┌──────────┐          ┌──────────┐
  │ Period=2 │          │ Period=3 │          │ Period=5 │
  │(11kHz)   │          │(7.35kHz) │          │(4.4kHz)  │
  │ Conv2d×3 │          │ Conv2d×3 │          │ Conv2d×3 │
  │ score[]  │          │ score[]  │          │ score[]  │
  │ feats[]  │          │ feats[]  │          │ feats[]  │
  └────┬─────┘          └────┬─────┘          └────┬─────┘
       │                     │                     │
       ▼                     ▼                     ▼
  ┌──────────┐          ┌──────────┐
  │ Period=7 │          │ Period=11│
  │(3.15kHz) │          │(2.0kHz)  │
  │ Conv2d×3 │          │ Conv2d×3 │
  │ score[]  │          │ score[]  │
  │ feats[]  │          │ feats[]  │
  └────┬─────┘          └────┬─────┘
       │                     │
       └─────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
  all_scores (5)     all_features (5 groups)
       │                 │
       ▼                 ▼
   adversarial loss    feature matching loss
   (G wants HIGH)      (match real vs fake layers)
```

---

### Training Flow — One Batch Step

```
BATCH: real_audio = [8, 1, 16384]   ← 8 clips, 0.74s each

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP A: Generate fake audio                                                │
└─────────────────────────────────────────────────────────────────────────────┘

  real_audio [8,1,16384]
       │
       ▼
  compute_mel()
       │
       ▼
  real_mel [8, 64, 82]    ← 16384 / 200 = ~82 frames
       │
       ▼
  ┌─────────────┐
  │  GENERATOR  │
  │  (mel →     │
  │   waveform) │
  └─────────────┘
       │
       ▼
  fake_audio [8, 1, 16384]


┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP B: Train DISCRIMINATOR (freeze generator)                             │
│  Goal: D(real) = HIGH, D(fake) = LOW                                        │
└─────────────────────────────────────────────────────────────────────────────┘

  real_audio ─────────┐
                      ▼
              ┌───────────────┐
              │     D_MPD     │
              │  (5 periods)  │
              └───────┬───────┘
                      │
              real_scores (HIGH ✓)
              real_features (saved)

  fake_audio ─────────┐
                      ▼
              ┌───────────────┐
              │     D_MPD     │
              │  (detached —  │
              │   no gradient │
              │   to G)       │
              └───────┬───────┘
                      │
              fake_scores (LOW ✓)

  d_loss = hinge_loss(real_scores, fake_scores)
  d_loss.backward()
  optimizer_d.step()


┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP C: Train GENERATOR (discriminator now provides gradients)             │
│  Goal: fool D + match mel + match features                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  fake_audio ─────────┐
                      ▼
              ┌───────────────┐
              │     D_MPD     │
              │  (NOT detached│
              │   — gradient  │
              │   flows to G) │
              └───────┬───────┘
                      │
              fake_scores (G wants HIGH)
              fake_features

  ┌─────────────────────────────────────────────────────────────────────────┐
│  COMPUTE 3 LOSSES                                                        │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  L_mel      │    │  L_fm       │    │  L_adv      │                  │
│  │  (weight 45)│    │  (weight 2) │    │  (weight 1) │                  │
│  │             │    │             │    │             │                  │
│  │ mel(fake)   │    │ compare     │    │ -mean(fake  │                  │
│  │ vs          │    │ D's internal│    │ _scores)    │                  │
│  │ mel(real)   │    │ layers      │    │             │                  │
│  │ L1 loss     │    │ L1 loss     │    │             │                  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                  │
│         │                  │                  │                          │
│         └──────────────────┼──────────────────┘                          │
│                            ▼                                             │
│         g_loss = 45×L_mel + 2×L_fm + 1×L_adv                             │
│                            │                                             │
│                            ▼                                             │
│                    g_loss.backward()                                     │
│                    optimizer_g.step()                                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  What each loss does:                                               │  │
│  │                                                                     │  │
│  │  L_mel (45×):  "Match THIS spectrogram!" ← CONTENT (dog, not cat) │  │
│  │  L_fm  (2×):   "Look natural inside D's brain" ← REALISM          │  │
│  │  L_adv (1×):   "Fool D into saying real" ← POLISH                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### The Alternating Dance

```
Epoch 1, Batch 0:
  G: terrible fake  ──▶  D: easily catches it  ──▶  G improves slightly
Epoch 1, Batch 50:
  G: okay fake      ──▶  D: still catches it    ──▶  G improves more
Epoch 5, Batch 0:
  G: good fake      ──▶  D: struggles           ──▶  D improves
Epoch 10, Batch 0:
  G: great fake     ──▶  D: barely catches it   ──▶  both improve
...
Epoch 30:
  G: perfect fake   ──▶  D: coin flip (50/50)   ──▶  Nash equilibrium ✓
```

At equilibrium, the discriminator outputs scores near zero for both real and fake — it genuinely can't tell the difference. That's when your generator produces audio indistinguishable from real recordings.

---

## References

- **Kong, Kim, Bae (2020):** "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" — [Paper](https://arxiv.org/abs/2010.05646)
- **Original repo:** [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- **Kazuki's explanation:** [Understanding HiFi-GAN](https://kazemnejad.com/blog/2024-03-19-hifi-gan/)
- **GAN basics:** [Ian Goodfellow's original paper (2014)](https://arxiv.org/abs/1406.2661)
- **Feature matching:** [Salimans et al. (2016)](https://arxiv.org/abs/1606.03498)
