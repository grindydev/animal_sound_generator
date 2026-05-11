# Phase 7a — HiFi-GAN Neural Vocoder

> **One-line summary:** HiFi-GAN converts your VAE's mel spectrograms into crisp audio. It does NOT replace the VAE. It is a separate model trained on raw audio.

---

## 1. The Problem HiFi-GAN Solves

Your VAE generates a **mel spectrogram** — a 2D grid of frequency × time. To hear it, you must convert back to a **waveform** — a 1D list of air-pressure numbers.

| Method | How it works | Sound quality |
|--------|-------------|---------------|
| **Griffin-Lim** | Math formula guesses missing phase info | Robotic, grainy, metallic |
| **HiFi-GAN** | Neural network learns what real waveforms look like | Crisp, natural |

```
"dog" label
    ↓
┌──────────┐
│   VAE    │  ← YOU ALREADY BUILT THIS
└────┬─────┘
     ↓
[64, 82] mel spectrogram     ← blurry "photo" of sound
     ↓
┌──────────┐                 ← HI-FI-GAN (this phase)
│ HiFi-GAN │                   replaces Griffin-Lim here
│Generator │
└────┬─────┘
     ↓
[1, 16400] waveform          ← audio you can actually hear
```

**HiFi-GAN does NOT generate animal sounds.** The VAE does that. HiFi-GAN only converts mel → waveform. It is a **vocoder** (voice coder), not a generator.

---

## 2. The Two Models

HiFi-GAN has two separate neural networks. They train together but have different jobs.

### Generator (`src/hifigan/generator.py`)

**Input:** mel spectrogram `[batch, 64, time_frames]`  
**Output:** audio waveform `[batch, 1, time_frames × 200]`  
**Job:** "Turn this frequency grid into realistic audio."

### Discriminator (`src/hifigan/discriminator.py`)

**Input:** audio waveform `[batch, 1, samples]`  
**Output:** scores `[batch, period, positions]` + internal features  
**Job:** "Is this audio real or fake? Give me a number."

```
                    ┌─────────────────┐
                    │  DISCRIMINATOR  │
                    │     (Judge)     │
                    │                 │
  Real audio ──────▶│  score = +5     │───▶ "REAL!"
                    │                 │
  Fake audio ──────▶│  score = -5     │───▶ "FAKE!"
                    │                 │
                    └─────────────────┘
                           ▲
                           │
    ┌──────────────────────┘
    │
    │   Generator wants HIGH scores (positive = real)
    │   Generator's loss = -score
    │   Lower loss = higher score = better fake
    │
    ▼
┌─────────────────┐
│   GENERATOR     │
│    (Forger)     │
│                 │
│  mel ──────▶    │───▶ fake audio ──────▶ D judges it
│  [64,82]        │     [1,16400]
│                 │
└─────────────────┘
```

---

## 3. Generator Architecture — Mel → Waveform

**File:** `src/hifigan/generator.py`  
**Class:** `HiFiGANGenerator` (line ~160)

The Generator is a 1D upsampling network. It stretches the time axis by 200× (equal to `hop_length`).

```
Input:  [B, 64, 82]     ← mel spectrogram (64 freq bins × 82 time frames)
           │
           ▼
    ┌─────────────────────────────┐
    │ PRE_CONV                    │  Conv1d(64→256, kernel=7)
    │ Code: generator.py line 175 │
    └─────────────────────────────┘
           │
           ▼  [B, 256, 82]
           │
    ┌─────────────────────────────┐
    │ MRF BLOCK 1                 │  ConvTranspose(stride=5)  256→128
    │ Upsample ×5                 │  + 3 ResBlocks(ks=3,7,11) summed
    │ Code: generator.py line 183 │  Time: 82 → 410
    └─────────────────────────────┘
           │
           ▼  [B, 128, 410]
           │
    ┌─────────────────────────────┐
    │ MRF BLOCK 2                 │  ConvTranspose(stride=5)  128→64
    │ Upsample ×5                 │  + 3 ResBlocks summed
    │                             │  Time: 410 → 2050
    └─────────────────────────────┘
           │
           ▼  [B, 64, 2050]
           │
    ┌─────────────────────────────┐
    │ MRF BLOCK 3                 │  ConvTranspose(stride=4)  64→32
    │ Upsample ×4                 │  + 3 ResBlocks summed
    │                             │  Time: 2050 → 8200
    └─────────────────────────────┘
           │
           ▼  [B, 32, 8200]
           │
    ┌─────────────────────────────┐
    │ MRF BLOCK 4                 │  ConvTranspose(stride=2)  32→16
    │ Upsample ×2                 │  + 3 ResBlocks summed
    │                             │  Time: 8200 → 16400
    └─────────────────────────────┘
           │
           ▼  [B, 16, 16400]
           │
    ┌─────────────────────────────┐
    │ POST_CONV                   │  Conv1d(16→1, kernel=7)
    │ Code: generator.py line 188 │  No activation
    └─────────────────────────────┘
           │
           ▼
Output: [B, 1, 16400]    ← audio waveform
```

**Total upsample:** `5 × 5 × 4 × 2 = 200 = hop_length` ✅

### Inside a ResBlock (`generator.py` line ~30)

```python
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        self.convs = nn.ModuleList()
        for d in dilations:          # d = 1, 3, 5
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size,
                          dilation=d, padding=get_padding(kernel_size, d)),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size,
                          dilation=1, padding=get_padding(kernel_size, 1)),
            ))
```

Each ResBlock has **3 dilated conv paths** (dilation = 1, 3, 5). Dilation lets a small kernel "see" farther without more weights:

```
Dilation=1:  ●●●        ← sees 3 adjacent samples
Dilation=3:  ●  ●  ●    ← sees samples 3 apart (wider view)
Dilation=5:  ●    ●    ●  ← sees samples 5 apart (widest view)
```

A residual connection (`x = x + residual`) lets the original signal skip through. If the convolutions mess up, the signal still gets through.

### Inside an MRFBlock (`generator.py` line ~65)

```python
class MRFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ...):
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels,
                                           stride=upsample_rate)  # stretches time
        self.resblocks = nn.ModuleList()
        for ks in (3, 7, 11):       # 3 parallel paths
            self.resblocks.append(ResBlock(out_channels, ks, (1,3,5)))

    def forward(self, x):
        x = self.upsample(x)
        outputs = [rb(x) for rb in self.resblocks]
        return sum(outputs)         # sum all 3 paths
```

Three kernel sizes run in parallel:
- **kernel=3:** catches fast transients (attack of a bark)
- **kernel=7:** catches medium patterns (vocal timbre)
- **kernel=11:** catches slow modulation (pitch contour)

Their outputs are **summed** — the Generator gets all time scales at once.

---

## 4. Discriminator Architecture — Waveform → Score

**File:** `src/hifigan/discriminator.py`  
**Class:** `PeriodDiscriminator` (line ~20)

The Discriminator does NOT look at mel spectrograms. It looks at **raw audio waveforms** and checks if they look natural.

### The Folding Trick (`discriminator.py` line ~81)

```python
def forward(self, x):
    B, C, L = x.shape           # x = [8, 1, 16400]

    # Pad to multiple of period
    if L % self.period != 0:
        n_pad = self.period - (L % self.period)
        x = F.pad(x, (0, n_pad), mode="reflect")

    # FOLD: [8, 1, 16400] → [8, 1, 5, 3280]
    x = x.view(B, C, self.period, L // self.period)
    # PERMUTE: [8, 5, 1, 3280]
    x = x.permute(0, 2, 1, 3).contiguous()
    # MERGE batch: [40, 1, 1, 3280]
    x = x.view(B * self.period, 1, L // self.period).unsqueeze(2)
```

**What folding does (period=5 example):**

```
Original audio: [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, ...]

Folded into 5 rows:
  Row 0: s1,  s6,  s11, s16, ...   (every 5th, offset 0)
  Row 1: s2,  s7,  s12, s17, ...   (every 5th, offset 1)
  Row 2: s3,  s8,  s13, s18, ...   (every 5th, offset 2)
  Row 3: s4,  s9,  s14, s19, ...   (every 5th, offset 3)
  Row 4: s5,  s10, s15, s20, ...   (every 5th, offset 4)
```

Now the waveform is a **grayscale image** `[40, 1, 1, 3280]`. A 4.4kHz tone (period ≈ 5 samples) becomes a vertical stripe that 2D convolutions detect instantly.

### The Conv2D Scan (`discriminator.py` line ~55-62)

```python
channels = [1, 16, 64, 128]
self.convs = nn.ModuleList()
for i in range(3):          # 3 Conv2D layers
    self.convs.append(
        nn.Sequential(
            nn.Conv2d(channels[i], channels[i+1],
                      kernel_size=(5, 5), stride=(3, 3),
                      padding=(2, get_padding(5))),
            nn.LeakyReLU(0.1),
        )
    )
```

```
After folding: [40, 1, 1, 3280]
       │
       ▼
┌────────────────────────────┐
│ Conv2d(1→16, 5×5, stride=3)│  ← scans for raw glitches
│ Output: [40, 16, 1, 1093]  │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│ Conv2d(16→64, 5×5, stride=3)│  ← scans for timbre artifacts
│ Output: [40, 64, 1, 364]   │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│ Conv2d(64→128, 5×5, stride=3)│ ← scans for overall structure
│ Output: [40, 128, 1, 121]  │
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│ Conv2d(128→1, 3×1)         │  ← 1 score per position
│ Output: [40, 1, 1, 121]    │
└────────────────────────────┘
       │
       ▼
Unfold: [8, 5, 121]          ← scores for this period
```

**5 period discriminators run in parallel** (periods = 2, 3, 5, 7, 11). Each catches artifacts at different frequencies.

---

## 5. The Three Losses

**File:** `src/hifigan/losses.py`

The Generator uses **three losses combined**. The Discriminator uses a separate hinge loss.

### Loss 1: Mel L1 Loss (λ = 45)

**Code:** `losses.py` line ~20, class `MelL1Loss`

```python
class MelL1Loss(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, ...):
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels, power=1)

    def forward(self, fake_audio, real_audio):
        fake_mel = self.mel_transform(fake_audio.squeeze(1))   # [B, 64, T]
        real_mel = self.mel_transform(real_audio.squeeze(1))   # [B, 64, T]
        return F.l1_loss(fake_mel, real_mel)                   # mean(|a-b|)
```

**What it does:** Compares the mel spectrogram of generated audio vs real audio.  
**Why weight = 45:** This is the PRIMARY objective. Without it, the Generator ignores the input mel and produces any "realistic" audio (e.g., human speech instead of dog barks).

### Loss 2: Feature Matching Loss (λ = 2)

**Code:** `losses.py` line ~36

```python
def feature_matching_loss(real_features, fake_features):
    loss = 0.0
    count = 0
    for real_group, fake_group in zip(real_features, fake_features):
        for r, f in zip(real_group, fake_group):
            loss += F.l1_loss(f, r.detach())    # |fake_feat - real_feat|
            count += 1
    return loss / max(count, 1)
```

**What it does:** Compares the Discriminator's internal layer outputs for real vs fake audio.  
**Why it helps:** The Discriminator's layers extract "what makes audio sound real" at different levels. Matching these guides the Generator toward natural waveforms.

**Example:** If Layer 1 of D outputs `[0.5, 0.3, 0.8]` for real and `[0.1, 0.9, 0.2]` for fake, the Generator learns to make its Layer 1 output closer to `[0.5, 0.3, 0.8]`.

### Loss 3: Adversarial Loss (λ = 1)

**Code:** `losses.py` line ~65, inside `generator_loss()`

```python
def generator_loss(fake_audio, real_audio, fake_scores, fake_features,
                   real_features, mel_loss_fn, lambda_mel, lambda_fm, lambda_adv):

    loss_mel = mel_loss_fn(fake_audio, real_audio)
    loss_fm  = feature_matching_loss(real_features, fake_features)

    # ── ADVERSARIAL LOSS ──
    loss_adv = 0.0
    count = 0
    for scores in fake_scores:          # 5 score tensors (one per period)
        loss_adv += -scores.mean()      # NEGATIVE of average D score
        count += 1
    loss_adv = loss_adv / max(count, 1)

    total = lambda_mel * loss_mel + lambda_fm * loss_fm + lambda_adv * loss_adv
    return total, {"g_mel": loss_mel.item(), "g_fm": loss_fm.item(),
                   "g_adv": loss_adv.item(), "g_total": total.item()}
```

**What it does:** Measures how positive the Discriminator's scores are for fake audio.  
**Why `-score`:** The Generator minimizes loss. Minimizing `-score` = maximizing `score`. If D says `+5` (real), loss = `-5` (good). If D says `-5` (fake), loss = `+5` (bad).

**Why weight = 1:** Just a gentle polish. Too strong and the Generator ignores the mel to chase D's approval.

### Discriminator Hinge Loss

**Code:** `losses.py` line ~93

```python
def discriminator_loss(real_scores, fake_scores):
    loss_real = 0.0
    loss_fake = 0.0
    count = 0
    for r_scores, f_scores in zip(real_scores, fake_scores):
        loss_real += F.relu(1.0 - r_scores).mean()   # real must be ≥ 1
        loss_fake += F.relu(1.0 + f_scores).mean()   # fake must be ≤ -1
        count += 1
    return (loss_real + loss_fake) / count, {"d_real": ..., "d_fake": ..., "d_total": ...}
```

**What it does:** Forces a margin of 2 between real and fake scores.  
**Why hinge:** Prevents the Discriminator from outputting extreme scores forever, which would give the Generator no gradient to learn from.

---

## 6. Training — Two Phases

**File:** `src/hifigan/train.py`

HiFi-GAN trains in two phases. You run them sequentially.

### Phase I: Meltrain — Generator Only (30 epochs)

**Mode:** `CONFIG["mode"] = "meltrain"`  
**Function:** `train_epoch_mel_only()` (line ~200)  
**What happens:**

```python
for audio in train_loader:              # audio = [B, 1, 16384]
    real_mel = compute_mel(audio)       # [B, 64, 82]
    target_len = real_mel.shape[-1] * cfg.hop_length   # 82 × 200 = 16400
    real_trim = audio[..., :target_len] # [B, 1, 16400]

    fake = generator(real_mel, target_length=target_len)  # [B, 1, 16400]

    mel_loss = mel_loss_fn(fake, real_trim)    # |mel(fake) - mel(real)|
    time_loss = F.l1_loss(fake, real_trim)     # |fake - real| directly
    loss = mel_loss + 1.0 * time_loss

    opt_g.zero_grad()
    loss.backward()
    clip_grad_norm_(generator.parameters(), 1.0)
    opt_g.step()
```

**Why train Generator first?** GANs are unstable. If you throw both networks together from scratch, the Discriminator dominates and the Generator never learns. Meltrain gives the Generator a solid baseline first.

**What to watch:**
```
Epoch 1:  mel_loss = 6.5    ← random noise
Epoch 10: mel_loss = 1.8    ← rough shape
Epoch 30: mel_loss = 0.5    ← good baseline, ready for GAN
```

### Phase II: Full GAN — Both Together (60 epochs)

**Mode:** `CONFIG["mode"] = "train"`  
**Function:** `train_epoch()` (line ~293)  
**What happens per batch:**

```
STEP A: Generate fake
─────────────────────────────────────────────────────────────
fake = generator(real_mel, target_len)     # [B, 1, 16400]


STEP B: Train DISCRIMINATOR
─────────────────────────────────────────────────────────────
Code: train.py lines 298-330

opt_d.zero_grad()

d_real = real_trim + noise(0.01)
d_fake = fake.detach() + noise(0.01)       # .detach() = stop gradient to G

r_score, r_feat = discriminator(d_real)    # D scores real audio
f_score_d, _    = discriminator(d_fake)    # D scores fake audio

d_loss, d_dict = discriminator_loss(r_score, f_score_d)
# d_loss = max(0, 1 - r_score) + max(0, 1 + f_score_d)

d_loss.backward()
clip_grad_norm_(discriminator.parameters(), 1.0)
opt_d.step()                                # ONLY D updates


STEP C: Train GENERATOR
─────────────────────────────────────────────────────────────
Code: train.py lines 333-365

opt_g.zero_grad()

f_score_g, f_feat_g = discriminator(fake)   # NO .detach()!
# Gradients flow through D into G

g_loss, g_dict = generator_loss(
    fake, real_trim,
    f_score_g, f_feat_g, r_feat, mel_loss_fn,
    lambda_mel=45, lambda_fm=2, lambda_adv=1,
)
# g_loss = 45 × mel_loss + 2 × fm_loss + 1 × adv_loss

g_loss.backward()
clip_grad_norm_(generator.parameters(), 1.0)
opt_g.step()                                # ONLY G updates
```

**Why D first in each batch?** The Generator plays against the **updated** Discriminator. If G trained first, it would be playing against an outdated D — too easy, no learning.

**What to watch:**
```
Healthy:
  G mel_loss: 6 → 3 → 2 → 1.5 → 1.0  (drops steadily)
  D loss: hovers around 0.5-1.5        (challenged but not crushed)

Unhealthy:
  D loss → 0:    Discriminator too strong → generator can't learn
  D loss > 5:    Discriminator collapsed → restart or lower LR
  G mel_loss flat:  Generator plateaued → check data loading
```

---

## 7. How Generator and Discriminator Play Against Each Other

```
BATCH 1:
  G makes terrible fake → D easily spots it (score = -5)
  D trains → gets better
  G trains → wants score = +5, gets -5, big loss, improves slightly

BATCH 2:
  G makes slightly better fake → D still spots it (score = -3)
  D trains → gets better
  G trains → wants +5, gets -3, medium loss, improves more

BATCH 100:
  G makes very good fake → D confused (score = 0.1)
  D trains → barely improves
  G trains → almost there, loss ≈ 0

BATCH 1000:
  G makes audio indistinguishable from real → D gives up (score = 0)
  Neither improves much → equilibrium → training converges
```

The Discriminator is the **grading machine**. The Generator is the **student**. Each batch:
1. The grader learns to spot the student's latest tricks.
2. The student learns to beat the updated grader.

---

## 8. Critical Hyperparameters

| Parameter | Value | Controls | If wrong |
|-----------|-------|----------|----------|
| `lambda_mel` | **45** | How hard G must match input mel | 1.0 → G drifts to wrong content (speech instead of bark) |
| `lambda_fm` | **2** | How much D's internal features guide G | 10 → overwhelms mel loss, similar drift |
| `lambda_adv` | **1** | How much G chases D's approval | 5 → G ignores mel, invents any "real" audio |
| `segment_size` | **16384** | Audio context per sample | 8192 → too short, misses full bark pattern |
| `batch_size` (GAN) | **8** | Samples per update | < 4 → D memorizes, mode collapse |
| `learning_rate` | **2e-4** | Step size | Higher → unstable; lower → painfully slow |
| `adam_betas` | **(0.8, 0.99)** | Momentum | (0.9, 0.999) → D overshoots |
| `num_workers` | **0** | Data loading workers | > 0 → silent data bugs on CUDA |

**The most important lesson:** Mel loss MUST dominate. It is the only thing that keeps the Generator producing the RIGHT sound (dog bark, not human speech). The Discriminator only cares about "does it sound real?" — it has no idea what animal it should be.

---

## 9. Quick File Reference

| What you want | File | Function/Class | Line |
|---------------|------|----------------|------|
| Generator model | `generator.py` | `HiFiGANGenerator` | ~160 |
| Generator pre-conv | `generator.py` | `self.pre_conv` | ~175 |
| Generator post-conv | `generator.py` | `self.post_conv` | ~188 |
| ResBlock | `generator.py` | `ResBlock` | ~30 |
| MRFBlock | `generator.py` | `MRFBlock` | ~65 |
| Discriminator model | `discriminator.py` | `PeriodDiscriminator` | ~20 |
| Discriminator folding | `discriminator.py` | `forward()` | ~81-90 |
| Discriminator Conv2D | `discriminator.py` | `__init__()` | ~55-62 |
| Mel L1 loss | `losses.py` | `MelL1Loss` | ~20 |
| Feature matching loss | `losses.py` | `feature_matching_loss()` | ~36 |
| Generator total loss | `losses.py` | `generator_loss()` | ~65 |
| Discriminator hinge loss | `losses.py` | `discriminator_loss()` | ~93 |
| Data loading | `train.py` | `HiFiGANDataset` | ~118 |
| Mel computation | `train.py` | `compute_mel()` | ~145 |
| Phase I training | `train.py` | `train_epoch_mel_only()` | ~200 |
| Phase II training | `train.py` | `train_epoch()` | ~293 |
| Main loop | `train.py` | `training_loop()` | ~380 |
| Config | `config.py` | `HiFiGANConfig` | ~1 |

---

## 10. How to Run Training

```bash
# Phase I: Generator only (30 epochs)
# Edit CONFIG["mode"] = "meltrain" in train.py
python src/hifigan/train.py

# Phase II: Full GAN (60 epochs)
# Edit CONFIG["mode"] = "train" in train.py
python src/hifigan/train.py
```

**Models saved:**
- `models/hifigan_generator_meltrain_best.pth` — after Phase I
- `models/hifigan_generator_train_best.pth` — after Phase II
- `models/hifigan_generator_train.pth` — final checkpoint

**At inference time, you only need the Generator:**
```python
from src.hifigan.generator import HiFiGANGenerator
from src.hifigan.config import config

generator = HiFiGANGenerator(config)
checkpoint = torch.load("models/hifigan_generator_train_best.pth")
generator.load_state_dict(checkpoint["generator"])

# Your VAE produces a mel spectrogram
mel = vae.generate("dog")        # [1, 64, 82]

# HiFi-GAN converts to audio
audio = generator(mel)           # [1, 1, 16400]
torchaudio.save("dog_bark.wav", audio.squeeze(0), 22050)
```

---

*Updated 2026-05-11. Refer to actual source code for exact line numbers as files may change.*
