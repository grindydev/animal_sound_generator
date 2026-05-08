# Audio Generation Workflow — Learning Notes

> A beginner-friendly reference covering everything from raw audio to HiFi-GAN.

---

## 1. Raw Audio: What Is a Sample?

Sound is air pressure wiggling. A microphone measures it many times per second.

```
Air pressure
    ↑    ╱╲        ╱╲
    │   ╱  ╲      ╱  ╲
  0 ├──╱────╲────╱────╲───→ Time
    │ ╱              ╲
    └──────────────────
         ↑    ↑    ↑
      samples: each dot is one number
```

- **Sample** = one number (like `0.5`, `-0.3`, `0.8`)
- **Sample rate** = how many samples per second (`22050` = 22,050 samples/sec)
- **1 second of audio** = a list of 22,050 numbers
- Range: typically **-1 to +1**
  - `0` = silence
  - `1` = max push
  - `-1` = max pull

```python
# 1 second of audio at 22,050 Hz
audio = [0.1, -0.3, 0.7, -0.9, 0.2, 0.5, -0.1, 0.8, ...]  # 22,050 numbers
```

---

## 2. Mel Spectrogram: A "Photo" of Sound

Raw audio `[22050]` is just a wiggly line. Hard for a neural network to "see" patterns. So we convert it to a **2D grid** — like a grayscale image.

### How It's Made

Chop the audio into chunks, then ask: "What frequencies are loud in each chunk?"

```
Audio: [0.1, -0.3, 0.7, -0.9, 0.2, 0.5, ...]  ← 8,200 samples

hop_length = 200

Chunk 0:  samples 0-199    → "mostly 500Hz"
Chunk 1:  samples 200-399  → "mostly 500Hz + 1kHz"
Chunk 2:  samples 400-599  → "mostly 1kHz"
...
```

For each chunk, compute **64 frequency buckets** (how loud is low, mid, high...).

### The Grid

```
            Time →
            frame0  frame1  frame2  ...  frame40
            (0ms)   (9ms)   (18ms)       (369ms)
Freq ↓
 bin0   [  0.1     0.2     0.3    ...    0.0  ]  ← deep rumble
 bin1   [  0.3     0.5     0.8    ...    0.1  ]
 bin2   [  0.0     0.1     0.2    ...    0.3  ]
  ...   [  ...     ...     ...    ...    ...  ]
 bin63  [  0.2     0.1     0.0    ...    0.5  ]  ← high whistle

Shape: [64, 41]  ← 64 frequency bins × 41 time frames
```

### Why Not Just Use Raw Audio?

| | Raw Audio | Mel Spectrogram |
|--|-----------|-----------------|
| Shape | `[22050]` (1D line) | `[64, 110]` (2D grid) |
| Pattern | Hard to see | Easy to see |
| Size | Big | 200× smaller |
| VAE can generate? | Hard | Easy |

Think of it like: raw audio is a wiggly line drawing. Mel spectrogram is a barcode/fingerprint — structured and compact.

### Key Terms

| Term | Value | Meaning |
|------|-------|---------|
| `sample_rate` | 22,050 | Samples per second |
| `hop_length` | 200 | How many audio samples per mel frame |
| `n_mels` | 64 | Number of frequency buckets (arbitrary, usually 64/80/128) |

**Formula:**
```
mel_frames = audio_samples / hop_length
# 8200 / 200 = 41 frames
```

> **Important:** `hop_length` is a fixed setting. If you change it, you must regenerate your entire dataset and retrain everything.

### Why 64 Frequency Buckets?

64 = how many "equalizer bars" you split the sound into. NOT "64 sounds humans hear." It's arbitrary:
- 32 = tiny, fast, blurry
- 64 = good balance (this project)
- 80 = sharper (common in papers)
- 128 = very sharp

More bins = finer frequency detail. 64 is enough for animal sounds.

---

## 3. HiFi-GAN Generator: Mel → Waveform

HiFi-GAN takes the compact mel grid and **reconstructs** the full audio wave.

```
Input:  [B, 64, 41]    ← mel spectrogram (B = batch size)
            ↓
     HiFi-GAN Generator
            ↓
Output: [B, 1, 8200]   ← audio waveform (41 × 200 = 8200 samples)
```

### Full Shape Evolution

Using **B=1, T=41** (0.37 second clip):

```
Stage        Operation                    Shape In        Shape Out       Time Change
──────────────────────────────────────────────────────────────────────────────────────
Input        —                            [1, 64, 41]     [1, 64, 41]     —
Pre-conv     Conv1d(64→256, k=7, p=3)     [1, 64, 41]     [1, 256, 41]    41 → 41 (no change)
MRF Block 1  ConvTranspose(stride=5)      [1, 256, 41]    [1, 128, 205]   41 × 5 = 205
MRF Block 2  ConvTranspose(stride=5)      [1, 128, 205]   [1, 64, 1025]   205 × 5 = 1025
MRF Block 3  ConvTranspose(stride=4)      [1, 64, 1025]   [1, 32, 4100]   1025 × 4 = 4100
MRF Block 4  ConvTranspose(stride=2)      [1, 32, 4100]   [1, 16, 8200]   4100 × 2 = 8200
Post-conv    Conv1d(16→1, k=7, p=3)       [1, 16, 8200]   [1, 1, 8200]    8200 → 8200 (no change)
```

**Total upsample:** `5 × 5 × 4 × 2 = 200 = hop_length` ✅

---

## 4. Key Concepts

### 4.1 Kernel vs Stride vs Padding

| Term | What it controls | In pre_conv | In ConvTranspose |
|------|-----------------|-------------|------------------|
| **Kernel size** | Width of the sliding window | 7 | 10, 10, 8, 4 |
| **Stride** | How many steps to hop | 1 (time stays same) | 5, 5, 4, 2 (time grows) |
| **Padding** | Zeros added at edges | 3 | calculated to keep math clean |

**The #1 rule:**
- `stride = 1` → output time = input time (no change)
- `stride > 1` in ConvTranspose → output time = input time × stride (stretches!)

### 4.2 Conv1d: How 64 Becomes 256

```python
nn.Conv1d(in_channels=64, out_channels=256, kernel_size=7)
```

**NOT one kernel.** **256 separate kernels**, each with shape `[64, 7]`.

```
Kernel 0:   [64, 7] weights → produces output channel 0
Kernel 1:   [64, 7] weights → produces output channel 1
...
Kernel 255: [64, 7] weights → produces output channel 255
```

At each time position:
1. Look at 7 time steps across all 64 input channels = 448 values
2. Each of 256 filters does its own weighted sum + bias
3. Output: 256 numbers

**Total params:** `256 × 64 × 7 = 114,688 weights` + `256 biases` = **114,944**

### 4.3 ConvTranspose1d (Upsample)

How it stretches time:
1. Insert zeros between samples
2. Run a normal conv to fill the gaps

```
Input:     [A, B]
           ↓ insert zeros (stride=2)
Stretched: [A, 0, B, 0]
           ↓ conv with kernel=4 fills the gaps
Output:    [smooth A, smooth between, smooth B, smooth between]
```

Why `kernel = 2 × stride`? Bigger kernels = smoother fill, less artifacts.

### 4.4 Dilation

Dilation = **gaps between kernel teeth** (like a comb).

```
Normal (dilation=1):     ●●●      ← looks at 3 adjacent samples
                         ───
                         width = 3

Dilated (dilation=2):    ● ● ●    ← 1-pixel gap between
                         ─────
                         width = 5 (sees farther!)

Dilated (dilation=3):    ●   ●   ●  ← 2-pixel gap
                         ───────
                         width = 7
```

**Same number of weights (3), but sees a wider area.** HiFi-GAN uses `(1, 3, 5)` to catch fast, medium, and slow patterns at once.

### 4.5 LeakyReLU

An activation function. Applied after every conv layer.

```
Input:   -3   -1    0    1    3
ReLU:     0    0    0    1    3     ← negatives die forever
LeakyReLU(0.1):
         -0.3  -0.1   0    1    3     ← negatives survive, shrunk by 10×
```

Why LeakyReLU? Normal ReLU kills negative neurons forever ("dead neurons"). LeakyReLU keeps them alive so gradients can flow back and they can learn.

### 4.6 ResBlock

```
Input x ────────────────────────┐
                                │ ← skip connection
     x → LeakyReLU → Conv → LeakyReLU → Conv
                                                │
                                                ▼
                                        x + output (add them)
```

The `+ x` is the residual/skip connection. If the conv layers mess up, the original signal still gets through. This lets you stack many layers without the signal dying.

### 4.7 MRF Block

```
                    Input
                      │
                      ▼
              ConvTranspose (upsample)
                      │
          ┌─────┬─────┴─────┬─────┐
          │     │           │     │
          ▼     ▼           ▼     ▼
      ResBlock ResBlock ResBlock
      k=3     k=7      k=11
      (fast)  (mid)    (slow)
          │     │           │
          └─────┴─────┬─────┘
                      │
                    SUM
```

Three parallel paths with different kernel sizes (3, 7, 11) catch patterns at different time scales, then add them together.

---

## 5. Discriminator (Brief)

Checks if audio is real or fake.

Your project uses **MPD only** (Multi-Period Discriminator):

```
Audio waveform [B, 1, 8200]
       │
       ├─── Period=2  → fold into grid → Conv2d → score
       ├─── Period=3  → fold into grid → Conv2d → score
       ├─── Period=5  → fold into grid → Conv2d → score
       ├─── Period=7  → fold into grid → Conv2d → score
       └─── Period=11 → fold into grid → Conv2d → score
```

Each period catches repeating artifacts at different frequencies. Returns scores + internal features.

---

## 6. Training Flow (One Batch)

```
BATCH: real_audio = [8, 1, 16384]  ← 8 clips, 0.74s each

STEP A: Generate fake
  real_audio → compute_mel() → [8, 64, 82]
  generator(mel) → fake_audio [8, 1, 16384]

STEP B: Train Discriminator
  D(real_audio) → high scores ✓
  D(fake_audio.detach()) → low scores ✓
  d_loss.backward()
  optimizer_d.step()

STEP C: Train Generator
  D(fake_audio) → scores + features
  Compute 3 losses:
    L_mel  (weight 45): mel(fake) vs mel(real)  ← CONTENT
    L_fm   (weight 2):  match D's inner layers  ← REALISM
    L_adv  (weight 1):  make D say "real"       ← POLISH
  g_loss = 45×L_mel + 2×L_fm + 1×L_adv
  g_loss.backward()
  optimizer_g.step()
```

Mel loss MUST dominate (45×) or the generator drifts to wrong content.

---

## 7. Quick Formula Cheat Sheet

```
# Audio ↔ Mel relationship
mel_frames = audio_samples / hop_length
audio_samples = mel_frames × hop_length

# Conv1d output length (stride=1)
output_length = input_length + 2×padding - kernel_size + 1
# With padding=(kernel-1)//2, output_length ≈ input_length

# ConvTranspose1d output length
output_length = input_length × stride

# Total upsample must equal hop_length
5 × 5 × 4 × 2 = 200
```

---

## 8. Glossary

| Term | Simple Meaning |
|------|---------------|
| **Sample** | One number = air pressure at one instant |
| **Waveform** | List of samples = the actual sound you hear |
| **Mel spectrogram** | 2D grid = "photo" of sound (freq × time) |
| **hop_length** | Audio samples per mel frame (200 in this project) |
| **n_mels** | Number of frequency buckets (64) |
| **Channel** | Like "color" — one slice of data |
| **Kernel** | A filter/sliding window with learned weights |
| **Stride** | How many steps the kernel hops |
| **Padding** | Zeros added at edges so kernel can slide to the border |
| **Dilation** | Gaps between kernel teeth — sees wider without more weights |
| **ConvTranspose** | Upsample layer — stretches time by inserting zeros |
| **ResBlock** | Conv + skip connection — prevents signal from dying in deep nets |
| **LeakyReLU** | Activation that lets negative values survive (prevents dead neurons) |
| **Feature matching loss** | "Make fake audio look like real audio inside the D's brain" |
| **Adversarial loss** | "Fool the discriminator into saying real" |
| **Mel loss** | "Your output must have the same spectrogram as the input" |

---

## 9. The Big Pipeline

```
"dog" label
    ↓
┌──────────┐
│   VAE    │  ← generates "what should it look like"
└────┬─────┘
     ↓
[64, 41] mel spectrogram  ← blurry thumbnail of sound
     ↓
┌──────────┐
│ HiFi-GAN │  ← converts "thumbnail" → "full photo"
│Generator │
└────┬─────┘
     ↓
[1, 8200] audio waveform  ← crisp sound you can hear!
```

---

*Generated from conversation on 2026-05-08. Refer back to `phase-7a-hifigan.md` for deeper architecture details and training tips.*
