# Phase 3 — Autoencoder (Reconstruct Audio)

## Overview

Your classifier compresses a spectrogram into a **class label** (1 of 8).

```
Spectrogram [1, 64, 552] → CNN → "Dog"
```

An autoencoder compresses a spectrogram into a **latent vector**, then reconstructs it back:

```
Spectrogram → Encoder → 256-dim vector → Decoder → Reconstructed Spectrogram
```

The decoder learns to **generate** spectrograms from a compressed representation.
That decoder becomes your **generator** in Phase 4 (VAE).

---

## Core Concepts

### 1. Why "Autoencoder"?

"Auto" = self. It teaches itself to compress and reconstruct. No labels needed.

```
┌─────────┐     ┌──────────┐     ┌─────────┐
│  Input   │ ──→ │  Latent  │ ──→ │  Output  │
│  (dog    │     │  vector  │     │  (dog    │
│  spectro)│     │  [256]   │     │  spectro)│
└─────────┘     └──────────┘     └─────────┘
     ↑              ↑                ↑
 what we        compressed       must match
 have           representation   the input!
```

Loss = MSE(output, input) — minimize the difference between output and input.

### 2. Encoder (you already know this!)

Your `SimpleAudioCNN` already does the encoder job:

```
Conv2d(1→32) + BN + ReLU + MaxPool     [B, 32, 32, 276]
Conv2d(32→64) + BN + ReLU + MaxPool    [B, 64, 16, 138]
Conv2d(64→128) + BN + ReLU + MaxPool   [B, 128, 8, 69]
Conv2d(128→256) + BN + ReLU + MaxPool  [B, 256, 4, 34]
Flatten → [B, 256*4*34 = 34,816]
```

Difference from classifier: **no AdaptiveAvgPool** and **no Linear(256, 8)**.
We keep the spatial info so the decoder can reverse it.

### 3. Decoder (NEW — ConvTranspose2d)

The decoder mirrors the encoder. Where encoder shrinks, decoder expands.

Tool: `nn.ConvTranspose2d` — also called "deconvolution".

```python
# Encoder: halve spatial size
nn.MaxPool2d(2, 2)  # [B, 256, 8, 69] → [B, 256, 4, 34]

# Decoder: double spatial size (reverse!)
nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
# [B, 256, 4, 34] → [B, 128, 8, 69]
```

Think of it:
- `Conv2d` slides a window **over** input → extracts features, shrinks
- `ConvTranspose2d` spreads each value **back out** → reconstructs, expands

### 4. Why NOT AdaptiveAvgPool?

In the classifier, `AdaptiveAvgPool2d((1,1))` squashes everything to [B, 256, 1, 1].
This destroys ALL spatial information — you can't reverse "average of everything".

For autoencoder: skip AdaptiveAvgPool. Use Flatten directly.
Since all crops are 5s = 552 time frames, sizes are deterministic.

### 5. Reconstruction Loss

```python
# Classifier:  CrossEntropy — "did you pick the right class?"
loss = nn.CrossEntropyLoss()(predictions, labels)

# Autoencoder: MSE — "is each pixel close to the original?"
loss = nn.MSELoss()(reconstructed, original)
```

MSE = Mean Squared Error = average of (predicted - actual)² for every pixel.

### 6. Training Loop (minimal changes from Phase 2!)

```python
# BEFORE (classifier):
outputs = model(spectrograms)           # [B, 8]
loss = CrossEntropyLoss()(outputs, labels)

# AFTER (autoencoder):
reconstructed = model(spectrograms)    # [B, 1, 64, 552]
loss = MSELoss()(reconstructed, spectrograms)  # target = input itself!
```

Same optimizer, same scheduler, same early stopping. Only the loss changes.

---

## Full Architecture Diagram

```
INPUT: Spectrogram [B, 1, 64, 552]
  │
  ▼ ENCODER (same as your CNN, minus the classifier head)
  │
  │  ConvBlock(1→32)   + MaxPool   → [B, 32, 32, 276]
  │  ConvBlock(32→64)  + MaxPool   → [B, 64, 16, 138]
  │  ConvBlock(64→128) + MaxPool   → [B, 128, 8, 69]
  │  ConvBlock(128→256)+ MaxPool   → [B, 256, 4, 34]
  │
  │  Flatten → [B, 256×4×34] = [B, 34,816]
  │  Linear(34,816 → latent_dim)   → [B, 256]   ← compressed!
  │
  ▼ LATENT SPACE
  │
  │  Linear(latent_dim → 256×4×34) → [B, 34,816]
  │  Reshape → [B, 256, 4, 34]
  │
  ▼ DECODER (mirror of encoder)
  │
  │  ConvTransposeBlock(256→128)   → [B, 128, 8, 69]
  │  ConvTransposeBlock(128→64)    → [B, 64, 16, 138]
  │  ConvTransposeBlock(64→32)     → [B, 32, 32, 276]
  │  ConvTransposeBlock(32→1)      → [B, 1, 64, 552]
  │
OUTPUT: Reconstructed Spectrogram [B, 1, 64, 552]
```

---

## Implementation Steps

You will build `src/autoencoder.py` and `src/train_autoencoder.py`.

Each step has:
- **What to implement** — the code
- **Why** — the concept
- **Verify** — how to test it before moving on

### Step 1: EncoderBlock

Reuse your `SimpleAudioCNNBlock` pattern from `model.py`:

```python
class EncoderBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool2d"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)
```

**Why**: Same as your classifier conv blocks. Compresses spatial info while expanding channels.

**Verify**: Pass a fake tensor through and check shape shrinks:
```python
x = torch.randn(2, 1, 64, 552)
block = EncoderBlock(1, 32)
out = block(x)
print(out.shape)  # should be [2, 32, 32, 276]
```

---

### Step 2: DecoderBlock

The mirror of EncoderBlock — expand spatial size instead of shrinking:

```python
class DecoderBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
```

**Why**: `ConvTranspose2d(kernel_size=2, stride=2)` is the inverse of `MaxPool2d(2, 2)`.
It doubles the height and width.

**Verify**:
```python
x = torch.randn(2, 256, 4, 34)
block = DecoderBlock(256, 128)
out = block(x)
print(out.shape)  # should be [2, 128, 8, 68]
```

⚠️ Note: 8, 68 not 8, 69. ConvTranspose2d doubles exactly: 4→8, 34→68.
The last decoder layer may need adjustment to match 552 exactly.
You can use `nn.functional.interpolate` or a final `Conv2d` to fix the last few pixels.
We'll handle this in Step 4.

---

### Step 3: Trace the exact sizes

Before building the full model, trace every shape so there are no surprises.
Write this as a comment in your file.

All input spectrograms are [B, 1, 64, 552] (64 mel bins, 552 time frames for 5s @ 22050Hz).

```
ENCODER:
  Input:                          [B, 1, 64, 552]
  EncoderBlock(1→32)   MaxPool   [B, 32, 32, 276]
  EncoderBlock(32→64)  MaxPool   [B, 64, 16, 138]
  EncoderBlock(64→128) MaxPool   [B, 128, 8, 69]
  EncoderBlock(128→256)MaxPool   [B, 256, 4, 34]
  Flatten:                        [B, 256*4*34] = [B, 34,816]

LATENT:
  Linear(34,816 → latent_dim):   [B, 256]        ← compressed!

DECODER:
  Linear(latent_dim → 34,816):   [B, 34,816]
  Reshape:                        [B, 256, 4, 34]
  DecoderBlock(256→128):          [B, 128, 8, 68]    ← 34*2=68, not 69!
  DecoderBlock(128→64):           [B, 64, 16, 136]   ← 68*2=136, not 138!
  DecoderBlock(64→32):            [B, 32, 32, 272]   ← 136*2=272, not 276!
  DecoderBlock(32→1):             [B, 1, 64, 544]    ← 272*2=544, not 552!

  ⚠️ 544 ≠ 552 — off by 8 time frames. Fix needed!
```

**Two options to fix the size mismatch:**

**Option A**: Use `nn.functional.interpolate` after the last decoder layer to resize to exact target:
```python
# After DecoderBlock(32→1): [B, 1, 64, 544]
output = nn.functional.interpolate(output, size=(64, 552), mode='bilinear')
# → [B, 1, 64, 552] ✅
```

**Option B**: Add padding in encoder so sizes divide evenly by 16:
```python
# In data_loader, pad time dim to next multiple of 16:
# 552 → 560 (552 + 8 zeros of padding)
# Then: 560→280→140→70→35→70→140→280→560, and crop back to 552
```

**Recommend Option A** — simpler, and bilinear interpolation for spectrograms is fine.

---

### Step 4: AudioAutoencoder class

Combine Encoder + Latent + Decoder into one module:

```python
class AudioAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            EncoderBlock(1, 32),
            EncoderBlock(32, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
        )

        # Latent bottleneck
        # After 4 encoder blocks: [B, 256, 4, 34] → flatten → [B, 34,816]
        self.flat_dim = 256 * 4 * 34  # 34,816

        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # Decoder
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 1),
        )

    def forward(self, x):
        # Save original size for final interpolation
        target_size = x.shape[2:]  # (64, 552)

        # Encode
        z = self.encoder(x)             # [B, 256, 4, 34]
        z = z.flatten(start_dim=1)       # [B, 34,816]
        z = self.fc_encode(z)            # [B, latent_dim]  ← compressed!

        # Decode
        z = self.fc_decode(z)            # [B, 34,816]
        z = z.view(-1, 256, 4, 34)       # [B, 256, 4, 34]
        z = self.decoder(z)              # [B, 1, 64, 544]

        # Fix size mismatch (544 → 552)
        z = nn.functional.interpolate(z, size=target_size, mode='bilinear')

        return z
```

**Verify**: Shape test before training:
```python
model = AudioAutoencoder(latent_dim=256)
x = torch.randn(4, 1, 64, 552)  # fake spectrogram batch
output = model(x)
print(f"Input:  {x.shape}")      # [4, 1, 64, 552]
print(f"Output: {output.shape}") # should be [4, 1, 64, 552]
assert output.shape == x.shape, "Shape mismatch!"
```

---

### Step 5: train_autoencoder.py

Copy `train.py` as a starting point. Key changes:

```python
# 1. Import autoencoder instead of classifier
from autoencoder import AudioAutoencoder

# 2. Loss = MSE (not CrossEntropy)
loss_function = nn.MSELoss()

# 3. No labels needed in loss computation!
# BEFORE:
#   outputs = model(spectrograms)
#   loss = loss_function(outputs, labels)

# AFTER:
#   reconstructed = model(spectrograms)
#   loss = loss_function(reconstructed, spectrograms)  # target = input!

# 4. Track MSE loss instead of accuracy for early stopping
# Lower MSE = better reconstruction
# "Best model" = lowest val MSE (not highest val accuracy)
```

**Verify**: Overfit 1 batch as a sanity check:
```python
# Take 1 batch, train 50 epochs on it
# MSE should drop to near 0 — the model memorizes those exact spectrograms
# If MSE doesn't drop, there's a bug in the architecture
```

---

### Step 6: Evaluate reconstruction quality

After training, visualize original vs reconstructed spectrograms:

```python
# Get a batch
spectrograms, labels = next(iter(test_loader))
spectrograms = eval_transform(spectrograms.to(device))

# Reconstruct
reconstructed = model(spectrograms)

# Plot side by side
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axes[0, i].imshow(spectrograms[i, 0].cpu(), origin='lower', aspect='auto')
    axes[0, i].set_title(f'Original ({CLASS_NAMES[labels[i]]})')
    axes[1, i].imshow(reconstructed[i, 0].detach().cpu(), origin='lower', aspect='auto')
    axes[1, i].set_title('Reconstructed')
plt.tight_layout()
plt.show()
```

**What to look for:**
- Sharp spectrograms → good reconstruction (model preserved details)
- Blurry spectrograms → model lost fine details (latent dim too small?)
- Missing frequency bands → encoder couldn't capture full range

---

### Step 7: Record results

Fill in this table after training:

```
┌──────────────────────────────────────────────┐
│ AUTOENCODER                                  │
│ Latent dim:        ???                       │
│ Train MSE:         ???                       │
│ Val MSE:           ???                       │
│ Does it look like original?  Yes/No          │
│ What's lost in compression?  ???             │
│ Best model: models/best_autoencoder.pth      │
└──────────────────────────────────────────────┘
```

---

## Checkpoint Questions

Answer these before you start implementing:

1. **Why doesn't the autoencoder need labels?**
   → The target IS the input. Loss = MSE(output, input).

2. **What does ConvTranspose2d do that Conv2d doesn't?**
   → Expands spatial dimensions (doubles H and W with stride=2).

3. **Why can't you use AdaptiveAvgPool in the autoencoder encoder?**
   → It destroys spatial info (averages everything to 1×1). Decoder can't reverse that.

4. **Why is the output 544 instead of 552?**
   → Each MaxPool2d(2) halves exactly: 552→276→138→69→34.
   ConvTranspose2d doubles back: 34→68→136→272→544.
   552 is not divisible by 16, so we lose 8 pixels.

5. **Why is the decoder the exciting part?**
   → It learned to generate spectrograms from a compressed vector.
   In Phase 4, we add class conditioning + random sampling = generator!

---

## Files to Create

```
src/
├── autoencoder.py         ← Step 1-4: EncoderBlock, DecoderBlock, AudioAutoencoder
└── train_autoencoder.py   ← Step 5-6: Training loop + evaluation
```

## Progress Tracker

| Step | Description | Verify | Status |
|------|------------|--------|--------|
| 1 | EncoderBlock | Shape test [B,1,64,552] → [B,32,32,276] | 🔲 |
| 2 | DecoderBlock | Shape test [B,256,4,34] → [B,128,8,68] | 🔲 |
| 3 | Trace exact sizes | Comment with all shapes | 🔲 |
| 4 | AudioAutoencoder | Output shape == input shape | 🔲 |
| 5 | train_autoencoder.py | Overfit 1 batch (MSE → 0) | 🔲 |
| 6 | Evaluate + visualize | Original vs reconstructed plots | 🔲 |
| 7 | Record results | Fill in the table above | 🔲 |
