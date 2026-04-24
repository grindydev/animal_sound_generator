# Autoencoder Improvement Plan — Lowering MSE

## Current Baseline

After initial fixes (latent_dim 256→1024, removed ReLU from output, weight_decay 0.05→0.001):

```
Phase 3 Autoencoder — Train Mode, 40 Epochs
Val MSE: 0.065237 (epoch 11)
Test MSE: ~0.065
```

This document describes 5 improvement steps to push MSE lower. These changes will be implemented in **Phase 7c** (U-Net skip connections) and applied to the VAE generator.

---

## Step 1: Skip Connections (U-Net Architecture) — Highest Impact

### Why MSE is High Without Skips

The encoder compresses information through 4 layers:

```
Input    [1, 64, 552]     35,328 pixels of fine detail
Layer 1  [32, 32, 276]     8,832 — edges, local texture
Layer 2  [64, 16, 138]     3,532 — shapes, patterns
Layer 3  [128, 8, 69]      1,766 — structures
Layer 4  [256, 4, 35]        896 — abstract concepts only
```

The decoder must reconstruct 35,328 pixels from just 896. Fine detail (edges, texture) is **permanently lost** because stride=2 convolution discards half the spatial info at each layer. The bottleneck cannot carry that information regardless of latent_dim size.

### How Skip Connections Fix This

Copy each encoder layer's output directly to the matching decoder layer. The decoder receives both:
- The upsampled signal from the bottleneck (big-picture structure)
- The original high-resolution features from the encoder (fine detail)

```
Encoder Layer 1 [32, 32, 276]  ────skip────→  Decoder Layer 4  (input: 32+32=64ch)
Encoder Layer 2 [64, 16, 138]  ────skip────→  Decoder Layer 3  (input: 64+64=128ch)
Encoder Layer 3 [128, 8, 69]   ────skip────→  Decoder Layer 2  (input: 128+128=256ch)
```

The bottleneck only needs to learn "what changed" — the skips carry "what stayed the same."

### Industry Context

Skip connections are the backbone of:
- **U-Net** (Ronneberger et al., 2015) — the standard for image reconstruction
- **Stable Diffusion** — uses U-Net for denoising
- **DALL-E** — uses similar architecture for image generation
- **Every modern segmentation model** — Deeplab, Mask R-CNN, etc.

### Code Change — `model.py`

The `forward()` method can no longer use `nn.Sequential` for the encoder because we need **intermediate outputs** from each block. We must extract them manually:

```python
# BEFORE — everything in Sequential, can't get intermediate outputs:
self.encode = nn.Sequential(
    SimpleEncoderBlock(1, 32),
    SimpleEncoderBlock(32, 64),
    ...
)

# AFTER — separate blocks so we can grab skip connections:
self.enc1 = SimpleEncoderBlock(1, 32)     # → skip1
self.enc2 = SimpleEncoderBlock(32, 64)    # → skip2
self.enc3 = SimpleEncoderBlock(64, 128)   # → skip3
self.enc4 = SimpleEncoderBlock(128, 256)
```

Decoder blocks now receive **concatenated** input (decoder output + skip):

```python
# BEFORE:
SimpleDecoderBlock(256, 128)   # input: 256ch

# AFTER (skip adds channels from encoder):
SimpleDecoderBlock(256 + 128, 128)  # input: 384ch (256 from bottleneck + 128 skip)
SimpleDecoderBlock(128 + 64, 64)    # input: 192ch (128 from dec + 64 skip)
SimpleDecoderBlock(64 + 32, 32)     # input: 96ch (64 from dec + 32 skip)
SimpleDecoderBlock(32, 1, activation=False)  # no skip for final layer
```

In `forward()`, use `torch.cat([decoder_output, skip], dim=1)` to concatenate skip connections along the channel dimension:

```python
def forward(self, x):
    # Encode — save intermediate outputs for skip connections
    s1 = self.enc1(x)     # [B,  32, 32, 280]
    s2 = self.enc2(s1)    # [B,  64, 16, 140]
    s3 = self.enc3(s2)    # [B, 128,  8,  70]
    z = self.enc4(s3)     # [B, 256,  4,  35]

    # Bottleneck
    z = z.flatten(start_dim=1)
    z = self.fc_encode(z)
    z = self.fc_decode(z)
    z = z.view(-1, 256, 4, 35)

    # Decode — concatenate skip connections along channel dim
    z = self.dec1(torch.cat([z, s3], dim=1))   # [256+128, 4, 35] → [128, 8, 70]
    z = self.dec2(torch.cat([z, s2], dim=1))   # [128+64, 8, 70]  → [64, 16, 140]
    z = self.dec3(torch.cat([z, s1], dim=1))   # [64+32, 16, 140] → [32, 32, 280]
    z = self.dec4(z)                            # [32, 32, 280]    → [1, 64, 560]

    return z
```

> **⚠️ Size alignment:** Skip connections require encoder and decoder feature maps to have matching spatial dimensions. This only works cleanly when the input is padded to a multiple of 16 — see **Step 4** below. Steps 1 and 4 should be implemented together.

---

## Step 2: Multi-Layer Bottleneck

### Why

The bottleneck does `35,840 → 1,024 → 35,840` in **two single linear layers**. That's a 35× compression jump with no nonlinearity in between. The network has to learn an extremely sharp projection in a single step.

Adding a hidden layer with ReLU makes the compression **gradual**:

```
Current:    35,840 ────────────→ 1,024 ────────────→ 35,840
                 (one giant 35× jump)    (one giant 35× expansion)

Better:     35,840 → 4,096 → 1,024 → 4,096 → 35,840
              9× down   4× down  center  4× up    9× up
            (gradual compression)    (gradual expansion)
```

The ReLU between layers lets the network learn a **hierarchical** compression — first remove obvious redundancy, then compress what's left.

### Code Change — `model.py` `__init__`

```python
# BEFORE:
self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

# AFTER:
hidden = latent_dim * 4  # 4,096 — intermediate compression stage
self.fc_encode = nn.Sequential(
    nn.Linear(self.flat_dim, hidden),
    nn.ReLU(),
    nn.Linear(hidden, latent_dim),
)
self.fc_decode = nn.Sequential(
    nn.Linear(latent_dim, hidden),
    nn.ReLU(),
    nn.Linear(hidden, self.flat_dim),
)
```

No change needed in `forward()` — `self.fc_encode(z)` and `self.fc_decode(z)` still work since `nn.Sequential` is callable.

---

## Step 3: Remove BatchNorm from Decoder

### Why

BatchNorm normalizes each channel using **batch statistics** (mean and variance across all samples in the batch). This is great in the encoder for training stability, but harmful in the decoder for three reasons:

1. **Batch-dependent output** — The same input produces different output depending on what else is in the batch. At inference with `batch_size=1`, the statistics differ from training → reconstruction quality drops.

2. **Constrains output range** — BN forces activations toward zero-mean/unit-variance, but the normalized spectrogram values span a specific range (roughly -1.5 to +2.3 after SimpleNormalize). The decoder needs freedom to output exact values, not normalized ones.

3. **Small batch noise** — With `batch_size=16`, batch statistics are noisy, adding random perturbation to the reconstruction.

### Industry Context

Modern generative models (StyleGAN, BigGAN-deep) replace BatchNorm in generators with:
- **Weight normalization** — normalize weights, not activations
- **Instance normalization** — normalize per-sample (no batch dependency)
- **No normalization** — let the model learn the exact output range

For our simple autoencoder, the easiest approach is to simply remove BatchNorm from decoder blocks.

### Code Change — `SimpleDecoderBlock`

```python
# BEFORE:
class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super(SimpleDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU() if activation else nn.Identity(),
        )

# AFTER:
class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super(SimpleDecoderBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        ]
        if activation:
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)
```

BatchNorm stays in `SimpleEncoderBlock` — it helps training stability there.

---

## Step 4: Fix the `F.interpolate` Hack — Pad Input to Even Dimensions

### Why

The root cause of all size mismatches: **552 is not perfectly divisible by 16** (552/16 = 34.5). Each stride=2 layer rounds down on odd numbers, causing 1-pixel drift that compounds:

```
Input 552 → 276 → 138 → 69 → 34  (rounds down from 34.5)
Decode back: 34 → 68 → 136 → 272 → 544  (lost 8 pixels)
Then F.interpolate(544 → 552) — bilinear stretch blurs the output
```

If we pad the input to **560** (35×16 = 560, perfectly divisible by 16):

```
Input 560 → 280 → 140 → 70 → 35 → 70 → 140 → 280 → 560
Every size is EXACT. No rounding. No F.interpolate needed.
```

### Why This Matters for Reconstruction Quality

`F.interpolate` with `mode='bilinear'` is a non-learnable, fixed mathematical operation. It spreads 544 pixels across 552 positions by blending adjacent values. This **blurs** the output — every pixel becomes a weighted average of its neighbors. The decoder spent capacity learning fine detail, and then we blur it all away with interpolation.

Cropping (560→552) just removes 8 columns of padding — no blurring, no information loss.

### Code Change — `model.py` `forward()`

```python
# BEFORE:
def forward(self, x):
    target_size = x.shape[2:]
    # ... encode/decode ...
    z = nn.functional.interpolate(z, size=target_size, mode='bilinear')
    return z

# AFTER:
def forward(self, x):
    B, C, H, W = x.shape
    pad_w = (16 - W % 16) % 16  # pad time to nearest multiple of 16
    x = nn.functional.pad(x, (0, pad_w))  # [B, 1, 64, 560]

    # ... encode/decode ... (all sizes align perfectly now)

    z = z[:, :, :H, :W]  # crop back to original size [B, 1, 64, 552]
    return z
```

The `(16 - W % 16) % 16` formula:
- W=552: `16 - 552%16 = 16 - 8 = 8` → pad to 560
- W=560: `16 - 560%16 = 16 - 0 = 16`, but `16 % 16 = 0` → no padding needed
- W=544: `16 - 544%16 = 16 - 0 = 0` → no padding needed

This works for **any** input width, not just 552.

---

## Step 5: Combined Loss (MSE + L1)

### Why

**MSE loss** (L2) penalizes errors quadratically — large errors are punished heavily, but many small errors are tolerated. The result: the model produces **smooth, blurry** reconstructions because that minimizes expected squared error. Fine detail (sharp edges, transient sounds) gets averaged away.

**L1 loss** penalizes errors linearly — every pixel error contributes equally. This produces **sharper** reconstructions because the model isn't "afraid" of small absolute errors to preserve edges.

**Combining both** gives the best of both worlds:
- MSE keeps overall energy/structure correct
- L1 preserves sharp edges and fine detail

```python
loss = 0.5 * MSE(reconstructed, target) + 0.5 * L1(reconstructed, target)
```

### Industry Context

Combined MSE+L1 loss is common in:
- **Super-resolution** (ESRGAN, Real-ESRGAN) — pixel loss + perceptual loss
- **Image reconstruction** — MSE for structure, L1 for sharpness
- **Audio source separation** — multi-scale reconstruction losses

### Code Change — `train_autoencoder.py`

```python
# BEFORE:
loss_function = nn.MSELoss()

# AFTER:
class ReconstructionLoss(nn.Module):
    """Combined MSE + L1 loss for sharper spectrogram reconstruction."""
    def __init__(self, mse_weight=0.5, l1_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + self.l1_weight * self.l1(pred, target)

loss_function = ReconstructionLoss()
```

No other changes needed — `loss_function(pred, target)` is called the same way in `train_epoch()` and `validate_epoch()`.

---

## Full Combined Architecture (Steps 1-5 Together)

```python
class SimpleDecoderBlock(nn.Module):
    """Decoder block without BatchNorm (Step 3)."""
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 2, 2)]
        if activation:
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleAudioAutoencoder(nn.Module):
    """U-Net autoencoder with skip connections, multi-layer bottleneck, and padding fix."""
    def __init__(self, latent_dim=1024):
        super().__init__()

        # Encoder (separate blocks — need intermediate outputs for skips)
        self.enc1 = SimpleEncoderBlock(1, 32)       # → skip1
        self.enc2 = SimpleEncoderBlock(32, 64)      # → skip2
        self.enc3 = SimpleEncoderBlock(64, 128)     # → skip3
        self.enc4 = SimpleEncoderBlock(128, 256)    # → deepest

        # Multi-layer bottleneck (Step 2)
        self.flat_dim = 256 * 4 * 35
        hidden = latent_dim * 4
        self.fc_encode = nn.Sequential(
            nn.Linear(self.flat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.flat_dim),
        )

        # Decoder (in_channels = decoder_ch + skip_ch) (Steps 1 + 3)
        self.dec1 = SimpleDecoderBlock(256 + 128, 128)
        self.dec2 = SimpleDecoderBlock(128 + 64, 64)
        self.dec3 = SimpleDecoderBlock(64 + 32, 32)
        self.dec4 = SimpleDecoderBlock(32, 1, activation=False)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_w = (16 - W % 16) % 16       # Step 4: pad to multiple of 16
        x = nn.functional.pad(x, (0, pad_w))

        # Encode — save intermediate outputs for skip connections
        s1 = self.enc1(x)     # [B,  32, 32, 280]
        s2 = self.enc2(s1)    # [B,  64, 16, 140]
        s3 = self.enc3(s2)    # [B, 128,  8,  70]
        z = self.enc4(s3)     # [B, 256,  4,  35]

        # Bottleneck
        z = z.flatten(start_dim=1)
        z = self.fc_encode(z)
        z = self.fc_decode(z)
        z = z.view(-1, 256, 4, 35)

        # Decode with skip connections
        z = self.dec1(torch.cat([z, s3], dim=1))   # [384, 4, 35] → [128, 8, 70]
        z = self.dec2(torch.cat([z, s2], dim=1))   # [192, 8, 70] → [64, 16, 140]
        z = self.dec3(torch.cat([z, s1], dim=1))   # [96, 16, 140] → [32, 32, 280]
        z = self.dec4(z)                            # [32, 32, 280] → [1, 64, 560]

        # Crop back to original size (no interpolation)
        z = z[:, :, :H, :W]  # [1, 64, 560] → [1, 64, 552]

        return z
```

---

## Expected Impact

| Step | Change | Expected MSE Range | Difficulty |
|------|--------|-------------------|------------|
| Baseline | Current architecture | 0.065 | — |
| 1 | Skip connections (U-Net) | 0.02–0.04 | Medium |
| 2 | Multi-layer bottleneck | 0.01–0.03 | Easy |
| 3 | Remove decoder BatchNorm | Marginal improvement | Easy |
| 4 | Fix interpolation hack | Marginal (cleaner output) | Easy |
| 5 | MSE + L1 combined loss | 0.01–0.02 | Easy |

**Steps 1+4 together will have the biggest impact.** Steps 2, 3, and 5 provide incremental improvements on top.

---

## Implementation Order

```
1. Steps 1 + 4 together (skip connections + padding fix)
   → These must be done together because skips require exact size alignment

2. Step 2 (multi-layer bottleneck)
   → Simple addition, no structural changes

3. Step 3 (remove decoder BatchNorm)
   → Simple removal, no structural changes

4. Step 5 (combined loss)
   → Only change is in train_autoencoder.py, model is unchanged
```

---

## Files to Modify

| File | Steps | What Changes |
|------|-------|-------------|
| `src/model.py` | 1, 2, 3, 4 | `SimpleDecoderBlock` (remove BN), `SimpleAudioAutoencoder` (skips, multi-layer bottleneck, padding) |
| `src/train_autoencoder.py` | 5 | Replace `nn.MSELoss()` with `ReconstructionLoss` |

This document will be referenced from `roadmap.md` Phase 7c (U-Net skip connections).
