# Workflow Fix Plan v2 — Full Pipeline Review

> **Date:** May 13, 2026  
> **Status:** Generation produces noise. Reconstruction works. Root cause: decoder skip dependency.  
> **Replaces:** `workflow-fix-plan.md` (incorrect diagnosis)

---

## 0. Executive Summary

The pipeline has **5 trained models** on Colab L4 (base_channels=32, 30 epochs):

| Model | File | Size | Status |
|-------|------|------|--------|
| Classifier | `best_audio_cnn_train.pth` | 20 MB | ✅ 95.3% accuracy |
| Autoencoder | `best_autoencoder_train.pth` | 568 MB | ✅ MSE=0.015 (solid) |
| VAE | `best_vae_finetune_train.pth` | 849 MB | ⚠️ Generation broken |
| HiFi-GAN | `hifigan_generator_train_best.pth` | 13 MB | ✅ Works fine |
| Diffusion | `diffusion_unet_train_best.pth` | 68 MB | Untested |

**What works:** Reconstruction (encode real audio → decode with skips → sounds real).  
**What fails:** Generation (random z → decode without skips → quiet noise).

---

## 1. Pipeline Architecture (How It Actually Works)

```
                    TRAINING PATH
                    =============
  Real audio .wav
       ↓
  MelSpectrogram + AmplitudeToDB + SimpleNormalize
       ↓  [B, 1, 64, 552]  mean≈0, std≈1  (normalized dB)
       ↓
  ┌─────────────────────────────────────────┐
  │              ENCODER                     │
  │  enc1 (1→32, stride2)  → skip s0 [32]   │
  │  enc2 (32→64, stride2) → skip s1 [64]   │
  │  enc3 (64→128, stride2)→ skip s2 [128]  │
  │  enc4 (128→256, stride2)→ [256, 4, 35]  │
  │  SelfAttention1D → flatten [35840]       │
  │  fc_mu → μ [2048]                        │
  │  fc_log_var → log σ² [2048]              │
  └──────────────────┬──────────────────────┘
                     │ z = μ + σ·ε
                     │ class_emb = Embedding(label) [128]
                     ▼
  ┌─────────────────────────────────────────┐
  │           DECODER (FiLM)                  │
  │  fc_decode [2048→35840] → [256, 4, 35]  │
  │  dec4 (256→128) + FiLM + concat(s2)     │ ← skip from enc3 (128ch)
  │  dec3 (128→64)  + FiLM + concat(s1)     │ ← skip from enc2 (64ch)
  │  dec2 (64→32)   + FiLM + concat(s0)     │ ← skip from enc1 (32ch)
  │  dec1 (32→16)   + FiLM                  │ ← no skip
  │  output_conv → [1, 64, 560]             │
  │  interpolate → [1, 64, 552]             │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
              GENERATION PATH
              ===============
  Random z ~ N(0, T·I) [2048]
  class_emb = Embedding(label) [128]
       ↓
  ┌─────────────────────────────────────────┐
  │  SAME DECODER — BUT NO SKIPS             │
  │  fc_decode → [256, 4, 35]               │
  │  dec4 (256→128) + FiLM                  │ ← NO concat!
  │  dec3 (128→64)  + FiLM                  │ ← NO concat!
  │  dec2 (64→32)   + FiLM                  │ ← NO concat!
  │  dec1 (32→16)   + FiLM                  │
  │  output_conv → [1, 64, 560] → [1,64,552]│
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │             HiFi-GAN                      │
  │  unnormalize: ×19.80 - 18.49 → dB       │
  │  dB → power → Griffin-Lim (optional)    │
  │  generator → waveform [1, 110400]        │
  │  lowpass @ 11025 Hz                      │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
              output .wav @ 22050 Hz
```

### 1.1 Skip Connection Role

Each decoder stage normally receives `enc_skip` from the matching encoder level.  
The skip carries **fine spatial detail** (edges, textures in the spectrogram).  
Without it, the decoder must hallucinate everything from z alone.

```
With skip:     dec4(z_features, class_emb, enc3_features) → concat(128+128=256ch) → project→128ch
Without skip:  dec4(z_features, class_emb, None)           → no concat, stays 128ch
```

The skip provides:
- **Spatial structure**: where high/low frequencies are in time
- **Fine detail**: sharp frequency transitions that MSE loss encourages
- **50% more channels**: 256→128 vs just 128

### 1.2 Current skip_dropout Training

During VAE training (`forward()`):
```python
if self.training and skip_dropout > 0 and torch.rand(1).item() < skip_dropout:
    skips = [None, None, None]
```

With `skip_dropout=0.5`, the decoder sees skips 50% of the time.  
**Problem:** 50% is enough for the decoder to learn to depend on skips.  
When skips are absent, it produces weak, low-detail output.

---

## 2. What's Already Correct

| Component | Status | Evidence |
|-----------|--------|----------|
| **Normalization** | ✅ Correct | SimpleNormalize(mean=-18.49, std=19.80) matches HiFi-GAN config exactly |
| **HiFi-GAN** | ✅ Works | Recon audio sounds right; config matches training |
| **Autoencoder** | ✅ Works | MSE=0.015 with base_ch=32 |
| **Classifier** | ✅ 95.3% | Used for supervision loss in VAE |
| **VAE reconstruction** | ✅ Works | RMS 0.17 vs real 0.15 — nearly identical |
| **base_channels=32** | ✅ Correct | All models trained consistently |
| **Colab configs** | ✅ Optimized | lr, batch, workers, epochs all tuned |

---

## 3. Root Cause: Decoder Skip Dependency

**Confirmed by diagnostic test:**

| Path | z source | Skips | RMS | Result |
|------|----------|:-----:|:---:|--------|
| Real audio | — | — | 0.15 | Baseline dog bark |
| Reconstruction | encoder μ | ✅ | 0.17 | **Sounds correct** ✅ |
| Generation | N(0, T·I) | ❌ | 0.04 | **Quiet noise** ❌ |

The decoder was trained with `skip_dropout=0.5`. This means:
- 50% of training steps: full encoder skips → decoder learns to lean on them
- 50% of training steps: no skips → decoder forced to work alone
- Result: decoder learned to use skips when available, collapses without them

**Why 50% dropout fails:** The decoder has 223M parameters. With skips, it can easily minimize MSE by copying fine detail from encoder features. Without skips, it struggles, but the loss only penalizes it 50% of the time — the other 50% it gets "free" reconstruction from skips. The optimizer tunes parameters for the easy case (with skips), sacrificing the hard case.

### 3.1 Secondary Issue: VAE Output Statistics Mismatch

Even ignoring quality, the VAE generates spectrograms with wrong statistics:

| | Real Mel (normalized) | Generated Mel |
|---|:---:|:---:|
| Mean | 1.03 | **-0.48** |
| Std | 0.45 | 0.40 |
| Range | -0.3 to +2.9 | -4.1 to +5.6 |

Generated mels have a shifted mean and extreme outliers. The decoder's output distribution doesn't match the training distribution when sampling from the prior.

---

## 4. Fix Plan

### Fix 1: Retrain VAE with skip_dropout=1.0 (MANDATORY)

**Goal:** Decoder must work entirely without encoder skip connections.

**Changes to `src/vae/finetune.py`:**
```python
CONFIG = {
    ...
    "skip_dropout": 1.0,       # was 0.5 → ALWAYS drop skips during training
    "warmup_epochs": 5,        # keep: frozen encoder/decoder, β=0
    "ramp_epochs": 15,         # keep: β ramps 0→0.005
    "class_loss_weight": 0.5,  # keep
}
```

**Why skip_dropout=1.0:** The decoder must never see encoder skips. This forces it to:
- Rely entirely on z + FiLM conditioning
- Learn to hallucinate fine detail from class information alone
- Match the exact conditions of `sample()` at inference time

**Trade-off:** Reconstruction MSE will increase (from 0.015 → probably 0.05-0.10) because the decoder can't copy fine detail from skips. But generation will actually work.

**Training time:** ~1.5 hrs on L4 (30 epochs, batch=8).

### Fix 2: Increase FiLM Conditioning Strength

**Goal:** Class embedding must carry enough information to replace skip connections.

**Changes to `src/vae/finetune.py`:**
```python
CONFIG = {
    ...
    "embed_dim": 256,          # was 128 → 2× stronger class signal
    "class_loss_weight": 1.0,  # was 0.5 → stronger classifier feedback
}
```

The FiLM in each decoder block does `h = h * (1+γ) + β`. With 256-dim embeddings (was 128), the MLP has more capacity to learn class-specific modulation patterns. This partially compensates for missing skips.

### Fix 3: Post-Generation Spectrogram Normalization (SAFETY NET)

**Goal:** If VAE output statistics are still off, rescale before HiFi-GAN.

**Add to `src/generate.py` after VAE generation:**
```python
def normalize_vae_output(mel, target_mean=0.0, target_std=1.0):
    """Rescale VAE output to match expected normalized mel statistics."""
    mel_mean = mel.mean()
    mel_std = mel.std()
    if mel_std > 0:
        mel = (mel - mel_mean) / mel_std * target_std + target_mean
    return torch.clamp(mel, -3.0, 3.0)  # clip extreme outliers
```

This is a band-aid, not a fix. Apply only if Fix 1+2 don't fully resolve the issue.

### Fix 4: Disable Griffin-Lim by Default

**Goal:** Griffin-Lim blending degrades HiFi-GAN output.

**Changes to `src/generate.py`:**
```python
# Default: skip Griffin-Lim
parser.add_argument("--griffin-lim", action="store_true",  # was --no-griffin-lim
                    help="Enable Griffin-Lim phase refinement (off by default)")
```

And in `generate_one()`:
```python
waveform = mel_to_waveform(
    vae_mel, device=device,
    use_griffin_lim=args.griffin_lim,  # OFF by default
)
```

**Evidence:** Without GL: RMS=0.014. With GL: RMS=0.005 — GL makes output 3× quieter. The 70/30 GL/HiFi-GAN blend favors a mathematical approximation over the neural waveform.

### Fix 5: Reduce Default Temperature

**Goal:** Keep z closer to the prior mean where decoder was trained.

**Changes to `src/generate.py`:**
```python
parser.add_argument("--temperature", type=float, default=0.5,  # was 0.7
```

With skip_dropout=1.0, the decoder is trained on z values from the encoder (which have non-zero mean). Sampling with lower temperature pulls random z toward zero (the prior mean), keeping it in a region the decoder handles better.

---

## 5. Execution Order

```
Step 1: Apply Fix 1+2 to src/vae/finetune.py
        (skip_dropout=1.0, embed_dim=256, class_loss=1.0)

Step 2: Apply Fix 4+5 to src/generate.py  
        (disable Griffin-Lim, temperature=0.5)

Step 3: Delete old VAE checkpoint + retrain on Colab
        rm models/best_vae_finetune_train.pth
        rm -rf models/vae_checkpoints/
        python src/vae/finetune.py   (~1.5 hrs on L4)

Step 4: Test generation
        python src/generate.py --label Dog --no-diff
        → Should produce recognizable dog-like audio

Step 5: Run full evaluation
        python src/evaluate_gen.py
        → Expected: agreement >50%, MSE ~0.05-0.10

Step 6: If still noise → apply Fix 3 (normalization safety net)
```

---

## 6. What NOT to Change

| Component | Why |
|-----------|-----|
| **Autoencoder** | Already works at MSE=0.015 |
| **HiFi-GAN** | Proven correct via reconstruction test |
| **HiFi-GAN config** | norm_mean/std match data_loader |
| **data_loader / SimpleNormalize** | Values are correct (-18.49, 19.80) |
| **Classifier** | 95.3% accuracy is sufficient |
| **Diffusion** | Not relevant to this fix |
| **base_channels** | 32 is correct for all models |
| **VAE architecture** | FiLM, ResBlocks, attention are fine |
| **Training epochs** | 30 is enough for this dataset |

---

## 7. Success Criteria

| Test | Before Fix | After Fix |
|------|:---:|:---:|
| Generate Dog audio | Noise ❌ | Recognizable bark ✅ |
| Reconstruction RMS | 0.17 | 0.10-0.20 |
| Generation RMS | 0.04 | >0.10 |
| Classification agreement | 67.8% | >50% |
| Audio doesn't hurt ears | ❌ | ✅ |
