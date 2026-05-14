# Workflow Fix Plan v3 — Self-Attention Generation Decoder

> **Date:** May 13, 2026  
> **Status:** Implemented. Training in progress.  
> **Previous plans:** v1 (normalization — wrong), v2 (skip_dropout=1.0 — insufficient) both superseded.

---

## 0. Root Cause (Confirmed)

The FiLMDecoderStage produces white noise without encoder skip connections. Three rounds of testing proved:

| Decoder input | Skips? | Output |
|---------------|:------:|--------|
| Real audio → encoder z | ✅ Yes | ✅ Sounds real (RMS 0.17) |
| Real audio → encoder z | ❌ No | ❌ White noise (Nyquist peak) |
| Random z ~ N(0,T) | ❌ No | ❌ White noise |
| Class-conditional z | ❌ No | ❌ White noise |

**The decoder architecture cannot generate spectrograms without skip connections.** No amount of hyperparameter tuning (skip_dropout, β, embed_dim) can fix this — it's an architectural limitation. Encoder skip connections provide spatial structure (edges, textures, temporal coherence) that the decoder's pure convolutional upsampling cannot hallucinate.

Additionally, the VAE latent space is collapsed: encoder μ ≈ 0 for all classes (std=0.14). The decoder relies on skips, not z, for content.

---

## 1. Fix: Self-Attention Replaces Skip Connections

### 1.1 Architecture Change

Added `gen_attn: SelfAttention1D` to each FiLMDecoderStage. Only activated when `enc_skip is None`:

```
FiLMDecoderStage.forward(h, cond, enc_skip):
    
    h → upsample → conv → GN → FiLM → SiLU → conv → GN → +residual
    
    if enc_skip is not None:
        → concat(h, enc_skip) → proj          # RECONSTRUCTION (unchanged)
    else:
        → gen_attn(h)                           # GENERATION (NEW)
    
    return h
```

**Why self-attention:** Attention along the time axis (`SelfAttention1D`) lets each time step attend to all other time steps. This creates the temporal coherence (rhythm, onsets, sustain) that encoder skips used to provide. It's the standard replacement for skip connections in generation architectures (AudioLDM, Stable Audio, MusicGen all use attention in their VAE decoders).

**Which stages get attention:**
- dec4 (256→128): ✅ gen_attn (128 dims, 4 heads)
- dec3 (128→64):  ✅ gen_attn (64 dims, 4 heads)  
- dec2 (64→32):   ✅ gen_attn (32 dims, 4 heads)
- dec1 (32→16):   No gen_attn (enc_skip_ch=0, this stage never had skips anyway)

### 1.2 Training Strategy

```python
CONFIG = {
    "skip_dropout": 1.0,        # Always generation mode → gen_attn always active
    "embed_dim": 256,           # Strong FiLM conditioning
    "class_loss_weight": 1.0,   # Classifier pushes hard
    "beta": 0.005,              # Same KL weight
    "warmup_epochs": 5,         # Freeze encoder/decoder, train VAE heads + gen_attn
    "ramp_epochs": 15,          # β ramps 0→0.005
}
```

**Loading from autoencoder checkpoint:** `strict=False` handles new gen_attn keys. Encoder/decoder weights are initialized from pretrained AE. gen_attn weights are randomly initialized (standard kaiming in SelfAttention1D).

**Total new parameters:** ~5M (4 attention layers × ~1.3M each). Total VAE: 228M (was 223M).

---

## 2. Implementation

### Files Changed

| File | Change |
|------|--------|
| `src/vae/model.py` | Added `gen_attn: SelfAttention1D` to FiLMDecoderStage |
| `src/vae/finetune.py` | Updated comment for skip_dropout=1.0 |
| `src/generate.py` | Auto-detect embed_dim, no Griffin-Lim, temp=0.5, post-gen normalization |
| `src/evaluate_gen.py` | Auto-detect embed_dim |

### What was NOT changed

- Autoencoder (works, MSE=0.015)
- HiFi-GAN (works, proven with real mels)
- Classifier (95.3%, proven)
- Diffusion (untested but optional)
- data_loader / SimpleNormalize (correct values)
- base_channels=32 (consistent across all models)

---

## 3. Training Schedule (Colab L4)

```
Session 1: Train VAE (~1.5 hrs)
  → models/best_vae_finetune_train.pth

Session 2: Evaluate + Generate
  → python src/evaluate_gen.py
  → python src/generate.py --label Dog --no-diff
```

---

## 4. Success Criteria

| Test | Before Fix | After Fix (Expected) |
|------|:---:|:---:|
| Generate Dog | White noise (Nyquist peak) | Recognizable bark |
| Generate Cat | White noise | Recognizable meow |
| Classification agreement | 67.8% | >60% |
| Reconstruction MSE | 0.015 | 0.02-0.05 (slightly worse without skips) |
| Audio peak freq | 11025 Hz | <5000 Hz |
| Audio doesn't hurt ears | ❌ | ✅ |

---

## 5. If This Doesn't Work

The nuclear option: train a completely separate `GenerationDecoder` that never saw skip connections even during design. ~6 hours of work. But attention-augmented VAE decoders are the industry standard approach — this should work.
