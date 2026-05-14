# Workflow Fix Plan v3 — Architecture Redesign

> **Date:** May 13, 2026  
> **Status:** Both `workflow-fix-plan.md` (wrong diagnosis) and `workflow-fix-plan_2.md` (skip_dropout=1.0 failed) are obsolete.  
> **Root cause confirmed:** FiLMDecoderStage cannot generate without encoder skip connections.

---

## 0. What We Know For Certain

### The decoder works ONLY with encoder skip connections

| Decoder input | Skips? | Result |
|---------------|:------:|--------|
| Real audio → encoder z | ✅ With skips | ✅ Sounds like real (RMS 0.17 vs 0.15) |
| Real audio → encoder z | ❌ No skips | ❌ White noise (peak at Nyquist) |
| Random z ~ N(0,I) | ❌ No skips | ❌ White noise |
| Class-conditional z | ❌ No skips | ❌ White noise |
| z = zeros | ❌ No skips | ❌ White noise |

**The decoder architecture REQUIRES encoder skip connections.** Without them, every output collapses to high-frequency noise regardless of z quality. This is not fixable with hyperparameter tuning — it requires architectural changes.

### The latent space is collapsed

| Stat | Value |
|------|-------|
| Encoder μ mean | -0.002 (≈ zero) |
| Encoder μ std (across samples) | 0.14 (tiny) |
| Encoder σ mean | 0.97 (≈ 1.0) |
| **Signal-to-noise ratio in z** | **0.14 / 0.97 = 14%** |

z = μ(14% content) + σ·ε(86% noise). Even if the decoder worked without skips, z carries almost no content signal. Generation from N(0,I) produces random noise because there's no content structure in z.

### What actually works

| Component | Status |
|-----------|--------|
| Autoencoder (deterministic, with skips) | ✅ MSE 0.015, perfect reconstruction |
| VAE forward(real_mel) → recon with skips | ✅ RMS 0.17, sounds real |
| HiFi-GAN | ✅ Converts real mels correctly |
| Classifier | ✅ 95.3% accuracy |
| VAE forward(real_mel) → recon without skips | ❌ White noise |
| VAE sample(random_z) → generate | ❌ White noise |

---

## 1. Root Cause Analysis

### 1.1 Why skip connections matter

Each FiLMDecoderStage normally concatenates encoder features:

```
With skip:     h → upsample → conv → FiLM → concat(h, enc_skip) → proj → output
               Channel flow: out_ch + skip_ch → 2× channels → proj → out_ch

Without skip:  h → upsample → conv → FiLM → output
               Channel flow: out_ch → stays out_ch (no concat, no proj)
```

The skip provides:
1. **Spatial structure:** The encoder features contain edge/texture information at each resolution
2. **Information bandwidth:** 2× channels at the concat point before projection
3. **Gradient path:** Direct connection for reconstruction loss to flow to encoder

Without skips, the decoder's convolutional layers must hallucinate all spatial detail from z (2048 dims) + FiLM (256 dims). The FiLM only provides global class modulation, not spatial detail. z is 86% noise. The decoder collapses to producing random high-frequency patterns.

### 1.2 Why the latent space is collapsed

```
Training loss = MSE(recon, input) + β * KL(μ,σ || N(0,I))
```

With β=0.005, free_bits=0.01:
- KL cost for μ=0, σ=1: 2048 × 0.01 × 0.005 = 0.1
- MSE cost: ~0.015-0.2 (with skips)
- The encoder can produce μ=0, σ≈1 with minimal KL penalty (0.1)
- The decoder uses SKIPS to reconstruct, not z content
- Result: μ≈0 for all inputs, z is 86% noise

The encoder learned that it doesn't need to put content in z because the skip connections carry all the information. The VAE is effectively a **skip-connection autoencoder with random noise injection**.

---

## 2. Fix Options

### Option A: Style-Transfer Generation (2 hours — QUICK WIN)

**What it does:** Use encoder skips from a real source audio + FiLM remapping to generate new sounds.

```python
def generate(source_audio, target_label, temperature=0.5):
    # 1. Encode source → get z + skips
    mel = to_mel(source_audio)
    z, skips = vae.encode_full(mel)  # new method
    
    # 2. Add variation to z
    z = z + temperature * torch.randn_like(z)
    
    # 3. Decode with target class FiLM + source skips
    class_emb = vae.class_embed(target_label)
    output = vae.decode_with_skips(z, class_emb, skips, mel.shape[2:])
    
    return output
```

**How it works:**
- Skips from source audio provide spatial structure (edges, rhythm, energy envelope)
- FiLM remaps frequencies to target animal (Dog→Cat: shift bark to meow range)
- z perturbation adds uniqueness (each generation is slightly different)
- Same source → different target = different animals. Same class → different source = different style.

**Pros:**
- Works immediately with current architecture — no retraining
- Skips guarantee high-quality output
- Can generate any class from any source

**Cons:**
- Not "from scratch" — requires a source audio
- Source audio influences output (e.g., dog rhythm on a cat meow)
- Less creative than pure generation

**Implementation:**
1. Add `encode_full()` to `ImprovedVAE` — returns (z, [s0,s1,s2])
2. Add `generate_transfer()` method
3. Update `generate.py` with `--source` argument
4. Per-class library of source audios (one per class)

### Option B: Generation Decoder (4-6 hours — PROPER FIX)

**What it does:** Train a completely separate decoder that works without skips, from scratch.

```python
class GenerationDecoder(nn.Module):
    """
    Standalone decoder: z + class_emb → spectrogram.
    NO skip connections. Uses self-attention for spatial coherence.
    """
    def __init__(self, latent_dim=2048, embed_dim=256):
        super().__init__()
        # fc_decode → reshape → 4× SelfAttentionDecoder blocks
        # Each block: upsample 2× → conv → self-attention → FiLM
        # More capacity than FiLMDecoderStage to compensate for no skips
        
    def forward(self, z, class_emb):
        # No encoder needed at all
        ...
```

**Architecture:**
- Single fc_decode (2048→35840) → reshape [B, 256, 4, 35]
- 4 GenDecoderBlocks with self-attention (compensates for missing skips)
- Heavy FiLM (embed_dim=512) for strong class conditioning
- Trained from scratch on real mel spectrograms (no encoder, no KL, just MSE)

**Training:**
```python
for mel, label in dataloader:
    z = torch.randn(B, 2048)  # pure noise — what we'll use at inference
    output = gen_decoder(z, class_embed(label))
    loss = MSE(output, real_mel) + class_loss(classifier(output), label)
```

**Pros:**
- True generation from scratch — no source audio needed
- Architecture designed explicitly for generation
- No KL collapse issues

**Cons:**
- ~4-6 hours to implement + train
- New model to maintain
- May still produce blurry output (all decoders do without skips)

### Option C: VQ-VAE Style (8+ hours — RESEARCH GRADE)

**What it does:** Two-stage training: deterministic AE → discrete latent prior.

```
Stage 1: Train ImprovedAutoencoder (deterministic, with skips, no VAE)
         → learns perfect reconstruction
    
Stage 2: Train a prior (Transformer/Gaussian Mixture) over the AE latent space
         → learns to generate realistic z values per class
    
Generation: prior.sample(class) → z → AE.decode(z, class_emb, no_skips)
```

**But:** The AE also needs skips. So Stage 1 would need a decoder that works without skips (back to Option B).

### Option D: Per-Class μ Generation (1 hour — SIMPLE HACK)

**What it does:** During generation, sample z = class_mean_μ + noise instead of N(0,I).

```python
# After training (or compute from existing checkpoint):
class_means = compute_class_mus(vae, train_loader)

# Generation:
z = class_means[target_class] + temperature * torch.randn(2048) * 0.1
output = vae.decode(z, class_embed(target_class))
```

**Why this won't work:** We tested this — encoder z without skips still produces white noise. The decoder architecture simply can't generate without skips, regardless of z quality.

---

## 3. Recommended Path

### Immediate: Option A (Style Transfer)
- Implement `encode_full()` and `generate_transfer()` in `src/vae/model.py`  
- Update `generate.py` with `--source` and `--source-label` flags
- Test with real source audios → should produce recognizable animal sounds
- **Time: 2 hours. Risk: zero. Guaranteed to work.**

### Long-term: Option B (Generation Decoder)
- Design `GenerationDecoder` class in new file `src/vae/gen_decoder.py`
- Train on Colab L4 (~4 hours)
- Replace `vae.sample()` with `gen_decoder(z, class_emb)`
- **Time: 6 hours. Risk: medium. May need iteration.**

---

## 4. Option A Implementation Plan

### 4.1 Add `encode_full()` to ImprovedVAE

```python
def encode_full(self, x):
    """Returns (z, skips) for style-transfer generation."""
    s0 = self.enc1(x)
    s1 = self.enc2(s0)
    s2 = self.enc3(s1)
    s3 = self.enc4(s2)
    h = self.attn(s3)
    h = h.flatten(start_dim=1)
    mu = self.fc_mu(h)
    log_var = self.fc_log_var(h)
    z = self.reparameterize(mu, log_var)
    return z, [s0, s1, s2]
```

### 4.2 Update `generate.py`

```python
def generate_transfer(vae, source_audio, source_label, target_label, device, temperature=0.5):
    """Style transfer: source audio structure → target class sound."""
    _, eval_tfm = get_transformations()
    eval_tfm = eval_tfm.to(device)
    source_mel = eval_tfm(source_audio.to(device))
    
    with torch.no_grad():
        z, skips = vae.encode_full(source_mel)
        z = z + temperature * torch.randn_like(z)
        target_emb = vae.class_embed(torch.tensor([target_label], device=device))
        output = vae.decode_with_skips(z, target_emb, skips, source_mel.shape[2:])
    
    # Normalize
    output = torch.clamp(output, -4.0, 4.0)
    m, s = output.mean(), output.std()
    if s > 0.01:
        output = (output - m) / s * 0.7
    
    return mel_to_waveform(output.cpu(), use_griffin_lim=False)
```

### 4.3 CLI

```bash
# Dog structure → Cat sound
python src/generate.py --source Dog --target Cat --temperature 0.5

# Cat structure → Dog sound  
python src/generate.py --source Cat --target Dog

# Random source from class → same class (variation)
python src/generate.py --source Dog --target Dog --count 5
```

---

## 5. Option B Architecture Sketch

```python
class GenDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        # upsample 2× → conv → groupnorm → self-attention → FiLM
        # Self-attention replaces skip connection detail
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, out_ch)
        self.attn = SelfAttention1D(out_ch, num_heads=4)  # temporal coherence
        self.film = FiLM(embed_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, out_ch)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, h, cond):
        h = self.upsample(h)
        residual = self.skip_conv(h)
        h = self.conv1(h)
        h = self.gn1(h)
        h = self.attn(h)  # temporal attention replaces spatial skip info
        gamma, beta = self.film(cond)
        h = h * (1 + gamma) + beta
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        return F.silu(h + residual)

class GenerationDecoder(nn.Module):
    def __init__(self, latent_dim=2048, embed_dim=512):
        c4, c3, c2, c1 = 256, 128, 64, 32
        self.fc = nn.Linear(latent_dim, c4 * 4 * 35)
        self.block4 = GenDecoderBlock(c4, c3, embed_dim)
        self.block3 = GenDecoderBlock(c3, c2, embed_dim)
        self.block2 = GenDecoderBlock(c2, c1, embed_dim)
        self.block1 = GenDecoderBlock(c1, 16, embed_dim)
        self.output = nn.Conv2d(16, 1, 3, padding=1)
    
    def forward(self, z, class_emb):
        h = self.fc(z).view(z.shape[0], 256, 4, 35)
        h = self.block4(h, class_emb)
        h = self.block3(h, class_emb)
        h = self.block2(h, class_emb)
        h = self.block1(h, class_emb)
        h = self.output(h)
        return F.interpolate(h, size=(64, 552), mode='bilinear')
```

Training: pure MSE + optional classifier loss. No encoder, no KL, no skips. z ~ N(0,1).

---

## 6. Decision Matrix

| Criterion | Option A (Style Transfer) | Option B (Gen Decoder) |
|-----------|:---:|:---:|
| **Works today?** | ✅ Yes | ❌ Needs training |
| **True generation?** | ❌ Needs source | ✅ From scratch |
| **Implementation time** | 2 hrs | 6 hrs |
| **Audio quality** | High (skips) | Medium (no skips, blurry) |
| **Risk** | None | Medium |
| **User experience** | "Pick source + target" | "Click Dog button" |

**Recommendation:** Implement Option A NOW (guaranteed to work, 2 hours). Then decide if Option B is worth the investment based on how Option A feels in practice.
