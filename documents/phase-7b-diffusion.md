# Phase 7b — Diffusion Refinement

> **What it does:** A small U-Net sharpens VAE-generated blurry spectrograms before HiFi-GAN converts them to audio.

---

## 1. The Problem

Your VAE generates mel spectrograms, but they're blurry — VAEs average over possibilities, so edges get smoothed out. HiFi-GAN converts whatever mel you give it, blur included.

Diffusion is a **refiner**: it takes the blurry VAE output and sharpens it.

```
"dog" → VAE → blurry mel → Diffusion → sharp mel → HiFi-GAN → crisp audio ✨
```

---

## 2. How Diffusion Works (Two Processes)

### Training: Add noise, learn to remove it

Take a real spectrogram. Add random noise. Ask the U-Net: "where's the noise?" Grade the answer. Repeat millions of times.

```
Real mel (clean) + noise → Noisy mel
                              ↓
                         U-Net guesses noise
                              ↓
                      Compare guess to real noise
                              ↓
                        Adjust model weights
```

### Inference: Start noisy, clean it up

Start with the VAE output + some noise added. Run the U-Net repeatedly, each time removing a little predicted noise. After ~50 steps, you get a sharp spectrogram.

```
VAE mel + noise → [U-Net cleans a bit] → [U-Net cleans more] → ... → sharp mel
```

---

## 3. The Critical Thing: Why t Matters

Here's the problem. The U-Net receives a noisy spectrogram and must predict what noise was added. But:

- A clean sound + heavy noise (t=900) can look **identical** to
- An already-noisy sound + light noise (t=100)

Same input, different correct answer. **The U-Net can't know which case it is just by looking.**

**t solves this.** t is a number (0 to 999) saying "I added noise to level 347 out of 1000." Now the model knows exactly how much noise to predict.

```
Without t:  U-Net sees noisy image → has to guess → often wrong
With t:     U-Net sees noisy image + "t=347" → predicts 35% noise → correct
```

---

## 4. How t Works Inside the Model (Code Trace)

### Step 1: t is just an integer

```python
# train.py, inside train_epoch():
t = torch.randint(0, 1000, (B,))   # e.g., [347, 891, 12]
```

### Step 2: t gets converted to a rich pattern (Clock Hands)

```python
# unet.py, SinusoidalTimeEmbedding.forward():
# t = [347]
# → 128 "clock hands" spin to positions based on 347
# → sin + cos = 256 numbers
# → small neural network = 1024 numbers
t_emb = self.time_embed(t)     # [B] → [B, 1024]
```

### Step 3: t never touches the spectrogram directly

t goes through a **separate path** and becomes **knobs** inside every ResBlock:

```python
# unet.py, ResBlock.forward():
h = self.conv1(x)                        # process the spectrogram features

t_knobs = self.time_mlp(t_emb)           # [B, 1024] → [B, channels*2]
t_scale, t_shift = t_knobs.chunk(2)      # one volume knob + one bias per channel

h = h * (1 + t_scale) + t_shift          # ← t touches features HERE

# t=10:   knobs near zero → "pass through, tiny noise to predict"
# t=500:  knobs at medium → "moderate transformation"
# t=999:  knobs cranked up → "heavy noise to find"
```

### Step 4: This happens in EVERY ResBlock

The U-Net has ~20 ResBlocks. Every single one gets `t_emb` and produces its own knobs. Shallow layers might amplify edges, deep layers might suppress textures — each learns what's right for its level.

```
t_emb [B, 1024] ──┬──→ ResBlock 0: time_mlp(t_emb) → knobs
                  ├──→ ResBlock 1: time_mlp(t_emb) → knobs  (different MLP!)
                  ├──→ ResBlock 2: time_mlp(t_emb) → knobs
                  ├──→ ...
                  └──→ ResBlock N: time_mlp(t_emb) → knobs
```

### Step 5: The model outputs a noise prediction

```python
# unet.py, SpectrogramUNet.forward():
t_emb = self.time_embed(t)          # [B] → [B, 1024]
c_emb = self.class_embed(labels)    # [B] → [B, 64]  (animal type)

h = self.input_proj(x_t)            # [B, 1, 64, 552] → [B, 64, 64, 552]

# Encoder: down + apply ResBlocks with t_emb
# Bottleneck: apply ResBlocks with t_emb + self-attention
# Decoder: up + skip connections + ResBlocks with t_emb

return self.output_proj(h)          # [B, 64, 64, 552] → [B, 1, 64, 552]
```

Output is the **same shape** as input — a grid of noise values, one per pixel.

---

## 5. The Full Training Picture

```
For each batch:

1. Take real spectrogram x₀   [B, 1, 64, 552]
2. Pick random t              [B], e.g. [347, 12, 891]
3. Add noise at that level    x_t = mix(x₀, noise, t)
4. U-Net(x_t, t, class)       guesses noise
5. Compare guess to real      loss = MSE(pred, real_noise)
6. Update weights             model gets slightly better

Repeat for 50 epochs × hundreds of batches.
The model sees every t value thousands of times.
```

---

## 6. Inference: How VAE Output Gets Sharpened

```python
# inference.py, refine_spectrogram():

# 1. Take VAE output, add partial noise
strength = 0.6                    # how much to refine (0=no change, 1=full regen)
start_t = 0.6 * 999 ≈ 599        # start from 60% noise level
x = vae_mel * √0.4 + noise * √0.6

# 2. Denoise in 50 steps
for step in [599, 587, 575, ..., 12, 0]:
    noise_guess = model(x, step, class_label)
    x = remove_noise(x, step, noise_guess)

# 3. Done: x is now a sharp spectrogram
```

---

## 7. Key Files

| What | File | Key Class/Function |
|------|------|-------------------|
| Config | `src/diffusion/config.py` | `DiffusionConfig` |
| U-Net model | `src/diffusion/unet.py` | `SpectrogramUNet` |
| Time embedding | `src/diffusion/unet.py` | `SinusoidalTimeEmbedding` |
| ResBlock + t injection | `src/diffusion/unet.py` | `ResBlock.forward()` |
| Add/remove noise | `src/diffusion/diffusion.py` | `DiffusionProcess` |
| Training loop | `src/diffusion/train.py` | `train_epoch()` |
| Inference | `src/diffusion/inference.py` | `refine_spectrogram()` |

---

## 8. Quick Config

```python
timesteps = 1000          # total noise levels
inference_steps = 50      # DDIM steps (don't need all 1000)
strength = 0.6            # how much refinement (recommended)
time_emb_dim = 256        # size of time embedding
base_channels = 64        # U-Net starting channels
channel_multipliers = (1, 2, 3, 3)   # ~17.8M params
```
