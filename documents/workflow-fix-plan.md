# Workflow Fix Plan — Broken VAE Generation Pipeline

## Root Causes Found

### 1. Normalization Mismatch (Critical)
```
HiFi-GAN expects:           norm_mean=-18.4903, norm_std=19.8031  (hardcoded in hifigan/config.py)
VAE was trained with:       mean=-30.8645, std=21.1952           (wrong)
Current data_loader uses:   mean=-18.2662, std=19.5589          (close but not exact)
```
HiFi-GAN's `compute_mel()` normalizes with `(db - norm_mean) / norm_std`.  
The VAE must produce outputs in the EXACT SAME normalized space for HiFi-GAN to convert correctly.

**Fix:** Set `SimpleNormalize(mean=-18.4903, std=19.8031)` — exact match with HiFi-GAN.

### 2. Skip-Connection-Only Decoder (Critical)
The VAE decoder was trained EXCLUSIVELY with encoder→decoder skip connections (`decode_with_skips`).  
During generation (`vae.sample()` → `decode()`), NO skips exist. The decoder has never seen this regime.

**Evidence:**
```
Generated spectrogram values:  [-115, 184]   (mean=32, std=36)
Real normalized spectrograms:  [-1.6, 1.5]   (mean=-0.7, std=0.6)
                              → 100× scale mismatch
```

**Fix:** Skip connection dropout during VAE training. Randomly drop ALL skips with probability `skip_dropout`, forcing the decoder to work in both modes. Ramp dropout from 0.0 → 0.5 during β ramp phase.

### 3. Classifier Needs Retraining
The `best_audio_cnn_train.pth` contains interrupted `ImprovedAudioCNN` weights (epoch 1, 64.6% accuracy).  
Must retrain. Use `SimpleAudioCNN` — fast (1.5M params), proven 91% accuracy.

---

## Fix Order (Must Execute in Sequence)

### Step 1: Fix Normalization
**File:** `src/data_loader.py`
```python
# Change SimpleNormalize to exact HiFi-GAN match
class SimpleNormalize(nn.Module):
    def __init__(self, mean = -18.4903, std = 19.8031):
```

### Step 2: Retrain Autoencoder
Delete old checkpoints, retrain with correct normalization.
```bash
rm -rf models/autoencoder_checkpoints/train/ae_*.pth
rm models/best_autoencoder_train.pth
python src/vae/train_ae.py
```
**Expected:** val_mse ~0.004–0.005 (same as before, architecture unchanged)

### Step 3: Add Skip Dropout to VAE + Retrain
**File:** `src/vae/finetune.py` — add to CONFIG:
```python
"skip_dropout": 0.5,         # randomly drop decoder skip connections
"skip_dropout_warmup": 5,    # ramp dropout from 0→0.5 over first N epochs of β ramp
```

**File:** `src/vae/model.py` — modify `forward()`:
```python
def forward(self, x, labels, skip_dropout=0.0):
    # ... encode, get skips ...
    
    # Randomly drop skips during training
    if self.training and skip_dropout > 0 and torch.rand(1).item() < skip_dropout:
        skips = [None, None, None]
    
    reconstructed = self.decode_with_skips(z, class_emb, skips, target_size)
    return reconstructed, mu, log_var
```

**File:** `src/vae/finetune.py` — pass skip_dropout in train_epoch:
```python
def get_skip_dropout(epoch):
    """Ramp skip dropout from 0 → target during β ramp."""
    if epoch < WARMUP_EPOCHS:
        return 0.0
    ramp_epoch = epoch - WARMUP_EPOCHS
    if ramp_epoch >= SKIP_DROPOUT_WARMUP:
        return SKIP_DROPOUT
    return SKIP_DROPOUT * ramp_epoch / SKIP_DROPOUT_WARMUP
```

Delete old VAE checkpoint, retrain:
```bash
rm -f models/vae_checkpoints/train/vae_resume.pth
rm models/best_vae_finetune_train.pth
python src/vae/finetune.py
```
**Expected:** val_mse slightly worse (~0.02–0.03 vs 0.015 before) because decoder now works without skips too. But generation quality will actually work.

### Step 4: Retrain Classifier
```bash
rm models/best_audio_cnn_train.pth
python src/train_classifier.py
```
**Expected:** val_accuracy ~91% (same as original SimpleAudioCNN)

---

## Verification After Fix

### 1. Check classifier accuracy
```bash
python src/evaluate.py
# Should show ~91% val accuracy
```

### 2. Check VAE generation quality
```bash
python src/evaluate_gen.py
# Classification agreement should be >30% (not 15%)
# Diversity score should be <1000 (not 13700 — values now in correct range)
```

### 3. Listen to generated audio
```bash
python src/generate.py --label Dog --no-diff --no-griffin-lim
# Listen to outputs/generated/Dog_*.wav
# Should sound like an actual animal, not noise
```

---

## What NOT to Change

- **HiFi-GAN** — trained on raw audio, normalization is correct in config. Don't touch.
- **Autoencoder architecture** — works fine, just needs normalization fix.
- **VAE architecture** — FiLM conditioning and skip connections are correct. Only need skip dropout.
- **generate.py** — already fixed (base_channels=16, soundfile for saving).
- **evaluate_gen.py** — already fixed (ImprovedVAE, correct split, val_loader).
