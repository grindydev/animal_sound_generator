# Colab Training Guide — Animal Sound Generator

> Move training from GTX 1650 (4GB) → Google Colab T4 (16GB)

---

## Why Colab?

| | GTX 1650 | Colab T4 | Improvement |
|---|----------|----------|-------------|
| VRAM | 4 GB | 16 GB | **4×** |
| Autoencoder batch | 2 | 8 | **4× faster epochs** |
| VAE batch | 1 + accum | 4 | **4× faster** |
| base_channels | 16 (37M params) | **32 (149M params)** | **Better model** |
| Autoencoder time | ~3 hrs | ~2 hrs | 33% less |
| Free tier limit | — | ~4-6 hrs/day | Manageable |

---

## Step-by-Step Setup

### 1. Upload Dataset to Google Drive

Your `data/animal_audio/` folder (3,001 .wav files, ~3GB):

```bash
# On your local machine, zip and upload:
tar -czf animal_audio.tar.gz data/animal_audio/
# Upload animal_audio.tar.gz to https://drive.google.com → MyDrive/
```

Or use the download script directly in Colab (slower but no upload needed).

### 2. Upload Existing Checkpoints (Optional)

If you want to resume training from where you left off:

```bash
# Upload your existing models:
# models/best_audio_cnn_train.pth        (1.8 MB - classifier)
# models/best_autoencoder_train.pth       (596 MB - autoencoder)
# models/autoencoder_checkpoints/         (epoch checkpoints for resume)
```

### 3. Push Code to GitHub

Colab needs to clone your repo:

```bash
cd /Users/thanhbm/Projects/animal_sound_generator
git add colab_train.ipynb colab_patch_configs.py
git commit -m "Add Colab training notebook and config patcher"
git push
```

### 4. Open Colab Notebook

- Go to https://colab.research.google.com
- File → Upload notebook → select `colab_train.ipynb`
- **Runtime → Change runtime type → T4 GPU**
- Run cells in order

### 5. Or: Use the "Open in Colab" Link

Add a badge to your README:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/YOUR_USERNAME/animal_sound_generator/blob/main/colab_train.ipynb
)
```

---

## Training Schedule (Colab T4)

Run one model per Colab session (Colab disconnects after 4-6 hours):

### Session 1: Autoencoder (~2-3 hrs)
```
1. Mount Drive, clone repo, pip install
2. Copy data from Drive
3. Run colab_patch_configs.py
4. python src/vae/train_ae.py
5. Save best_autoencoder_train.pth → Drive
```

### Session 2: VAE Fine-tune (~2-3 hrs)
```
1. Mount Drive, clone repo, pip install
2. Copy data + best_autoencoder_train.pth from Drive
3. python src/vae/finetune.py
4. Save best_vae_finetune_train.pth → Drive
```

### Session 3: Diffusion (~3 hrs)
```
1. Mount Drive, clone repo
2. Copy data + best_vae_finetune_train.pth from Drive
3. python src/diffusion/train.py
4. Save diffusion_unet_train_best.pth → Drive
```

---

## Important: Colab Limitations & Workarounds

| Issue | Solution |
|-------|----------|
| **Session times out after 90 min idle** | Click the page occasionally, or use a keep-alive script |
| **Runtime disconnects after ~6 hrs** | Save checkpoints to Drive EVERY epoch (already done!) |
| **Drive I/O is slow** | Copy data to `/content/` at start (local SSD), sync back at end |
| **Free GPU limit** | Use Colab Pro ($10/mo) for priority access + longer sessions |
| **Runtime resets** | Each session = fresh VM. Always restore from Drive checkpoints |
| **Pro: A100 GPU** | Colab Pro+ ($50/mo) = sometimes A100 (40GB). For 149M models, T4 is enough |

---

## Recovery: Resume After Timeout

Your training scripts already save checkpoints EVERY epoch. When Colab times out:

1. Start a new Colab session
2. Run cells 1-2b (setup + copy models from Drive)
3. Run the training cell again — it auto-resumes from the latest checkpoint

The checkpoint resume is built into all training scripts via `load_checkpoint()`.

---

## Post-Training: Sync Back to Local

### Option A: Google Drive Desktop (easiest)
Install Google Drive for desktop → models appear in your local filesystem

### Option B: Download from drive.google.com
Navigate to `MyDrive/animal_sound_generator/models/` → right-click → Download

### Option C: rclone (power users)
```bash
rclone copy gdrive:animal_sound_generator/models/ ./models/ -P
```

### Option D: zip + direct download
In Colab:
```python
!tar -czf models.tar.gz models/*.pth
from google.colab import files
files.download('models.tar.gz')
```

---

## What to Update Locally After Training

Once you have the new models, update your local config:

```python
# In src/generate.py or wherever you load models:
# Point to the new Colab-trained checkpoints
VAE_CHECKPOINT = "models/best_vae_finetune_train.pth"          # 223M params (ch=32)
AE_CHECKPOINT = "models/best_autoencoder_train.pth"            # 149M params (ch=32)
DIFFUSION_CHECKPOINT = "models/diffusion_unet_train_best.pth"  # 18M params
HIFIGAN_CHECKPOINT = "models/hifigan_generator_train.pth"     # 3.3M params
```

Update base_channels in config to match:
```python
CONFIG["base_channels"] = 32  # must match the Colab-trained model
```

---

## Cost Comparison

| | GTX 1650 (electricity) | Colab Free | Colab Pro |
|---|------|-----------|-----------|
| Cost | ~$0.50/day | **$0** | $10/mo |
| Training time (full) | ~13 hrs | ~10 hrs | ~8 hrs |
| VRAM | 4 GB | 16 GB | 16-40 GB |
| Can multitask? | GPU locked | Browser tab | Browser tab |
| Uptime limit | Unlimited | ~6 hrs | ~24 hrs |
| **Recommendation** | - | ✅ **Use this** | Optional upgrade |
