# Audio Generation Workflow

> A repeatable workflow for building audio generation models. Follow this step-by-step. Each step is a gate — don't proceed until the metric passes.

---

## The Big Picture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Phase 1    │────▶│  Phase 2    │────▶│  Phase 3    │────▶│  Phase 4    │
│ Understand  │     │ Classifier  │     │ Autoencoder │     │ VAE         │
│ Audio Data  │     │ (Evaluator) │     │ (Reconstruct│     │ (Generate   │
│             │     │             │     │  by class)  │     │  by class)  │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
                                                      ┌────────────▼────────────┐
                                                      │  Evaluate (Step 6)      │
                                                      │  Classification Agree-  │
                                                      │  ment > 50%?            │
                                                      └────────────┬────────────┘
                                                                   │
                                              ┌────────────────────┼────────────────────┐
                                              │                    │                    │
                                              ▼                    ▼                    ▼
                                     ┌─────────────┐       ┌─────────────┐      ┌─────────────┐
                                     │ < 30%       │       │ 30-50%      │      │ > 50% ✅    │
                                     │ Broken      │       │ Partial     │      │ Ready for   │
                                     │ Fix mode    │       │ Improve β   │      │ Phase 7     │
                                     │ collapse    │       │ arch, loss  │      │ HiFi-GAN    │
                                     └─────────────┘       └─────────────┘      └──────┬──────┘
                                                                                        │
                                                                             ┌──────────▼──────────┐
                                                                             │  Phase 7a           │
                                                                             │  HiFi-GAN Vocoder   │
                                                                             │  (crisp audio)      │
                                                                             └──────────┬──────────┘
                                                                                        │
                                                                             ┌──────────▼──────────┐
                                                                             │  Phase 7b-7d        │
                                                                             │  Diffusion, Long    │
                                                                             │  sounds, Mixing     │
                                                                             └─────────────────────┘
```

---

## Step-by-Step

### Step 1 — Understand Your Audio Data

**Goal:** Load audio, compute spectrograms, build a dataset.

```
Build:
  data_loader.py          # Dataset, collate_fn, transforms
  download_data.py         # Fetch dataset

Output:
  DataLoader yielding: (waveform, label)
  Mel spectrogram: [B, n_mels, T]

Check:
  ✅ Can load all files?
  ✅ Spectrogram shape correct?
  ✅ All classes present?
  ✅ Min/max values look reasonable?
```

### Step 2 — Build a Classifier (Your Future Evaluator)

**Goal:** Train a CNN that classifies audio. This becomes your quality metric later.

```
Build:
  model.py                 # CNN classifier
  train.py                 # Training loop
  evaluate.py              # Confusion matrix, F1

Output:
  best_audio_cnn_train.pth  ← Save this, you'll need it in Step 6

Check:
  ✅ Test accuracy > 80%? (if not, your data or classifier is broken)
  ✅ Per-class accuracy reasonable?
```

**Why this first?** You need a reliable evaluator BEFORE building a generator. The classifier answers: "Does this generated audio actually sound like a dog?"

### Step 3 — Autoencoder (Learn to Reconstruct)

**Goal:** Compress → reconstruct. Prove you can rebuild audio from a latent representation.

```
Build:
  autoencoder.py           # Encoder + Decoder + Griffin-Lim

Pipeline:
  waveform → mel → encode → latent → decode → mel → Griffin-Lim → audio

Output:
  best_autoencoder_train.pth

Check:
  ✅ Reconstruction MSE dropping?
  ✅ Listen to output — does it sound like the input? (Griffin-Lim will be grainy — that's fine)
```

### Step 4 — Conditional VAE (Generate by Class)

**Goal:** Generate audio by specifying a class label.

```
Build:
  vae.py                   # Encoder → μ,σ → sample → Decoder
  train_vae.py             # Training with KL + reconstruction loss
  finetune_vae.py          # Optional: fine-tune from autoencoder weights

Pipeline:
  random_noise + class_label → VAE decoder → mel spectrogram
  mel spectrogram → Griffin-Lim → audio

Output:
  best_vae_scratch_train.pth
  best_vae_finetune_train.pth

Check:
  ✅ Can generate "dog" and "cat" and they sound different?
  ✅ Same class → different sounds each time? (diversity)
```

### Step 5 — Fix Common VAE Problems

**Most likely problem: Mode Collapse**

```
Symptom:
  Every class generates the same "generic animal" sound

Root causes & fixes:
  ┌──────────────────┬─────────────────────────────┬──────────────────────────┐
  │ Problem          │ Fix                         │ File to change           │
  ├──────────────────┼─────────────────────────────┼──────────────────────────┤
  │ Class signal     │ z = z + class_emb           │ z = cat([z, class_emb])  │
  │ diluted by KL    │ → class disappears          │ → dedicated channels     │
  │                  │                             │ vae.py                   │
  ├──────────────────┼─────────────────────────────┼──────────────────────────┤
  │ β too high       │ KL pushes everything to 0   │ Lower β: 0.005→0.002     │
  │ (aggressive KL)  │ → all classes collapse      │ train_vae.py             │
  ├──────────────────┼─────────────────────────────┼──────────────────────────┤
  │ No class signal  │ MSE doesn't care about class│ Add γ × classifier loss  │
  │ in loss          │ → "average animal" is safest│ train_vae.py             │
  └──────────────────┴─────────────────────────────┴──────────────────────────┘
```

### Step 6 — Evaluate (The Go/No-Go Gate)

**Goal:** Quantify generation quality. This decides whether you proceed to HiFi-GAN.

```
Run:
  python src/evaluate_gen.py

Metrics:
  ┌──────────────────────────────┬─────────────┬──────────────────────────┐
  │ Metric                       │ What it     │ Target                   │
  │                              │ measures    │                          │
  ├──────────────────────────────┼─────────────┼──────────────────────────┤
  │ Classification Agreement     │ Does gen-   │ > 50% average            │
  │ (THE KEY METRIC)             │ erated audio│                          │
  │                              │ match the   │ Per-class: each > 20%    │
  │                              │ intended    │                          │
  │                              │ class?      │                          │
  ├──────────────────────────────┼─────────────┼──────────────────────────┤
  │ Reconstruction MSE           │ How well    │ Lower = better           │
  │                              │ does it     │ Typical: 0.05-0.15       │
  │                              │ reconstruct │                          │
  ├──────────────────────────────┼─────────────┼──────────────────────────┤
  │ Diversity Score              │ Same class, │ Higher = better          │
  │                              │ different   │ Typical: 50-100          │
  │                              │ outputs?    │                          │
  ├──────────────────────────────┼─────────────┼──────────────────────────┤
  │ t-SNE visualization          │ Do real and │ Clusters should overlap  │
  │                              │ generated   │ per class                │
  │                              │ occupy same │                          │
  │                              │ space?      │                          │
  └──────────────────────────────┴─────────────┴──────────────────────────┘
```

**Decision tree:**

```
Classification Agreement Average:
  │
  ├─ < 30% → STOP. VAE is broken (mode collapse).
  │           Fix architecture, β, add class loss. Go back to Step 4.
  │           HiFi-GAN will NOT fix this.
  │
  ├─ 30-50% → PARTIAL. VAE works but quality is low.
  │            Try: lower β, more epochs, better architecture.
  │            Can try HiFi-GAN but results will be mediocre.
  │
  └─ > 50% → ✅ GOOD. VAE produces correct content.
              PROCEED to Step 7 (HiFi-GAN).
```

**Golden Rule:**
> HiFi-GAN fixes HOW audio sounds (grainy → crisp).
> It cannot fix WHAT audio sounds like (dog → not dog).
> **Always verify content BEFORE training the vocoder.**

### Step 7 — HiFi-GAN Neural Vocoder

**Goal:** Replace Griffin-Lim with a neural network that produces crisp audio.

```
Build:
  src/hifigan/
    config.py              # All hyperparameters
    generator.py           # MRF architecture
    discriminator.py       # MSD + MPD
    losses.py              # mel + feature matching + adversarial
    train.py               # Training loop

Training strategy:
  Phase 7a-1: Mel-only pretraining (mode="meltrain")
    → Generator learns mel → audio mapping
    → No discriminator, stable training
    → 15-30 epochs, watch mel loss drop

  Phase 7a-2: Full GAN training (mode="train")
    → Generator + discriminator train together
    → 20-30 epochs, mel loss keeps dropping
    → Discriminator loss hovers around 1-2

Critical hyperparameters:
  ┌──────────────────┬─────────┬─────────────────────────────────────┐
  │ Parameter        │ Value   │ Why                                 │
  ├──────────────────┼─────────┼─────────────────────────────────────┤
  │ segment_size     │ 16384   │ 0.74s — long enough for full bark   │
  │                  │         │ (8192 = 0.37s too short)            │
  ├──────────────────┼─────────┼─────────────────────────────────────┤
  │ lambda_mel       │ 45      │ ANCHOR — forces generator to match  │
  │                  │         │ input mel. Without this, generator  │
  │                  │         │ drifts to "realistic but wrong"     │
  │                  │         │ (e.g., human speech instead of dog) │
  ├──────────────────┼─────────┼─────────────────────────────────────┤
  │ lambda_adv       │ 1       │ Small polish. Must NOT overpower    │
  │                  │         │ mel loss. 2+ → content drift.       │
  ├──────────────────┼─────────┼─────────────────────────────────────┤
  │ lambda_fm        │ 2       │ Feature matching guides realism     │
  ├──────────────────┼─────────┼─────────────────────────────────────┤
  │ num_workers      │ 0       │ Avoid multiprocessing bugs.         │
  │                  │         │ Data loading is fast enough.        │
  └──────────────────┴─────────┴─────────────────────────────────────┘

Check:
  ✅ Mel loss drops from ~6 → ~1 (mel-only phase)
  ✅ GAN mel loss keeps dropping (full GAN phase)
  ✅ D loss hovers 1-2 (not 0, not exploding)
  ✅ Listen to comparison files — GAN should sound crisper
```

### Step 8 — Final Pipeline

```
"dog" ──▶ VAE ──▶ [mel spectrogram] ──▶ HiFi-GAN ──▶ [crisp audio ✨]

The complete quality chain:
  Griffin-Lim baseline  → 01_griffinlim.wav  (grainy, baseline)
  HiFi-GAN meltrain     → 03_meltrain.wav    (better, no GAN)
  HiFi-GAN full GAN     → 04_gantrain.wav    (best, crisp)
```

---

## Checklist for a New Audio Generation Project

Print this and check off each item:

```
Phase 1: Data
  [ ] Can load audio files
  [ ] Mel spectrogram computation works
  [ ] Train/val/test split created
  [ ] All classes present and balanced

Phase 2: Classifier (evaluator)
  [ ] Classifier trained and saved
  [ ] Test accuracy > 80%
  [ ] Confusion matrix reviewed

Phase 3: Autoencoder
  [ ] Encoder compresses correctly
  [ ] Decoder reconstructs from latent
  [ ] Griffin-Lim converts mel → audio
  [ ] Can hear: reconstruction sounds like input

Phase 4: Conditional VAE
  [ ] Can generate by class label
  [ ] Same class → diverse outputs
  [ ] Different classes → different sounds

Phase 5: Debug (if needed)
  [ ] Mode collapse fixed?
  [ ] β tuned (not too high, not too low)?
  [ ] Class conditioning works?
  [ ] Classification loss added?

Phase 6: Evaluate (GATE)
  [ ] Classification agreement > 50%?
      NO → fix VAE, don't proceed
      YES → proceed to HiFi-GAN
  [ ] Diversity score reasonable?
  [ ] t-SNE shows class separation?
  [ ] Spectrograms look correct?

Phase 7: HiFi-GAN
  [ ] Mel-only pretraining converges?
  [ ] Full GAN training stable?
  [ ] lambda_mel=45 (NOT 1.0!)
  [ ] segment_size=16384 (NOT 8192!)
  [ ] num_workers=0 (avoid bugs)
  [ ] Comparison audio sounds better?

Phase 8: Polish (optional)
  [ ] Diffusion for sharper mels?
  [ ] Longer sequential generation?
  [ ] Latent space mixing?
  [ ] Web app deployed?
```

---

## Common Mistakes and How to Avoid Them

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Skip classifier training | No way to evaluate generation quality | Always build classifier FIRST (Step 2) |
| Skip evaluation | Train HiFi-GAN on broken VAE | Always run evaluate_gen.py (Step 6) before HiFi-GAN |
| lambda_mel too low | GAN generates speech instead of animal sounds | Use lambda_mel=45, not 1.0 |
| segment_size too small | Generator can't capture full bark/meow | Use 16384 (0.74s), not 8192 (0.37s) |
| num_workers > 0 | Silent data loading bugs, loss=0.0000 | Use num_workers=0 for audio data |
| β too high in VAE | Mode collapse — all classes same sound | Start with β=0.002, not 0.01 |
| Class signal via addition | Class disappears when KL pushes μ→0 | Use concatenation, not addition |
| HiFi-GAN before VAE is ready | Crisp audio of wrong content | Content first (VAE), quality second (HiFi-GAN) |

---

## File Mapping

For this project, here's where each step lives:

```
Step 1:  src/data_loader.py
Step 2:  src/model.py, src/train.py, src/evaluate.py
Step 3:  src/autoencoder.py, src/train_autoencoder.py
Step 4:  src/vae.py, src/train_vae.py, src/finetune_vae.py
Step 5:  src/vae.py (fix mode collapse), src/train_vae.py (fix β)
Step 6:  src/evaluate_gen.py          ← THE GATE
Step 7:  src/hifigan/                 ← Neural vocoder
Step 8:  src/diffusion_refine.py, src/sequential_generator.py, src/latent_mixing.py
```
