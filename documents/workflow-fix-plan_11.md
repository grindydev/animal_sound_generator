# Workflow Fix Plan v11 — More Clean Data + 150 Epochs

> **Date:** May 16, 2026  
> **Status:** Implementing.  
> **Builds on:** v10 (ESC-50 — 4/8 audible, first class-specific frequencies)  
> **Goal:** 200+ clean files per class. 150 epochs. All 8 classes distinct.

---

## 1. v10 Achievement

```
OLD data:   0/8 audible, all 32Hz hum, all classes identical
ESC-50:     4/8 audible, Dog=345Hz, Frog=1895Hz, Rooster=1034Hz
```

Proved: **architecture works. More data = more classes audible.**

## 2. Data Sources

| Source | What | Files | Download |
|--------|------|:---:|------|
| ESC-50 (have) | Clean 5s clips | 640 | ✅ Already in `data/esc50/` |
| Old data, energy-filtered | Smart-crop noise removal | ~500 clean segments | ✅ Already in `data/animal_audio/` |
| **Combined** | | **~1,000+** | Merge + deduplicate |

## 3. Energy-Based Filtering

The old data has real animal sounds buried in silence. We extract only high-energy segments:

```python
def extract_clean_segments(wav_path, min_rms=0.02):
    """Keep only 5s chunks with RMS > 0.02 (actual sound, not silence)."""
    wav = load(wav_path)
    for start in range(0, len(wav)-5s, 2s):  # 2s stride for overlap
        chunk = wav[start:start+5s]
        if chunk.rms > min_rms:
            yield chunk
```

This filters ~3001 old files down to ~500 clean 5s segments.

## 4. Combined Dataset

| Class | ESC-50 | Old (filtered) | Total |
|-------|:---:|:---:|:---:|
| Dog | 40 | ~150 | ~190 |
| Cat | 40 | ~80 | ~120 |
| Rooster | 40 | ~40 | ~80 |
| Frog | 40 | ~20 | ~60 |
| Crow | 80 | ~15 | ~95 |
| Insect | 80 | ~50 | ~130 |
| Hen | 40 | ~30 | ~70 |
| Noise | 280 | 0 | 280 |
| **Total** | 640 | ~385 | **~1,025** |

## 5. Changes

| File | Change |
|------|--------|
| `src/scripts/build_dataset.py` | New: merge ESC-50 + filtered old data |
| `src/diffusion/config.py` | `data_dir="data/combined"`, `num_epochs=150` |
| (nothing else) | Same UNet, same GAN, same training loop |

## 6. Expected Outcome

| Metric | v10 (ESC-50, 50 epochs) | v11 Target |
|--------|:---:|:---:|
| Audible classes | 4/8 | 6-8/8 |
| Dog peak | 345 Hz | 300-800 Hz (stable bark) |
| Frog peak | 1895 Hz | 1500-2500 Hz (croak) |
| Cat peak | 172 Hz | 200-2000 Hz (meow) |
| Training time | 40 min | 60 min (150 epochs, ~1,000 files) |
