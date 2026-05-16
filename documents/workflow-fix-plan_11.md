# Workflow Fix Plan v11 (revised) — Better Data Sources

> **Date:** May 16, 2026  
> **Status:** Implementing.  
> **Builds on:** v10 (ESC-50 — 4/8 audible). Old data removed — too noisy.  
> **Goal:** 150-300 clean files per class from multiple curated sources.

---

## 1. Data Sources (all free, direct download)

| Dataset | Best For | Files | Download |
|---------|----------|:---:|------|
| **ESC-50** (have) | All 8 classes | 640 | Already in `data/esc50/` |
| **UrbanSound8K** | Dog barks | ~1,000 | `wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz` |
| **Cat Sound Dataset** | Cat meows | ~160 | Kaggle or `wget https://storage.googleapis.com/...` |
| **Xeno-Canto** | Bird calls (Crow, Hen, Rooster) | ~200 | API: `xeno-canto.org/api/2/recordings` |
| **Freesound** | Frog, Insect | ~100 each | Search + batch download |

**Combined: ~2,000+ clean files. All classes with 100-300 unique sounds.**

## 2. Implementation

### Phase 1: Dog (UrbanSound8K)
```bash
wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
tar -xzf UrbanSound8K.tar.gz
# Extract "dog_bark" class (label 4) → data/sources/urbansound/Dog/
```

### Phase 2: Cat (Freesound + Kaggle)
Cat meowing samples available from:
- Freesound search "cat meow" → ~200 results  
- Kaggle "cat sound dataset"

### Phase 3: Birds (Xeno-Canto API)
```python
# Rooster: query="Gallus gallus" type="song"
# Crow: query="Corvus" type="call"
# Hen: query="chicken" type="call"
# ~50-100 recordings each, free download
```

### Phase 4: Frog, Insect (Freesound)
```python
# Frog: search "frog croak" → ~150 results
# Insect: search "cricket" OR "insect buzz" → ~200 results
```

## 3. Changes

| File | Change |
|------|--------|
| `src/scripts/build_dataset.py` | Add UrbanSound8K + Xeno-Canto + Freesound sources |
| `src/diffusion/config.py` | `data_dir="data/combined"` (kept) |
| Notebook | Download additional datasets |

## 4. Plan

Start with UrbanSound8K (easiest — direct download, clean). That alone gives Dog 40→1,040 files.

```bash
# Download UrbanSound8K
wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
tar -xzf UrbanSound8K.tar.gz
python src/scripts/import_urbansound.py  # extract dog barks
```

Then add the other sources one by one. Each new source = more class diversity.
