"""
build_dataset_v17.py — Download Kaggle animal sounds + Freesound for missing classes

Kaggle dataset: https://www.kaggle.com/datasets/caoofficial/animal-sounds
  - Cat: 200 files (16kHz, high quality)
  - Dog: 200 files (16kHz, high quality)  
  - Frog: 35 files (varies, decent)
  - Chicken: 30 files → Hen (varies, decent)
  - Bird: 200 files → skip (1-2s chirps, not crow/rooster)

Combined with our existing UrbanSound8K dogs (1040), we get:
  Dog: 1240 | Cat: 240 | Frog: 75 | Hen: 70

Then downloads missing classes from Freesound.

Usage:
  python src/scripts/build_dataset_v17.py              # Kaggle + Freesound scrape
  python src/scripts/build_dataset_v17.py --freesound-api KEY  # Kaggle + Freesound API
"""
import os, sys, shutil, argparse, subprocess

DATA_DIR = "data/animal1000"
KAGGLE_DATASET = "caoofficial/animal-sounds"

CLASS_MAP = {
    "Cat": "Cat",
    "Dog": "Dog",
    "Frog": "Frog",
    "Bird": "Bird",
    "Chicken": "Chicken",
    "Cow": "Cow",
}

# Merge old ESC-50 classes into new Kaggle classes
MERGE_CLASSES = {
    "Crow": "Bird",      # ESC-50 Crow → Bird
    "Insect": "Cow",     # ESC-50 Insect → Cow  
    "Hen": "Chicken",    # ESC-50 Hen → Chicken
    "Rooster": "Chicken", # ESC-50 Rooster → Chicken
}

# Classes still needed after Kaggle
TARGET_REAL = 200  # target real files per class


def merge_old_classes():
    """Move ESC-50 files from old class folders into new merged folders."""
    print("\n🔄 Merging old classes...")
    for old_cls, new_cls in MERGE_CLASSES.items():
        old_dir = os.path.join(DATA_DIR, old_cls)
        if not os.path.isdir(old_dir):
            continue
        new_dir = os.path.join(DATA_DIR, new_cls)
        os.makedirs(new_dir, exist_ok=True)

        moved = 0
        for fname in os.listdir(old_dir):
            if not fname.endswith(".wav"):
                continue
            src = os.path.join(old_dir, fname)
            dst = os.path.join(new_dir, f"esc50_{fname}")
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1

        print(f"   {old_cls} → {new_cls}: {moved} files")

        # Remove empty old directory
        remaining = [f for f in os.listdir(old_dir) if f.endswith(".wav")]
        if not remaining:
            shutil.rmtree(old_dir, ignore_errors=True)


def download_kaggle():
    """Download Kaggle dataset and copy relevant classes."""
    print("📦 Downloading Kaggle dataset...")
    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)
    except ImportError:
        print("   Installing kagglehub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)

    src_dir = os.path.join(path, "Animal-SDataset")
    print(f"   Source: {src_dir}\n")

    for kaggle_cls, our_cls in CLASS_MAP.items():
        src = os.path.join(src_dir, kaggle_cls)
        if not os.path.isdir(src):
            print(f"   ⚠️  {kaggle_cls} not found in dataset")
            continue

        dst = os.path.join(DATA_DIR, our_cls)
        os.makedirs(dst, exist_ok=True)

        # Count existing real files (not augmented)
        existing = len([f for f in os.listdir(dst)
                       if f.endswith(".wav") and not f.startswith("aug_")])

        copied = 0
        for fname in sorted(os.listdir(src)):
            if fname.startswith("."):
                continue
            src_file = os.path.join(src, fname)
            dst_file = os.path.join(dst, f"kaggle_{fname}")

            if os.path.exists(dst_file):
                continue

            shutil.copy2(src_file, dst_file)
            copied += 1

        # Remove augmented files
        aug_removed = 0
        for f in os.listdir(dst):
            if f.startswith("aug_"):
                os.remove(os.path.join(dst, f))
                aug_removed += 1

        total_real = len([f for f in os.listdir(dst)
                         if f.endswith(".wav") and not f.startswith("aug_")])
        print(f"   {our_cls:8s}: +{copied} real, -{aug_removed} aug → {total_real} total real")


def print_summary():
    """Show final dataset state."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    total = 0
    for cls_name in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        all_wavs = [f for f in os.listdir(cls_dir) if f.endswith(".wav")]
        real = [f for f in all_wavs if not f.startswith("aug_")]
        print(f"  {cls_name:10s}: {len(all_wavs):4d} total = {len(real):4d} real + {len(all_wavs)-len(real)} aug")
        total += len(real)
    print(f"  {'TOTAL':10s}: {total} real files")


def main():
    parser = argparse.ArgumentParser(description="Build v17 dataset (Kaggle + Freesound)")
    parser.add_argument("--freesound-api", type=str, metavar="API_KEY",
                        help="Freesound API key for full quality (skip for scrape)")
    parser.add_argument("--target", type=int, default=200, help="Target real files per class")
    args = parser.parse_args()

    global TARGET_REAL
    TARGET_REAL = args.target

    # Step 1: Download from Kaggle
    print("=" * 50)
    print("STEP 1: Kaggle Dataset + Merge old classes")
    print("=" * 50)
    download_kaggle()
    merge_old_classes()
    print_summary()

    # Step 2: Remaining classes need Freesound?
    print("\n" + "=" * 50)
    print("STEP 2: Remaining classes")
    print("=" * 50)
    any_needed = False
    for cls_name in ['Dog', 'Cat', 'Chicken', 'Frog', 'Bird', 'Cow']:
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if os.path.isdir(cls_dir):
            real = len([f for f in os.listdir(cls_dir)
                       if f.endswith(".wav") and not f.startswith("aug_")])
        else:
            real = 0
        needed = max(0, TARGET_REAL - real)
        flag = " ✅" if needed == 0 else f" → need {needed} more"
        print(f"  {cls_name:10s}: {real} real{flag}")
        if needed > 0:
            any_needed = True

    if any_needed:
        print(f"\nRun: python src/scripts/download_freesound.py --no-api --n {TARGET_REAL}")
        if args.freesound_api:
            print(f"Or:  python src/scripts/download_freesound.py --api {args.freesound_api} --n {TARGET_REAL}")
    else:
        print("\n✅ All classes have ≥{TARGET_REAL} real files!")


if __name__ == "__main__":
    main()
