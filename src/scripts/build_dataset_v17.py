"""
build_dataset_v17.py — Download from 2 Kaggle datasets + merge ESC-50 → 6 classes

Sources:
  Kaggle 1: caoofficial/animal-sounds        (75-200 files/class)
  Kaggle 2: rushibalajiputthewad/...          (50 files/class)
  ESC-50:    existing local files              (40-80 files/class, merged)
  UrbanSound8K: existing local Dog files       (1040 files)
  Freesound: scrape for remaining              (optional)

Target: 500 real unique files per class

Usage:
  python src/scripts/build_dataset_v17.py
  python src/scripts/build_dataset_v17.py --freesound-api KEY
"""
import os, sys, shutil, argparse, subprocess

DATA_DIR = "data/animal1000"
TARGET_REAL = 500

KAGGLE_DATASETS = [
    {
        "name": "caoofficial/animal-sounds",
        "subdir": "Animal-SDataset",
        "map": {"Cat": "Cat", "Dog": "Dog", "Frog": "Frog",
                "Bird": "Bird", "Chicken": "Chicken", "Cow": "Cow"},
    },
    {
        "name": "rushibalajiputthewad/sound-classification-of-animal-voice",
        "subdir": "Animal-Soundprepros",
        "map": {"Cat": "Cat", "Dog": "Dog", "Frog": "Frog",
                "Chicken": "Chicken", "Cow": "Cow"},
    },
]

# Merge old ESC-50 classes into new classes
MERGE_CLASSES = {
    "Crow": "Bird",
    "Insect": "Cow",
    "Hen": "Chicken",
    "Rooster": "Chicken",
}

# For Freesound download
FREESOUND_QUERIES = {
    "Cat":     ["cat meow", "cat meowing", "kitten meow", "cat purr"],
    "Dog":     ["dog bark", "dog barking", "dog growl", "puppy bark"],
    "Bird":    ["bird chirp", "bird singing", "bird call", "bird sound"],
    "Chicken": ["chicken cluck", "hen clucking", "rooster crow", "rooster crowing"],
    "Cow":     ["cow moo", "cow mooing", "cattle sound"],
    "Frog":    ["frog croak", "frog sound", "toad croak", "frog call"],
}


def download_kaggle():
    """Download both Kaggle datasets and copy relevant classes."""
    try:
        import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub

    for ds in KAGGLE_DATASETS:
        print(f"\n📦 Kaggle: {ds['name']}...")
        path = kagglehub.dataset_download(ds["name"])
        src_dir = os.path.join(path, ds["subdir"])
        print(f"   Source: {src_dir}")

        for kaggle_cls, our_cls in ds["map"].items():
            src = os.path.join(src_dir, kaggle_cls)
            if not os.path.isdir(src):
                continue

            dst = os.path.join(DATA_DIR, our_cls)
            os.makedirs(dst, exist_ok=True)

            copied = 0
            for fname in sorted(os.listdir(src)):
                if fname.startswith("."):
                    continue
                dst_file = os.path.join(dst, f"kaggle2_{fname}" if "rushibala" in ds["name"] else f"kaggle_{fname}")
                if not os.path.exists(dst_file):
                    shutil.copy2(os.path.join(src, fname), dst_file)
                    copied += 1

            print(f"   {our_cls:10s}: +{copied}")


def merge_old_classes():
    """Move ESC-50 files from old class folders into new merged folders."""
    print("\n🔄 Merging ESC-50 classes...")
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
        remaining = [f for f in os.listdir(old_dir) if f.endswith(".wav")]
        if not remaining:
            shutil.rmtree(old_dir, ignore_errors=True)


def clean_augmented():
    """Remove all augmented (pitch-shifted) files."""
    print("\n🧹 Cleaning augmented files...")
    for cls_name in os.listdir(DATA_DIR):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        removed = 0
        for f in os.listdir(cls_dir):
            if f.startswith("aug_"):
                os.remove(os.path.join(cls_dir, f))
                removed += 1
        if removed:
            print(f"   {cls_name}: -{removed} augmented")


def print_summary():
    """Show final dataset state."""
    print("\n" + "=" * 55)
    print(f"{'DATASET SUMMARY':^55}")
    print("=" * 55)
    print(f"{'Class':12s} {'Real':>6s}  {'Needed':>6s}  Bar")
    print("-" * 55)
    total = 0
    rows = []
    for cls_name in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        real = len([f for f in os.listdir(cls_dir)
                    if f.endswith(".wav") and not f.startswith("aug_")])
        needed = max(0, TARGET_REAL - real)
        bar = "█" * (real // 20) if real > 0 else ""
        rows.append((cls_name, real, needed, bar))
        total += real

    for cls_name, real, needed, bar in sorted(rows, key=lambda x: -x[1]):
        flag = " ✅" if needed == 0 else f" +{needed}"
        print(f"  {cls_name:10s} {real:5d}  {flag:>8s}  {bar}")

    print("-" * 55)
    print(f"  {'TOTAL':10s} {total:5d}")
    print(f"  {'Target':10s} {TARGET_REAL * len(rows):5d}")


def main():
    parser = argparse.ArgumentParser(description="Build v17 dataset from Kaggle + ESC-50")
    parser.add_argument("--target", type=int, default=500, help="Target real files per class")
    args = parser.parse_args()

    global TARGET_REAL
    TARGET_REAL = args.target

    print("=" * 55)
    print("BUILDING v17 DATASET")
    print("=" * 55)
    print(f"Target: {TARGET_REAL} real files/class")

    download_kaggle()
    merge_old_classes()
    clean_augmented()
    print_summary()

    # Check if Freesound needed
    needed_classes = []
    for cls_name in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        real = len([f for f in os.listdir(cls_dir)
                    if f.endswith(".wav") and not f.startswith("aug_")])
        if real < TARGET_REAL:
            needed_classes.append(cls_name)

    if needed_classes:
        print(f"\n⚠️  {len(needed_classes)} classes below target: {', '.join(needed_classes)}")
        print(f"Run: python src/scripts/download_freesound.py --no-api --n {TARGET_REAL}")
    else:
        print(f"\n✅ All classes have ≥{TARGET_REAL} real files!")


if __name__ == "__main__":
    main()
