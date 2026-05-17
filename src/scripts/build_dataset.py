"""
build_dataset.py — Download training data (1000+ samples/class)

Sources:
  ESC-50:        640 clean animal clips (direct download)
  UrbanSound8K:  ~1,000 dog barks (direct download, 6GB)
  Xeno-Canto:    Crow, Rooster, Hen, Frog (free API, ~500 each)

Usage:
  python src/scripts/build_dataset.py

Output:
  data/animal1000/          # all training wav files
  data/animal1000.zip       # zip for Google Drive upload
"""
import os, sys, json, time, shutil, csv, tarfile, zipfile, urllib.request, urllib.parse, subprocess

DATA = "data/animal1000"
CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']

os.makedirs(DATA, exist_ok=True)
for cls in CLASSES:
    os.makedirs(f"{DATA}/{cls}", exist_ok=True)


def check_ffmpeg():
    """Ensure ffmpeg is available for mp3→wav conversion."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return True
    except:
        return False


def count_files(cls):
    return len([f for f in os.listdir(f"{DATA}/{cls}") if f.endswith('.wav')])


def convert_mp3_to_wav(cls):
    """Convert all mp3 files in class dir to wav (22050Hz, mono)."""
    cls_dir = f"{DATA}/{cls}"
    for f in os.listdir(cls_dir):
        if not f.endswith('.mp3'):
            continue
        mp3_path = f"{cls_dir}/{f}"
        wav_path = f"{cls_dir}/{f.replace('.mp3', '.wav')}"
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', mp3_path,
                '-ar', '22050', '-ac', '1', '-sample_fmt', 's16',
                wav_path
            ], capture_output=True, timeout=30)
            os.remove(mp3_path)
        except:
            pass


# ═══════════════════════════════════════════════════════════════
#  STEP 1: ESC-50 (640 clean clips)
# ═══════════════════════════════════════════════════════════════
print("=" * 50)
print("STEP 1: ESC-50 (640 files)")
print("=" * 50)

ESC50_ZIP = "/tmp/esc50.zip"
ESC50_DIR = "/tmp/ESC-50-master"

if not os.path.exists(ESC50_DIR):
    print("Downloading ESC-50...")
    urllib.request.urlretrieve(
        "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
        ESC50_ZIP
    )
    with zipfile.ZipFile(ESC50_ZIP) as z:
        z.extractall("/tmp/")
    print("✅ Downloaded")

# Map ESC-50 categories to our classes
cls_map = {
    'dog': 'Dog', 'cat': 'Cat', 'rooster': 'Rooster', 'cock': 'Rooster',
    'frog': 'Frog', 'crow': 'Crow', 'insect': 'Insect', 'cricket': 'Insect',
    'hen': 'Hen', 'chicken': 'Hen', 'chirping_birds': 'Crow', 'crickets': 'Insect',
}

audio_dir = f"{ESC50_DIR}/audio"
meta_path = f"{ESC50_DIR}/meta/esc50.csv"

if os.path.exists(meta_path):
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            cat = row['category'].lower()
            for kw, cls in cls_map.items():
                if kw in cat:
                    src = f"{audio_dir}/{row['filename']}"
                    dst = f"{DATA}/{cls}/{row['filename']}"
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    break

for cls in CLASSES:
    print(f"  {cls}: {count_files(cls)}")


# ═══════════════════════════════════════════════════════════════
#  STEP 2: UrbanSound8K → Dog barks (~1,000)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 2: UrbanSound8K → Dog (6GB download)")
print("=" * 50)

US8K_DIR = "/tmp/UrbanSound8K"

if not os.path.exists(US8K_DIR):
    print("Downloading UrbanSound8K (6GB, this will take a few minutes)...")
    urllib.request.urlretrieve(
        "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz",
        "/tmp/us8k.tar.gz"
    )
    print("Extracting...")
    with tarfile.open("/tmp/us8k.tar.gz") as tar:
        tar.extractall(path="/tmp/")
    print("✅ Downloaded + extracted")

us8k_meta = f"{US8K_DIR}/metadata/UrbanSound8K.csv"
if os.path.exists(us8k_meta):
    with open(us8k_meta) as f:
        for row in csv.DictReader(f):
            if 'dog' in row['class'].lower():
                src = f"{US8K_DIR}/audio/fold{row['fold']}/{row['slice_file_name']}"
                dst = f"{DATA}/Dog/d_{row['slice_file_name']}"
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
    print(f"  Dog: {count_files('Dog')}")
else:
    print("  ⚠️ UrbanSound8K metadata not found")


# ═══════════════════════════════════════════════════════════════
#  STEP 3: Xeno-Canto → Crow, Rooster, Hen, Frog
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 3: Xeno-Canto API (birds + frog)")
print("=" * 50)

XC_QUERIES = {
    'Crow': 'Corvus corone',
    'Rooster': 'Gallus gallus',
    'Hen': 'Gallus gallus',
    'Frog': 'frog',
}

for cls, query in XC_QUERIES.items():
    current = count_files(cls)
    target = max(0, 500 - current)
    if target <= 0:
        print(f"  {cls}: already {current}, skipping")
        continue

    print(f"  {cls}: searching '{query}' (need {target})...")
    url = f"https://xeno-canto.org/api/2/recordings?query={urllib.parse.quote(query)}"

    try:
        with urllib.request.urlopen(url) as r:
            data = json.loads(r.read())
        recs = data.get('recordings', [])
        print(f"    Found {len(recs)} recordings")

        downloaded = 0
        for rec in recs:
            if downloaded >= target:
                break
            audio_url = rec.get('file', '')
            if not audio_url.startswith('http'):
                continue
            fname = f"xc_{rec['id']}.mp3"
            path = f"{DATA}/{cls}/{fname}"
            if os.path.exists(path):
                downloaded += 1
                continue
            try:
                urllib.request.urlretrieve(audio_url, path)
                downloaded += 1
                if downloaded % 100 == 0:
                    print(f"      {downloaded}/{target}")
                time.sleep(0.3)
            except:
                pass

        print(f"    Downloaded: {downloaded} mp3 files")

        # Convert mp3 → wav
        if check_ffmpeg():
            print(f"    Converting mp3 → wav...")
            convert_mp3_to_wav(cls)
        else:
            print(f"    ⚠️ ffmpeg not found — install: brew install ffmpeg")

        print(f"    ✅ {cls}: {count_files(cls)} total")

    except Exception as e:
        print(f"    ❌ {e}")


# ═══════════════════════════════════════════════════════════════
#  SUMMARY + ZIP
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

total = 0
for cls in CLASSES:
    n = count_files(cls)
    total += n
    bar = "█" * (n // 20)
    print(f"  {cls:<12}: {n:4d} {bar}")

print(f"\n  Total: {total} files across {len(CLASSES)} classes")
print(f"  Target: 1000 per class = 7000 total")

# Create zip
zip_path = f"{DATA}.zip"
print(f"\n📦 Creating {zip_path}...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for cls in CLASSES:
        cls_dir = f"{DATA}/{cls}"
        for f in os.listdir(cls_dir):
            if f.endswith('.wav'):
                z.write(f"{cls_dir}/{f}", f"animal1000/{cls}/{f}")

size_mb = os.path.getsize(zip_path) / 1024**2
print(f"✅ {zip_path} ({size_mb:.0f} MB)")
print(f"\nUpload to: MyDrive/animal_sound_generator/data/animal1000.zip")
