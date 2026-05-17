"""
collect_data.py — Download animal sounds to reach 1000+ samples per class

Sources:
  UrbanSound8K: Dog barks (~1,000) — direct download
  Xeno-Canto:   Crow, Rooster, Hen, Frog — free API, no auth
  Freesound:    Cat, Insect — free API

Usage: python src/scripts/collect_data.py
"""
import os, sys, json, time, shutil, zipfile
import urllib.request
import subprocess

DATA_DIR = "data/animal1000"
os.makedirs(DATA_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# STEP 1: Copy existing ESC-50 files
# ═══════════════════════════════════════════════════════════════
print("📦 Copying ESC-50 files...")
for cls in ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']:
    src = f"data/esc50/{cls}"
    dst = f"{DATA_DIR}/{cls}"
    os.makedirs(dst, exist_ok=True)
    if os.path.isdir(src):
        for f in os.listdir(src):
            if f.endswith('.wav'):
                shutil.copy2(f"{src}/{f}", f"{dst}/{f}")
    count = len(os.listdir(dst)) if os.path.isdir(dst) else 0
    print(f"  {cls}: {count} files")

# ═══════════════════════════════════════════════════════════════
# STEP 2: UrbanSound8K → Dog barks (~1,000 files)
# ═══════════════════════════════════════════════════════════════
print("\n🐕 UrbanSound8K → Dog...")
US8K_ZIP = "/tmp/urbansound8k.zip"
US8K_DIR = "/tmp/urbansound8k"

if not os.path.exists(US8K_DIR):
    print("  Downloading (6GB)...")
    url = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
    urllib.request.urlretrieve(url, "/tmp/urbansound8k.tar.gz")
    import tarfile
    with tarfile.open("/tmp/urbansound8k.tar.gz") as tar:
        tar.extractall(path="/tmp/")
    print("  ✅ Downloaded")

# Extract dog barks (classID=3)
import csv
meta_path = f"{US8K_DIR}/metadata/UrbanSound8K.csv"
if os.path.exists(meta_path):
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['class'] == 'dog_bark':
                src = f"{US8K_DIR}/audio/fold{row['fold']}/{row['slice_file_name']}"
                dst = f"{DATA_DIR}/Dog/{row['slice_file_name']}"
                if os.path.exists(src):
                    shutil.copy2(src, dst)
    print(f"  Dog: {len(os.listdir(f'{DATA_DIR}/Dog'))} files")
else:
    print("  ⚠️ UrbanSound8K metadata not found — skip")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Xeno-Canto API → Crow, Rooster, Hen, Frog
# ═══════════════════════════════════════════════════════════════
print("\n🐦 Xeno-Canto API (birds + frog)...")

XC_QUERIES = {
    'Crow': 'Corvus corone',
    'Rooster': 'Gallus gallus domesticus',
    'Hen': 'Gallus gallus domesticus type:call',
    'Frog': 'frog type:call',
}
XC_DIR = f"{DATA_DIR}/_xeno_canto"
os.makedirs(XC_DIR, exist_ok=True)

for cls, query in XC_QUERIES.items():
    print(f"  {cls}: searching '{query}'...")
    url = f"https://xeno-canto.org/api/2/recordings?query={urllib.parse.quote(query)}"
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
        recordings = data.get('recordings', [])
        print(f"    Found {len(recordings)} recordings")

        downloaded = 0
        target = 400
        dst_dir = f"{DATA_DIR}/{cls}"

        for rec in recordings:
            if downloaded >= target:
                break
            audio_url = rec.get('file')
            if not audio_url or not audio_url.startswith('http'):
                continue
            # Only download .mp3 files (convert to wav later)
            if not audio_url.endswith('.mp3'):
                continue

            fname = f"xc_{rec['id']}.mp3"
            outpath = f"{dst_dir}/{fname}"
            if os.path.exists(outpath):
                downloaded += 1
                continue

            try:
                urllib.request.urlretrieve(audio_url, outpath)
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"      {downloaded}/{target}")
                time.sleep(0.5)  # Rate limit
            except Exception as e:
                print(f"      ⚠️ Failed: {e}")
                continue

        print(f"    ✅ {cls}: {len(os.listdir(dst_dir))} files")

    except Exception as e:
        print(f"    ❌ API error: {e}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Freesound API → Cat, Insect
# ═══════════════════════════════════════════════════════════════
print("\n🐱 Freesound API (cat + insect)...")
print("  ⚠️ Needs Freesound API key. Set FREESOUND_API_KEY env var.")
print("  Skipping for now — run manually or sign up at freesound.org")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("📊 Data Summary")
print(f"{'='*50}")
for cls in ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']:
    d = f"{DATA_DIR}/{cls}"
    count = len(os.listdir(d)) if os.path.isdir(d) else 0
    bar = "=" * (count // 20)
    print(f"  {cls:<12}: {count:4d} {bar}")

print(f"\n🎯 Target: 1000 per class")
print(f"📁 Data in: {DATA_DIR}/")
print(f"\nNext: update config to data_dir='data/animal1000'")
