"""
download_freesound.py — Download animal sounds from Freesound.org

Two modes:
  1. --no-api          : Scrape search results (no key, MP3 preview ~64kbps)
  2. --api API_KEY     : Use Freesound API v2 (full quality WAV)

Get API key: https://freesound.org/apiv2/apply/ → copy key from your applications page

Usage:
  python src/scripts/download_freesound.py --no-api
  python src/scripts/download_freesound.py --api YOUR_API_KEY
  python src/scripts/download_freesound.py --api YOUR_API_KEY --n 200
"""
import os, sys, time, json, argparse, re
import urllib.request
import urllib.parse
import subprocess

# ── Config ──────────────────────────────────────────────

BASE_URL = "https://freesound.org"
SEARCH_URL = f"{BASE_URL}/search/?q={{query}}&page={{page}}&sort=rating_desc"
API_SEARCH = "https://freesound.org/apiv2/search/text/"

OUTPUT_DIR = "data/animal1000"
N_PER_CLASS = 150  # target new files per class

QUERIES = {
    "Dog":     ["dog bark", "dog barking", "dog growl", "puppy bark"],
    "Cat":     ["cat meow", "cat meowing", "cat purr", "kitten meow"],
    "Bird":    ["bird chirp", "bird singing", "bird call", "bird song"],
    "Chicken": ["chicken cluck", "hen clucking", "rooster crow", "rooster crowing"],
    "Cow":     ["cow moo", "cow mooing", "cattle sound", "bull moo"],
    "Frog":    ["frog croak", "frog sound", "toad croak", "frog call"],
}


# ═══════════════════════════════════════════════════════════
#  MODE 1: Scrape HTML (no API key)
# ═══════════════════════════════════════════════════════════

def scrape_sound_ids(query, max_pages=3):
    """Scrape Freesound search results for sound IDs and preview URLs."""
    results = []
    for page in range(1, max_pages + 1):
        url = SEARCH_URL.format(query=urllib.parse.quote(query), page=page)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode("utf-8")
        except Exception as e:
            if page == 1:
                print(f"    ❌ {e}")
            break

        ids = re.findall(r'data-sound-id="(\d+)"', html)
        mp3_urls = re.findall(r'https?://cdn\.freesound\.org/previews/[^"\s]+\.mp3', html)

        for i, sid in enumerate(ids):
            mp3 = mp3_urls[i] if i < len(mp3_urls) else None
            if mp3:
                results.append({"id": sid, "preview": mp3, "query": query})

        if len(ids) < 10:
            break
        time.sleep(0.3)

    return results


def download_preview(sound, out_path):
    """Download a Freesound preview file."""
    preview_url = sound["preview"]
    if not preview_url.startswith("http"):
        preview_url = "https:" + preview_url if preview_url.startswith("//") else BASE_URL + preview_url

    try:
        req = urllib.request.Request(preview_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    ❌ Download failed: {e}")
        return False


def convert_to_wav(src_path, wav_path):
    """Convert audio to WAV using ffmpeg."""
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ar", "22050", "-ac", "1",
                        "-sample_fmt", "s16", wav_path],
                       capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def scrape_mode():
    """Download sounds via HTML scraping (no API key)."""
    print("🔍 Freesound Scrape Mode (no API key)")

    for cls_name, queries in QUERIES.items():
        out_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(out_dir, exist_ok=True)

        existing_real = len([f for f in os.listdir(out_dir)
                            if f.endswith(".wav") and not f.startswith("aug_")])
        needed = max(0, N_PER_CLASS - existing_real)
        if needed <= 0:
            print(f"\n📦 {cls_name}: {existing_real} real — skip")
            continue

        print(f"\n📦 {cls_name}: need {needed} (have {existing_real})")

        # Scrape
        all_sounds = []
        for query in queries:
            try:
                sounds = scrape_sound_ids(query, max_pages=2)
                all_sounds.extend(sounds)
            except Exception as e:
                print(f"  ⚠️  query '{query}': {e}")
            if len(all_sounds) >= needed * 2:
                break

        # Deduplicate
        seen = set()
        unique = [s for s in all_sounds if s["id"] not in seen and not seen.add(s["id"])]
        print(f"  Scraped {len(unique)} unique sounds")

        # Download
        downloaded = existing_real
        errors = 0
        for sound in unique:
            if downloaded >= N_PER_CLASS:
                break
            if errors > 20:
                print(f"  ⚠️  Too many errors, moving on")
                break

            sid = sound["id"]
            wav_path = os.path.join(out_dir, f"fs_{sid}.wav")
            if os.path.exists(wav_path):
                downloaded += 1
                continue

            mp3_path = os.path.join(out_dir, f"_tmp_{sid}.mp3")
            if download_preview(sound, mp3_path):
                if convert_to_wav(mp3_path, wav_path):
                    os.remove(mp3_path)
                    downloaded += 1
                    if downloaded % 50 == 0:
                        print(f"  ... {downloaded}/{N_PER_CLASS}")
                else:
                    errors += 1
                    if os.path.exists(mp3_path):
                        os.remove(mp3_path)
            else:
                errors += 1

            time.sleep(0.2)

        print(f"  ✅ {cls_name}: {downloaded} real files")


# ═══════════════════════════════════════════════════════════
#  MODE 2: API (full quality)
# ═══════════════════════════════════════════════════════════

def api_search(query, api_key, page=1, page_size=50):
    """Search Freesound API v2."""
    url = f"{API_SEARCH}?query={urllib.parse.quote(query)}&page={page}&page_size={page_size}&fields=id,name,previews,download,type&token={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_download(url, out_path):
    """Download file from URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)


def api_mode(api_key):
    """Download full-quality sounds via API."""
    print("🔑 Freesound API Mode\n")

    for cls_name, queries in QUERIES.items():
        out_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(out_dir, exist_ok=True)

        existing_real = len([f for f in os.listdir(out_dir)
                            if f.endswith(".wav") and not f.startswith("aug_")])
        needed = max(0, N_PER_CLASS - existing_real)
        if needed <= 0:
            print(f"\n📦 {cls_name}: {existing_real} real + aug (≥{N_PER_CLASS}, skip)")
            continue

        print(f"\n📦 {cls_name}: {existing_real} real → need {needed} more (target {N_PER_CLASS})")

        downloaded = existing_real
        seen_ids = set()

        for query in queries:
            if downloaded >= N_PER_CLASS:
                break

            try:
                for page in range(1, 5):
                    data = api_search(query, api_key, page=page, page_size=50)
                    results = data.get("results", [])
                    if not results:
                        break

                    for sound in results:
                        if downloaded >= N_PER_CLASS:
                            break
                        sid = str(sound["id"])
                        if sid in seen_ids:
                            continue
                        seen_ids.add(sid)

                        wav_path = os.path.join(out_dir, f"fs_{sid}.wav")
                        if os.path.exists(wav_path):
                            downloaded += 1
                            continue

                        # Try download URL (needs auth), fall back to preview
                        dl_url = sound.get("download")
                        if not dl_url:
                            dl_url = sound.get("previews", {}).get("preview-lq-mp3")

                        if dl_url:
                            if api_download(dl_url, wav_path):
                                downloaded += 1
                                if downloaded % 25 == 0:
                                    print(f"    ... {downloaded}/{N_PER_CLASS}")
                            else:
                                print(f"    ❌ Failed: {sid}")
                        time.sleep(0.2)

            except urllib.error.HTTPError as e:
                if e.code == 401:
                    print(f"    ❌ Invalid API key. Get one at https://freesound.org/apiv2/apply/")
                    return
                print(f"    ❌ API error: {e}")
                break
            time.sleep(1)  # rate limit between pages

        print(f"  ✅ {cls_name}: {downloaded} files")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download animal sounds from Freesound")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--no-api", action="store_true", help="Scrape mode (no API key, OGG preview)")
    group.add_argument("--api", type=str, metavar="API_KEY", help="API key mode (full quality)")
    parser.add_argument("--n", type=int, default=150, help="Target files per class (default: 150)")
    parser.add_argument("--output", type=str, default="data/animal1000", help="Output directory")
    args = parser.parse_args()

    N_PER_CLASS = args.n
    OUTPUT_DIR = args.output

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.no_api:
        scrape_mode()
    elif args.api:
        api_mode(args.api)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for cls_name in QUERIES:
        out_dir = os.path.join(OUTPUT_DIR, cls_name)
        if os.path.isdir(out_dir):
            all_wavs = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
            real = [f for f in all_wavs if not f.startswith("aug_")]
            aug = [f for f in all_wavs if f.startswith("aug_")]
            print(f"  {cls_name:10s}: {len(all_wavs)} total = {len(real)} real + {len(aug)} augmented")
            # Remove augmented files (they are copies of real files)
            if aug:
                for f in aug:
                    os.remove(os.path.join(out_dir, f))
                print(f"    🧹 Removed {len(aug)} augmented files")
        else:
            print(f"  {cls_name:10s}: 0")
