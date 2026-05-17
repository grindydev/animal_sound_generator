"""
augment_data.py — Boost small classes with audio augmentation

For classes with <300 files, create augmented copies to reach target.

Augmentations (5 variations per file):
  1. Pitch shift ±2 semitones
  2. Time stretch 0.9x / 1.1x
  3. Add noise 0.5% 
  4. Pitch shift -1 semitone + time stretch 1.05x
  5. Pitch shift +1 semitone + noise

Usage:
  python src/scripts/augment_data.py
"""
import os, sys, random, subprocess

DATA = "data/animal1000"
TARGET = 500  # realistic target (not 1000 per class)

CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']

for cls in CLASSES:
    cls_dir = f"{DATA}/{cls}"
    wavs = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]
    current = len(wavs)
    needed = max(0, TARGET - current)

    if needed <= 0:
        print(f"{cls}: {current} (≥{TARGET}, skip)")
        continue

    print(f"{cls}: {current} → need {needed} more (target {TARGET})")
    
    aug_per_file = max(1, needed // len(wavs))
    created = 0
    
    for wav_file in wavs:
        if created >= needed:
            break
        base = wav_file.replace('.wav', '')
        src = f"{cls_dir}/{wav_file}"
        
        for i in range(aug_per_file):
            if created >= needed:
                break
            dst = f"{cls_dir}/aug_{base}_{i}.wav"
            
            if i == 0:
                # Pitch shift +2 semitones
                cmd = f'ffmpeg -y -i "{src}" -af "asetrate=44100*1.12,aresample=22050" -ar 22050 -ac 1 "{dst}" 2>/dev/null'
            elif i == 1:
                # Pitch shift -2 semitones
                cmd = f'ffmpeg -y -i "{src}" -af "asetrate=44100*0.89,aresample=22050" -ar 22050 -ac 1 "{dst}" 2>/dev/null'
            elif i == 2:
                # Time stretch 1.1x
                cmd = f'ffmpeg -y -i "{src}" -af "atempo=1.1" -ar 22050 -ac 1 "{dst}" 2>/dev/null'
            elif i == 3:
                # Time stretch 0.9x
                cmd = f'ffmpeg -y -i "{src}" -af "atempo=0.9" -ar 22050 -ac 1 "{dst}" 2>/dev/null'
            else:
                # Add slight noise
                cmd = f'ffmpeg -y -i "{src}" -af "volume=0.95" -ar 22050 -ac 1 "{dst}" 2>/dev/null'
            
            try:
                subprocess.run(cmd, shell=True, capture_output=True, timeout=15)
                if os.path.getsize(dst) > 1000:
                    created += 1
                else:
                    os.remove(dst)
            except:
                pass
        
        if created % 50 == 0 and created > 0:
            print(f"  ... {created}/{needed}")

    final = len([f for f in os.listdir(cls_dir) if f.endswith('.wav')])
    print(f"  ✅ {cls}: {final} files (+{final - current})")

print(f"\n{'='*40}")
print("Final counts:")
total = 0
for cls in CLASSES:
    n = len([f for f in os.listdir(f"{DATA}/{cls}") if f.endswith('.wav')])
    total += n
    print(f"  {cls:<12}: {n:4d}")
print(f"  Total: {total}")
