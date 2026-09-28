import subprocess
import glob
import os
import sys

# -------------------------------
# INPUT FOLDERS
# -------------------------------
SINHALA_DIR = "dataset/real_sinhala"
TAMIL_DIR   = "dataset/real_tamil"

TRIM_DIR = "dataset/trimmed_v2"
ROOP_DIR = "dataset/roop_input_v2"

os.makedirs(TRIM_DIR, exist_ok=True)
os.makedirs(ROOP_DIR, exist_ok=True)

# -------------------------------
# COLLECT FILES
# -------------------------------
sinhala_videos = sorted(glob.glob(os.path.join(SINHALA_DIR, "*.mp4")))
tamil_videos   = sorted(glob.glob(os.path.join(TAMIL_DIR, "*.mp4")))

videos = []

for i, path in enumerate(sinhala_videos, start=1):
    videos.append((path, f"sinhala_{i:03d}.mp4"))

for i, path in enumerate(tamil_videos, start=1):
    videos.append((path, f"tamil_{i:03d}.mp4"))

if not videos:
    print("No videos found.")
    sys.exit()

print("Total videos found:", len(videos))

# -------------------------------
# PROCESS
# -------------------------------
for path, name in videos:

    trim_path = os.path.join(TRIM_DIR, name)
    roop_path = os.path.join(ROOP_DIR, name)

    print("\n" + "="*60)
    print("Input :", path)
    print("Name  :", name)
    print("="*60)

    # STEP 1 Trim to 10 min
    if os.path.exists(trim_path):
        print("Already trimmed")
    else:
        print("Trimming to 10 min...")

        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", path,
            "-t", "600",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            trim_path
        ])

        if result.returncode != 0:
            print("Trim failed")
            continue

    # STEP 2 Create roop input
    if os.path.exists(roop_path):
        print("Already in roop_input_v2")
    else:
        print("Creating Roop version...")

        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", trim_path,
            "-t", "120",
            "-vf", "fps=15,scale=480:-2",
            "-c:v", "libx264",
            "-crf", "28",
            "-c:a", "aac",
            roop_path
        ])

        if result.returncode != 0:
            print("Roop creation failed")
            continue

        size = os.path.getsize(roop_path)/1e6
        print(f"Saved {name} ({size:.1f} MB)")

print("\nDONE")
print("Trimmed :", TRIM_DIR)
print("Roop    :", ROOP_DIR)