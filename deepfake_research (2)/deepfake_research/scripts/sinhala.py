# scripts/process_sinhala.py
# ONE-RUN PIPELINE:
# real_sinhala (long videos)
# -> trimmed_sinhala (10 min)
# -> roop_input (2 min / 480p / 15fps)

import subprocess
import glob
import os
import sys

# -------------------------------
# CONFIG
# -------------------------------
INPUT_DIR   = "dataset/real_sinhala"
TRIM_DIR    = "dataset/trimmed_sinhala"
ROOP_DIR    = "dataset/roop_input"

os.makedirs(TRIM_DIR, exist_ok=True)
os.makedirs(ROOP_DIR, exist_ok=True)

# -------------------------------
# FIND INPUT FILES
# -------------------------------
videos = sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4")))

if not videos:
    print(" No Sinhala videos found in dataset/real_sinhala/")
    sys.exit()

print(f"Found Sinhala videos: {len(videos)}")

# -------------------------------
# PROCESS EACH VIDEO
# -------------------------------
for i, path in enumerate(videos, start=1):

    name = f"sinhala_{i:02d}.mp4"

    trimmed_path = os.path.join(TRIM_DIR, name)
    roop_path    = os.path.join(ROOP_DIR, name)

    print("\n" + "=" * 50)
    print("Input   :", path)
    print("Name    :", name)
    print("=" * 50)

    # --------------------------------
    # STEP 1: Trim to first 10 minutes
    # --------------------------------
    if os.path.exists(trimmed_path):
        print(" Already trimmed:", name)
    else:
        print(" Trimming to 10 minutes...")

        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", path,
            "-t", "600",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            trimmed_path
        ])

        if result.returncode != 0 or not os.path.exists(trimmed_path):
            print(" Trim failed:", name)
            continue

        print("Trimmed:", trimmed_path)

    # --------------------------------
    # STEP 2: Create roop_input version
    # 2 min / 480p / 15fps
    # --------------------------------
    if os.path.exists(roop_path):
        print("⏭ Already in roop_input:", name)
    else:
        print("⚙ Creating Roop version...")

        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", trimmed_path,
            "-t", "120",
            "-vf", "fps=15,scale=480:-2",
            "-c:v", "libx264",
            "-crf", "28",
            "-c:a", "aac",
            roop_path
        ])

        if result.returncode != 0 or not os.path.exists(roop_path):
            print(" Roop input creation failed:", name)
            continue

        size_mb = os.path.getsize(roop_path) / 1e6
        print(f" Added to roop_input: {name} ({size_mb:.1f} MB)")

# -------------------------------
# DONE
# -------------------------------
print("\n" + "=" * 50)
print(" Sinhala pipeline complete")
print("Trimmed videos  :", TRIM_DIR)
print("Roop input vids :", ROOP_DIR)
print("=" * 50)