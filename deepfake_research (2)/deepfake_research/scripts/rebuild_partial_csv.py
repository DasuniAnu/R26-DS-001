# scripts/rebuild_partial_csv.py
# FULL rebuild of partial fake CSV
# Recovers donor faces using same random seed logic

import os
import glob
import csv
import subprocess
import json
import random

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
VIDEO_DIR   = "dataset/fake_partial"
DONOR_DIR   = "dataset/donor_faces"
OUTPUT_CSV  = "dataset/ground_truth_partial_fixed.csv"

FAKE_SEGMENT_SEC = 10

# SAME seed used in original script
random.seed(123)

# -------------------------------------------------
# Load donor faces in SAME sorted order
# -------------------------------------------------
donor_faces = sorted(
    glob.glob(os.path.join(DONOR_DIR, "*.jpg")) +
    glob.glob(os.path.join(DONOR_DIR, "*.JPG")) +
    glob.glob(os.path.join(DONOR_DIR, "*.png")) +
    glob.glob(os.path.join(DONOR_DIR, "*.PNG"))
)

if not donor_faces:
    print(" No donor faces found")
    exit()

# -------------------------------------------------
# Load partial fake videos
# -------------------------------------------------
videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))

if not videos:
    print("❌ No partial fake videos found")
    exit()

# -------------------------------------------------
# Duration helper
# -------------------------------------------------
def get_duration(path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                path
            ],
            capture_output=True,
            text=True
        )

        data = json.loads(result.stdout)
        return round(float(data["format"]["duration"]), 2)

    except:
        return 0.0


# -------------------------------------------------
# Build rows
# -------------------------------------------------
rows = []

for path in videos:

    name = os.path.basename(path).replace(".mp4", "")

    # Recover original donor selection
    donor = random.choice(donor_faces)
    donor_name = os.path.basename(donor)

    # Language detection
    if "sinhala" in name.lower():
        language = "sinhala"
    else:
        language = "tamil"

    # Duration
    duration = get_duration(path)

    # Recreate fake segment timing
    if duration > 25:
        min_start = 5.0
        max_start = duration * 0.5

        fake_start = round(random.uniform(min_start, max_start), 2)
        fake_end   = round(
            min(fake_start + FAKE_SEGMENT_SEC, duration - 3),
            2
        )
    else:
        fake_start = 0.0
        fake_end   = 0.0

    fake_duration = round(fake_end - fake_start, 2)

    rows.append({
        "video_id": name,
        "video_path": os.path.abspath(path),
        "language": language,
        "total_duration": duration,
        "fake_start_sec": fake_start,
        "fake_end_sec": fake_end,
        "fake_duration": fake_duration,
        "donor_face": donor_name,
        "type": "partial_fake"
    })

# -------------------------------------------------
# Save CSV
# -------------------------------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

print(" FULL CSV rebuilt successfully")
print("Saved :", OUTPUT_CSV)
print("Rows  :", len(rows))