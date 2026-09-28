# scripts/build_labels.py

import csv
import glob
import os
import subprocess
import json


#  Get video duration using ffprobe
def get_duration(path):
    r = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            path
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except:
        return 0.0


rows = []

#  Sinhala videos
for path in glob.glob("dataset/real_sinhala/*.mp4"):
    dur = get_duration(path)

    #  Only keep videos longer than 5 minutes
    if dur < 300:
        continue

    rows.append({
        "video_path": path,
        "label": 0,
        "type": "real",
        "language": "sinhala",
        "duration": round(dur, 2),
        "fake_start": "",
        "fake_end": "",
        "donor_face": ""
    })


#  Tamil videos
for path in glob.glob("dataset/real_tamil/*.mp4"):
    dur = get_duration(path)

    #  Only keep videos longer than 5 minutes
    if dur < 300:
        continue

    rows.append({
        "video_path": path,
        "label": 0,
        "type": "real",
        "language": "tamil",
        "duration": round(dur, 2),
        "fake_start": "",
        "fake_end": "",
        "donor_face": ""
    })


#  Safety check (avoid crash if no valid videos)
if len(rows) == 0:
    print(" No valid videos found (all < 5 min or missing).")
    print(" Download longer videos (>10 min recommended).")
    exit()


#  Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)


#  Save CSV (UTF-8 safe)
with open("dataset/master_labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


#  Summary
sinhala = sum(1 for r in rows if r["language"] == "sinhala")
tamil   = sum(1 for r in rows if r["language"] == "tamil")
total_videos = len(rows)
total_minutes = sum(r["duration"] for r in rows) / 60


print("\n=== Dataset Summary ===")
print(f"Sinhala videos : {sinhala}")
print(f"Tamil videos   : {tamil}")
print(f"Total videos   : {total_videos}")
print(f"Total duration : {total_minutes:.1f} minutes")

print("\nCSV saved to dataset/master_labels.csv")
print("Ready to generate fake videos ")