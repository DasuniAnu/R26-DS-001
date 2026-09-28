import csv
import glob
import os
import subprocess
import json

# -----------------------------------
# GET VIDEO DURATION
# -----------------------------------
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

# -----------------------------------
# SINHALA
# -----------------------------------
files = sorted(glob.glob("dataset/real_sinhala/*.mp4"))

for i, path in enumerate(files, start=1):

    dur = get_duration(path)

    if dur < 120:
        continue

    rows.append({
        "video_id": f"sinhala_{i:03d}",
        "video_path": os.path.abspath(path),
        "label": 0,
        "class_name": "real",
        "language": "sinhala",
        "duration_sec": round(dur,2),
        "fake_start_sec": "",
        "fake_end_sec": "",
        "donor_face": ""
    })


# -----------------------------------
# TAMIL
# -----------------------------------
files = sorted(glob.glob("dataset/real_tamil/*.mp4"))

for i, path in enumerate(files, start=1):

    dur = get_duration(path)

    if dur < 120:
        continue

    rows.append({
        "video_id": f"tamil_{i:03d}",
        "video_path": os.path.abspath(path),
        "label": 0,
        "class_name": "real",
        "language": "tamil",
        "duration_sec": round(dur,2),
        "fake_start_sec": "",
        "fake_end_sec": "",
        "donor_face": ""
    })


# -----------------------------------
# CHECK
# -----------------------------------
if len(rows) == 0:
    print("No valid videos found.")
    exit()


# -----------------------------------
# SAVE
# -----------------------------------
os.makedirs("dataset", exist_ok=True)

out = "dataset/real_labels_v2.csv"

with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


# -----------------------------------
# SUMMARY
# -----------------------------------
sinhala = sum(1 for r in rows if r["language"]=="sinhala")
tamil   = sum(1 for r in rows if r["language"]=="tamil")

print("\nSaved:", out)
print("Sinhala :", sinhala)
print("Tamil   :", tamil)
print("Total   :", len(rows))