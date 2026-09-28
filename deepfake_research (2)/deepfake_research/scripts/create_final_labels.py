import os
import glob
import pandas as pd
import random
import subprocess
import json

random.seed(42)

rows = []

# -----------------------------------
# Helpers
# -----------------------------------
def detect_language(name):
    if "sinhala" in name.lower():
        return "sinhala"
    return "tamil"

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

# -----------------------------------
# Load partial metadata if exists
# -----------------------------------
partial_meta = {}

partial_csv = "dataset/ground_truth_partial_fixed.csv"

if os.path.exists(partial_csv):
    p = pd.read_csv(partial_csv)

    for _, row in p.iterrows():
        partial_meta[row["video_id"]] = row

# -----------------------------------
# REAL
# -----------------------------------
for path in sorted(glob.glob("dataset/roop_input/*.mp4")):

    vid = os.path.basename(path).replace(".mp4","")
    dur = get_duration(path)

    rows.append({
        "video_id": vid,
        "video_path": os.path.abspath(path),
        "label": 0,
        "class_name": "real",
        "language": detect_language(vid),
        "source_type": "real",
        "total_duration": dur,
        "fake_start_sec": "",
        "fake_end_sec": "",
        "fake_duration": "",
        "has_fake_region": 0,
        "donor_face": ""
    })

# -----------------------------------
# FULL FAKE
# -----------------------------------
for path in sorted(glob.glob("dataset/fake_full/*.mp4")):

    vid = os.path.basename(path).replace(".mp4","")
    dur = get_duration(path)

    rows.append({
        "video_id": vid,
        "video_path": os.path.abspath(path),
        "label": 1,
        "class_name": "full_fake",
        "language": detect_language(vid),
        "source_type": "fake_full",
        "total_duration": dur,
        "fake_start_sec": 0,
        "fake_end_sec": dur,
        "fake_duration": dur,
        "has_fake_region": 1,
        "donor_face": ""
    })

# -----------------------------------
# PARTIAL FAKE
# -----------------------------------
for path in sorted(glob.glob("dataset/fake_partial/*.mp4")):

    vid = os.path.basename(path).replace(".mp4","")
    dur = get_duration(path)

    if vid in partial_meta:
        meta = partial_meta[vid]

        fs = meta["fake_start_sec"]
        fe = meta["fake_end_sec"]
        fd = meta["fake_duration"]
        donor = meta["donor_face"]

    else:
        fs = ""
        fe = ""
        fd = ""
        donor = ""

    rows.append({
        "video_id": vid,
        "video_path": os.path.abspath(path),
        "label": 2,
        "class_name": "partial_fake",
        "language": detect_language(vid),
        "source_type": "fake_partial",
        "total_duration": dur,
        "fake_start_sec": fs,
        "fake_end_sec": fe,
        "fake_duration": fd,
        "has_fake_region": 1,
        "donor_face": donor
    })

# -----------------------------------
# DataFrame
# -----------------------------------
df = pd.DataFrame(rows)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# split
n = len(df)
train_end = int(n*0.70)
val_end = int(n*0.85)

df["split"] = ""

df.loc[:train_end-1,"split"] = "train"
df.loc[train_end:val_end-1,"split"] = "val"
df.loc[val_end:,"split"] = "test"

# save
df.to_csv("dataset/final_labels.csv", index=False)

print("Saved final_labels.csv")
print(df["class_name"].value_counts())
print(df["split"].value_counts())
print("Total:", len(df))