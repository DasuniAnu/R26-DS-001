"""
generate_partial_from_fullfake.py
==================================
Creates partial fake videos WITHOUT running Roop.

How it works:
    Real video    : [=====REAL=====|=====REAL=====|=====REAL=====]
    Full fake     : [=====FAKE=====|=====FAKE=====|=====FAKE=====]
    Output partial: [=====REAL=====|=====FAKE=====|=====REAL=====]
                                    ^ 8-12 sec ^

Takes a random 8-12 second segment from the full fake video
and splices it into the matching real video.
Produces ground_truth_partial_v2.csv with exact timestamps.

2-5 minutes total. No Roop. No GPU needed.
"""

import os
import csv
import glob
import json
import random
import subprocess
import time
from datetime import timedelta

random.seed(42)

# ===========================================================
# PATHS — update if your structure differs
# ===========================================================

BASE         = "dataset/dataset"

REAL_DIR     = f"{BASE}/roop_input_v2"
FAKE_DIR     = f"{BASE}/fake_full_v2"
OUTPUT_DIR   = f"{BASE}/fake_partial_v2"
TMP_DIR      = f"{BASE}/tmp"
CSV_OUT      = f"{BASE}/ground_truth_partial_v2.csv"

# ===========================================================
# CONFIG
# ===========================================================

FAKE_SUFFIX      = "_fake"          # fake filename = real + this + .mp4
MIN_DURATION     = 30               # skip real videos shorter than this (sec)
FAKE_MIN_LEN     = 8                # minimum fake segment length (sec)
FAKE_MAX_LEN     = 12               # maximum fake segment length (sec)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR,    exist_ok=True)

# ===========================================================
# HELPERS
# ===========================================================

def run(args, timeout=120):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout
    )

def ffmpeg(*args):
    result = run(["ffmpeg", "-y"] + list(args))
    return result.returncode == 0

def get_duration(path):
    try:
        r = run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", path
        ])
        return float(json.loads(r.stdout)["format"]["duration"])
    except:
        return 0.0

def is_valid(path, min_size=50_000):
    return (
        os.path.exists(path) and
        os.path.getsize(path) > min_size and
        get_duration(path) > 1.0
    )

def remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except:
        pass

def detect_language(name):
    return "tamil" if "tamil" in name.lower() else "sinhala"

# ===========================================================
# FIND ALL MATCHING PAIRS
# ===========================================================

real_videos = sorted(glob.glob(f"{REAL_DIR}/*.mp4"))

pairs = []
for real_path in real_videos:
    name     = os.path.basename(real_path).replace(".mp4", "")
    fake_path = os.path.join(FAKE_DIR, f"{name}{FAKE_SUFFIX}.mp4")
    if os.path.exists(fake_path):
        pairs.append((name, real_path, fake_path))
    else:
        print(f"  [SKIP] No matching fake for: {name}")

print("=" * 65)
print("PARTIAL FAKE GENERATOR — No Roop, No GPU")
print("=" * 65)
print(f"Real videos found    : {len(real_videos)}")
print(f"Matched pairs        : {len(pairs)}")
print(f"Output folder        : {OUTPUT_DIR}")
print(f"Ground truth CSV     : {CSV_OUT}")
print("=" * 65)

# ===========================================================
# PROCESS EACH PAIR
# ===========================================================

csv_rows      = []
success_count = 0
skip_count    = 0
fail_count    = 0
failed_list   = []

START = time.time()

for idx, (name, real_path, fake_path) in enumerate(pairs, start=1):

    out_path = os.path.join(OUTPUT_DIR, f"{name}_partial.mp4")

    print(f"\n[{idx}/{len(pairs)}] {name}")

    # ── Already done ────────────────────────────────────────
    if is_valid(out_path):
        print("  Already exists — skipping")
        dur = get_duration(out_path)
        csv_rows.append({
            "video_id"      : f"{name}_partial",
            "video_path"    : os.path.abspath(out_path),
            "language"      : detect_language(name),
            "total_duration": round(dur, 2),
            "fake_start_sec": -1,
            "fake_end_sec"  : -1,
            "fake_duration" : -1,
            "donor_face"    : "previously_generated",
            "source"        : "splice_from_fullfake"
        })
        skip_count += 1
        continue

    # ── Check durations ─────────────────────────────────────
    real_dur = get_duration(real_path)
    fake_dur = get_duration(fake_path)

    print(f"  Real duration : {real_dur:.1f}s")
    print(f"  Fake duration : {fake_dur:.1f}s")

    if real_dur < MIN_DURATION:
        print(f"  SKIP — real video too short ({real_dur:.1f}s < {MIN_DURATION}s)")
        skip_count += 1
        continue

    use_dur = min(real_dur, fake_dur)

    # ── Pick random fake segment ─────────────────────────────
    fake_len   = random.uniform(FAKE_MIN_LEN, FAKE_MAX_LEN)
    max_start  = use_dur - fake_len - 3
    if max_start < 5:
        print("  SKIP — video too short for fake segment")
        skip_count += 1
        continue

    fake_start = round(random.uniform(use_dur * 0.2, min(use_dur * 0.7, max_start)), 2)
    fake_end   = round(min(fake_start + fake_len, use_dur - 2), 2)
    actual_len = round(fake_end - fake_start, 2)

    print(f"  Fake segment  : {fake_start}s → {fake_end}s ({actual_len}s)")

    # ── Temp file paths ──────────────────────────────────────
    seg_before  = os.path.join(TMP_DIR, f"{name}_A.mp4")
    seg_fake    = os.path.join(TMP_DIR, f"{name}_B.mp4")
    seg_after   = os.path.join(TMP_DIR, f"{name}_C.mp4")
    concat_file = os.path.join(TMP_DIR, f"{name}_list.txt")

    for f in [seg_before, seg_fake, seg_after, concat_file]:
        remove(f)

    ok = True

    try:
        # ── Extract BEFORE from real video ───────────────────
        print("  Extracting before segment...")
        ok = ffmpeg(
            "-i",  real_path,
            "-t",  str(fake_start),
            "-c:v", "libx264", "-c:a", "aac",
            "-crf", "23",
            seg_before
        )
        if not ok or not is_valid(seg_before):
            print("  FAIL — before segment extraction failed")
            ok = False

        # ── Extract FAKE segment from full fake video ─────────
        if ok:
            print("  Extracting fake segment from full fake...")
            ok = ffmpeg(
                "-i",  fake_path,
                "-ss", str(fake_start),
                "-to", str(fake_end),
                "-c:v", "libx264", "-c:a", "aac",
                "-crf", "23",
                seg_fake
            )
            if not ok or not is_valid(seg_fake, min_size=10_000):
                print("  FAIL — fake segment extraction failed")
                ok = False

        # ── Extract AFTER from real video ────────────────────
        if ok:
            print("  Extracting after segment...")
            ok = ffmpeg(
                "-i",  real_path,
                "-ss", str(fake_end),
                "-c:v", "libx264", "-c:a", "aac",
                "-crf", "23",
                seg_after
            )
            if not ok or not is_valid(seg_after, min_size=10_000):
                print("  FAIL — after segment extraction failed")
                ok = False

        # ── Write concat list ─────────────────────────────────
        if ok:
            with open(concat_file, "w", encoding="utf-8") as f:
                f.write(f"file '{os.path.abspath(seg_before)}'\n")
                f.write(f"file '{os.path.abspath(seg_fake)}'\n")
                f.write(f"file '{os.path.abspath(seg_after)}'\n")

            # ── Merge all 3 segments ──────────────────────────
            print("  Merging segments...")
            ok = ffmpeg(
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264", "-c:a", "aac",
                "-crf", "23",
                out_path
            )
            if not ok or not is_valid(out_path):
                print("  FAIL — merge failed")
                remove(out_path)
                ok = False

    except Exception as e:
        print(f"  ERROR — {e}")
        ok = False

    finally:
        for f in [seg_before, seg_fake, seg_after, concat_file]:
            remove(f)

    # ── Record result ─────────────────────────────────────────
    if ok:
        final_dur = get_duration(out_path)
        print(f"  SUCCESS — {final_dur:.1f}s output video")
        success_count += 1
        csv_rows.append({
            "video_id"      : f"{name}_partial",
            "video_path"    : os.path.abspath(out_path),
            "language"      : detect_language(name),
            "total_duration": round(final_dur, 2),
            "fake_start_sec": fake_start,
            "fake_end_sec"  : fake_end,
            "fake_duration" : actual_len,
            "donor_face"    : "spliced_from_fullfake",
            "source"        : "splice_from_fullfake"
        })
    else:
        fail_count += 1
        failed_list.append(name)

# ===========================================================
# SAVE GROUND TRUTH CSV
# ===========================================================

print("\n" + "=" * 65)
print("SAVING GROUND TRUTH CSV")
print("=" * 65)

# Also pick up any previously existing partial videos
# not processed in this run
existing = glob.glob(f"{OUTPUT_DIR}/*_partial.mp4")
existing_ids = {
    os.path.basename(v).replace("_partial.mp4", "")
    for v in existing
}
csv_ids = {r["video_id"].replace("_partial", "") for r in csv_rows}

for vid_id in existing_ids - csv_ids:
    path = os.path.join(OUTPUT_DIR, f"{vid_id}_partial.mp4")
    dur  = get_duration(path)
    csv_rows.append({
        "video_id"      : f"{vid_id}_partial",
        "video_path"    : os.path.abspath(path),
        "language"      : detect_language(vid_id),
        "total_duration": round(dur, 2),
        "fake_start_sec": -1,
        "fake_end_sec"  : -1,
        "fake_duration" : -1,
        "donor_face"    : "unknown",
        "source"        : "previously_generated"
    })

csv_rows.sort(key=lambda r: r["video_id"])

fields = [
    "video_id", "video_path", "language",
    "total_duration", "fake_start_sec",
    "fake_end_sec", "fake_duration",
    "donor_face", "source"
]

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"CSV saved : {CSV_OUT}")
print(f"CSV rows  : {len(csv_rows)}")

# ===========================================================
# SUMMARY
# ===========================================================

elapsed = time.time() - START

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"Total pairs      : {len(pairs)}")
print(f"Successfully made: {success_count}")
print(f"Already existed  : {skip_count}")
print(f"Failed           : {fail_count}")
print(f"Time taken       : {timedelta(seconds=int(elapsed))}")
print(f"Output folder    : {os.path.abspath(OUTPUT_DIR)}")
print(f"Ground truth CSV : {os.path.abspath(CSV_OUT)}")

# Verify CSV has timestamp data
with_timestamps = sum(
    1 for r in csv_rows if r["fake_start_sec"] != -1
)
print(f"\nCSV entries with timestamps : {with_timestamps}")
print(f"CSV entries without         : {len(csv_rows) - with_timestamps}")

if with_timestamps > 0:
    print("\nTemporal IoU CAN be computed for",
          with_timestamps, "videos.")
else:
    print("\nNo timestamp data — run this script fresh to generate.")

if failed_list:
    print("\nFailed videos:")
    for v in failed_list:
        print(f"  {v}")

print("=" * 65)
print("DONE")
print("=" * 65)