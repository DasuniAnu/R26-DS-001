"""
Generate partial fake videos: real | middle | real structure
For testing, we'll just use the middle segment as-is (placeholder for Roop)
This verifies concatenation works correctly
"""
import subprocess
import os
import glob
import csv
import random
import json
import sys

# ── Configuration ────────────────────────────────────────────────
REAL_VIDEOS_DIR = "dataset/roop_input_v2"
OUTPUT_DIR = "dataset/fake_partial_v2"
TMP_DIR = "dataset/tmp"
GROUND_TRUTH = "dataset/ground_truth_partial_v2.csv"

# Fake segment length in seconds — SHORT for testing
FAKE_SEGMENT_SEC = random.choice([5, 7, 10])

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ── Load videos ──────────────────────────────────────────────────
real_videos = sorted(glob.glob(f"{REAL_VIDEOS_DIR}/*.mp4"))

if not real_videos:
    print(f"ERROR: No videos found in {REAL_VIDEOS_DIR}")
    sys.exit(1)

print(f"Real videos: {len(real_videos)}")

# ── Helper: get video duration ───────────────────────────────────
def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except:
        return 0.0


# ── Helper: run ffmpeg silently ──────────────────────────────────
def run_ffmpeg(args):
    cmd = ["ffmpeg", "-y"] + args
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="ignore")
    return result.returncode == 0


# ── Main loop ────────────────────────────────────────────────────
ground_truth_rows = []
random.seed(123)

succeeded = 0
failed = []
MAX_VIDEOS = 10  # Process first 10 for testing

for video_path in real_videos[:MAX_VIDEOS]:
    vid_id = os.path.basename(video_path).replace(".mp4", "")
    language = "sinhala" if "sinhala" in video_path.lower() else "tamil"
    output = os.path.join(OUTPUT_DIR, f"{vid_id}_partial.mp4")

    # Skip if already done
    if os.path.exists(output) and os.path.getsize(output) > 100000:
        print(f"Already done: {vid_id} — skipping")
        succeeded += 1
        continue

    # Get duration
    duration = get_duration(video_path)
    if duration < 20:
        print(f"Too short ({duration:.1f}s): {vid_id} — skipping")
        continue

    # Choose random fake segment in middle 20–70% of video
    min_start = 5.0
    max_start = duration * 0.5
    fake_start = round(random.uniform(min_start, max_start), 2)
    fake_end = round(min(fake_start + FAKE_SEGMENT_SEC, duration - 3.0), 2)

    # Safety check
    if fake_end - fake_start < 5:
        print(f"Cannot fit fake segment: {vid_id} — skipping")
        continue

    print(f"\n{'='*50}")
    print(f"Video      : {vid_id}")
    print(f"Duration   : {duration:.1f}s")
    print(f"Fake seg   : {fake_start}s → {fake_end}s ({fake_end - fake_start:.1f}s)")
    print(f"{'='*50}")

    # Define temp file paths
    seg1 = os.path.abspath(f"{TMP_DIR}/{vid_id}_s1.mp4")
    seg_mid = os.path.abspath(f"{TMP_DIR}/{vid_id}_smid.mp4")
    seg_fake = os.path.abspath(f"{TMP_DIR}/{vid_id}_sfake.mp4")
    seg2 = os.path.abspath(f"{TMP_DIR}/{vid_id}_s2.mp4")
    cat_file = os.path.abspath(f"{TMP_DIR}/{vid_id}_concat.txt")
    abs_vid = os.path.abspath(video_path)

    # Step 1 — Cut segment 1: start to fake_start (real)
    print("  Cutting segment 1 (real)...")
    ok = run_ffmpeg([
        "-i", abs_vid,
        "-t", str(fake_start),
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "1",
        seg1
    ])
    if not ok or not os.path.exists(seg1):
        print(f"  FAILED: Could not cut segment 1")
        failed.append(vid_id)
        continue

    # Step 2 — Cut middle segment: fake_start to fake_end
    print("  Cutting middle segment...")
    ok = run_ffmpeg([
        "-i", abs_vid,
        "-ss", str(fake_start),
        "-to", str(fake_end),
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "1",
        seg_mid
    ])
    if not ok or not os.path.exists(seg_mid):
        print(f"  FAILED: Could not cut middle segment")
        failed.append(vid_id)
        continue

    # Step 3 — Cut segment 2: fake_end to end (real)
    print("  Cutting segment 2 (real)...")
    ok = run_ffmpeg([
        "-i", abs_vid,
        "-ss", str(fake_end),
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "1",
        seg2
    ])
    if not ok or not os.path.exists(seg2):
        print(f"  FAILED: Could not cut segment 2")
        failed.append(vid_id)
        continue

    # Step 4 — For now, just copy middle segment (placeholder for Roop)
    # In production, this would run Roop face swap
    print("  Using middle segment as fake placeholder...")
    import shutil
    try:
        shutil.copy2(seg_mid, seg_fake)
    except Exception as e:
        print(f"  FAILED: Could not copy segment: {e}")
        failed.append(vid_id)
        continue

    print(f"  Fake segment ready: {os.path.getsize(seg_fake)/1e6:.1f} MB")

    # Step 5 — Write concat list
    with open(cat_file, "w", encoding="utf-8") as f:
        f.write(f"file '{seg1}'\n")
        f.write(f"file '{seg_fake}'\n")
        f.write(f"file '{seg2}'\n")

    # Step 6 — Merge all 3 segments (re-encode to ensure proper merge)
    print("  Merging segments...")
    abs_output = os.path.abspath(output)
    ok = run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", cat_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-crf", "23",
        abs_output
    ])

    # Clean up temp files
    for f in [seg1, seg_mid, seg_fake, seg2, cat_file]:
        if os.path.exists(f):
            os.remove(f)

    # Verify output
    if os.path.exists(abs_output) and os.path.getsize(abs_output) > 100000:
        size_mb = os.path.getsize(abs_output) / 1e6
        print(f"  SUCCESS: {vid_id} ({size_mb:.1f} MB)")

        ground_truth_rows.append({
            "video_id": vid_id + "_partial",
            "video_path": abs_output,
            "language": language,
            "total_duration": round(duration, 2),
            "fake_start_sec": fake_start,
            "fake_end_sec": fake_end,
            "fake_duration": round(fake_end - fake_start, 2),
            "donor_face": "placeholder"
        })
        succeeded += 1
    else:
        print(f"  FAILED: merge did not produce output")
        failed.append(vid_id)

# ── Save ground truth CSV ────────────────────────────────────────
if ground_truth_rows:
    with open(GROUND_TRUTH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ground_truth_rows[0].keys())
        w.writeheader()
        w.writerows(ground_truth_rows)

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"Succeeded: {succeeded}")
print(f"Failed   : {len(failed)}")
if failed:
    print(f"Failed videos: {failed}")
print(f"Ground truth saved to: {GROUND_TRUTH}")
