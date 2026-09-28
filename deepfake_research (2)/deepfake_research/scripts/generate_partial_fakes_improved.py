import subprocess
import os
import glob
import csv
import random
import json
import sys
import time
from datetime import timedelta

# ==================================================
# FAST PARTIAL FAKE GENERATOR (NO PREPROCESSING)
# ==================================================

REAL_VIDEOS_DIR = "dataset/roop_input_v2"
DONOR_DIR       = "dataset/donor_faces"
OUTPUT_DIR      = "dataset/fake_partial_v2"
TMP_DIR         = "dataset/tmp"
GROUND_TRUTH    = "dataset/ground_truth_partial_v2.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

random.seed(42)

# ==================================================
# LOAD FILES
# ==================================================
real_videos = sorted(glob.glob(f"{REAL_VIDEOS_DIR}/*.mp4"))
donors = sorted(glob.glob(f"{DONOR_DIR}/*.*"))

print("Real videos :", len(real_videos))
print("Donors      :", len(donors))

# ==================================================
# HELPERS
# ==================================================
def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True
    )
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except:
        return 0.0


def ffmpeg(args):
    cmd = ["ffmpeg", "-y"] + args
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


def run_roop(source, target, output):
    r = subprocess.run(
        [
            sys.executable,
            "roop/run.py",
            "-s", os.path.abspath(source),
            "-t", os.path.abspath(target),
            "-o", os.path.abspath(output),
            "--execution-provider", "cpu",
            "--skip-audio"
        ],
        capture_output=True,
        text=True
    )

    if r.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 50000:
        return True

    return False


# ==================================================
# MAIN
# ==================================================
rows = []
done = 0
failed = []
no_face = []

START_TIME = time.time()

for idx, video_path in enumerate(real_videos, start=1):

    vid = os.path.basename(video_path).replace(".mp4", "")
    language = "sinhala" if "sinhala" in vid.lower() else "tamil"

    output = os.path.join(OUTPUT_DIR, f"{vid}_partial.mp4")

    if os.path.exists(output) and os.path.getsize(output) > 100000:
        print("Already done:", vid)
        done += 1
        continue

    duration = get_duration(video_path)
    if duration < 40:
        continue

    success = False

    #  RETRY LOGIC (3 attempts with different time ranges)
    for attempt in range(3):

        # SHORTER SEGMENTS = FASTER PROCESSING ON CPU
        fake_len = random.choice([5, 6, 7, 8])  # Much shorter = much faster

        #  DIFFERENT REGIONS on each attempt
        if attempt == 0:
            fake_start = round(random.uniform(duration * 0.05, duration * 0.25), 2)
        elif attempt == 1:
            fake_start = round(random.uniform(duration * 0.30, duration * 0.50), 2)
        else:
            fake_start = round(random.uniform(duration * 0.50, duration * 0.70), 2)
        
        fake_end   = round(min(fake_start + fake_len, duration - 3), 2)

        donor = random.choice(donors)

        print(f"\n[{idx}] Attempt {attempt+1}/3 - {fake_end-fake_start:.1f}s")
        print(f"Video: {vid}")

        s1 = f"{TMP_DIR}/{vid}_1.mp4"
        sm = f"{TMP_DIR}/{vid}_m.mp4"
        sf = f"{TMP_DIR}/{vid}_f.mp4"
        s2 = f"{TMP_DIR}/{vid}_2.mp4"
        txt = f"{TMP_DIR}/{vid}.txt"

        try:
            # Extract Segment 1 (before)
            ok = ffmpeg(["-i", video_path, "-t", str(fake_start), "-c:v", "libx264", "-c:a", "aac", s1])
            if not ok:
                continue

            # Extract Segment 2 (middle - to be faked)
            ok = ffmpeg(["-i", video_path, "-ss", str(fake_start), "-to", str(fake_end), "-c:v", "libx264", "-c:a", "aac", sm])
            if not ok or os.path.getsize(sm) < 50000:
                continue

            # Extract Segment 3 (after)
            ok = ffmpeg(["-i", video_path, "-ss", str(fake_end), "-c:v", "libx264", "-c:a", "aac", s2])
            if not ok:
                continue

            # Run Roop face swap (with GPU if available, else CPU)
            print("  Running Roop...")
            try:
                # Try GPU first (cuda), fallback to CPU if not available
                r = subprocess.run([sys.executable, "roop/run.py", "-s", os.path.abspath(donor), "-t", os.path.abspath(sm), "-o", os.path.abspath(sf), "--execution-provider", "cuda", "--skip-audio"], capture_output=True, text=True, timeout=300)
                
                # If CUDA not available, try CPU
                if r.returncode != 0:
                    r = subprocess.run([sys.executable, "roop/run.py", "-s", os.path.abspath(donor), "-t", os.path.abspath(sm), "-o", os.path.abspath(sf), "--execution-provider", "cpu", "--skip-audio"], capture_output=True, text=True, timeout=600)
            except subprocess.TimeoutExpired:
                print("    Timeout, retrying...")
                continue
            
            if r.returncode != 0 or not os.path.exists(sf) or os.path.getsize(sf) < 50000:
                print("    Roop failed, retrying...")
                continue

            # Merge all 3 segments
            with open(txt, "w") as f:
                f.write(f"file '{os.path.abspath(s1)}'\n")
                f.write(f"file '{os.path.abspath(sf)}'\n")
                f.write(f"file '{os.path.abspath(s2)}'\n")

            ok = ffmpeg(["-f", "concat", "-safe", "0", "-i", txt, "-c:v", "libx264", "-c:a", "aac", output])

            if ok and os.path.exists(output):
                success = True
                break

        finally:
            for f in [s1, sm, sf, s2, txt]:
                if os.path.exists(f):
                    os.remove(f)

    # ==================================================
    # RESULT
    # ==================================================
    if success:
        done += 1
        print(" ✓ SUCCESS:", vid)

        rows.append({
            "video_id": vid + "_partial",
            "video_path": os.path.abspath(output),
            "language": language,
            "total_duration": round(duration, 2),
            "fake_start_sec": fake_start,
            "fake_end_sec": fake_end,
            "fake_duration": round(fake_end - fake_start, 2),
            "donor_face": os.path.basename(donor)
        })

    else:
        print("  FAILED:", vid)
        failed.append(vid)
        no_face.append(vid)

# ==================================================
# SAVE CSV (AND INCLUDE ANY PREVIOUSLY CREATED VIDEOS)
# ==================================================

# First, scan folder for ALL existing partial videos
all_existing_videos = sorted(glob.glob(f"{OUTPUT_DIR}/*_partial.mp4"))

print(f"\nScanning {OUTPUT_DIR} for all successful videos...")
print(f"Found {len(all_existing_videos)} partial videos in folder")

# Create a dict of existing videos for quick lookup
existing_video_ids = {os.path.basename(v).replace("_partial.mp4", ""): v for v in all_existing_videos}

# Add any missing videos from the folder to rows
for vid, video_path in existing_video_ids.items():
    # Check if already in rows
    if not any(r["video_id"] == vid + "_partial" for r in rows):
        language = "sinhala" if "sinhala" in vid.lower() else "tamil"
        duration = get_duration(video_path)
        
        rows.append({
            "video_id": vid + "_partial",
            "video_path": os.path.abspath(video_path),
            "language": language,
            "total_duration": round(duration, 2),
            "fake_start_sec": 0,
            "fake_end_sec": 0,
            "fake_duration": 0,
            "donor_face": "unknown (previously created)"
        })
        print(f"   Added missing video: {vid}")

# Save updated CSV
if rows:
    with open(GROUND_TRUTH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n Updated CSV with {len(rows)} total videos")

# ==================================================
# SUMMARY
# ==================================================
total_time = time.time() - START_TIME

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f" SUCCESS  : {done}")
print(f"FAILED   : {len(failed)}")
print(f"Success Rate: {(done/(done+len(failed))*100):.1f}%" if (done+len(failed)) > 0 else "N/A")
print(f"\nTotal Time : {str(timedelta(seconds=int(total_time)))}")
print(f"CSV Output : {GROUND_TRUTH}")
print("="*60)