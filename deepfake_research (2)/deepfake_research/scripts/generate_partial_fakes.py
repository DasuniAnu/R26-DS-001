import os
import sys
import csv
import glob
import json
import random
import shutil
import subprocess
import time
from datetime import timedelta

# ==========================================================
# PARTIAL FACESWAP VIDEO GENERATOR
#
# Research Project
#
# Generates:
#   • Partial FaceSwap videos
#   • Ground-truth CSV
#
# Features
# --------
# ✓ Resume if interrupted
# ✓ Skip completed videos
# ✓ CUDA -> CPU fallback
# ✓ Automatic retries
# ✓ Validation
# ✓ Temporary cleanup
# ✓ CSV regeneration
#
# ==========================================================

random.seed(42)

# ==========================================================
# PATHS
# ==========================================================

REAL_VIDEO_DIR = "dataset/dataset/roop_input_v2"

DONOR_DIR = "dataset/dataset/donor_faces"

OUTPUT_DIR = "dataset/dataset/fake_partial_v2"

TMP_DIR = "dataset/dataset/tmp"

GROUND_TRUTH_CSV = "dataset/dataset/ground_truth_partial_v2.csv"

MAX_VIDEOS = None

MIN_VIDEO_DURATION = 40

RETRY_PER_VIDEO = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ==========================================================
# LOAD VIDEOS
# ==========================================================

real_videos = sorted(

    glob.glob(os.path.join(REAL_VIDEO_DIR, "*.mp4"))

)

donor_images = sorted(

    glob.glob(os.path.join(DONOR_DIR, "*.*"))

)

if MAX_VIDEOS is not None:
    real_videos = real_videos[:MAX_VIDEOS]

print("=" * 70)
print("PARTIAL FACE SWAP GENERATOR")
print("=" * 70)

print("Real videos :", len(real_videos))
print("Donor faces :", len(donor_images))

print()

# ==========================================================
# CHECK REQUIREMENTS
# ==========================================================

if len(real_videos) == 0:
    raise Exception(
        "No real videos found inside:\n"
        + REAL_VIDEO_DIR
    )

if len(donor_images) == 0:
    raise Exception(
        "No donor images found inside:\n"
        + DONOR_DIR
    )

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def run_command(command):

    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True

    )

    return result

# ----------------------------------------------------------

def ffmpeg(args):

    command = ["ffmpeg", "-y"] + args

    result = run_command(command)

    return result.returncode == 0

# ----------------------------------------------------------

def get_duration(video):

    command = [

        "ffprobe",

        "-v",

        "quiet",

        "-print_format",

        "json",

        "-show_format",

        video

    ]

    result = run_command(command)

    try:

        info = json.loads(result.stdout)

        return float(info["format"]["duration"])

    except:

        return 0.0

# ----------------------------------------------------------

def remove_file(path):

    if os.path.exists(path):

        try:

            os.remove(path)

        except:

            pass

# ----------------------------------------------------------

def cleanup(files):

    for f in files:

        remove_file(f)

# ----------------------------------------------------------

def is_good_video(path):

    if not os.path.exists(path):
        return False

    if os.path.getsize(path) < 100000:
        return False

    d = get_duration(path)

    if d <= 1:
        return False

    return True

# ----------------------------------------------------------

def run_roop(provider, source, target, output):

    command = [

        sys.executable,

        "roop/run.py",

        "-s",

        os.path.abspath(source),

        "-t",

        os.path.abspath(target),

        "-o",

        os.path.abspath(output),

        "--execution-provider",

        provider,

        "--skip-audio"

    ]

    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True

    )

    if result.returncode != 0:

        return False

    return is_good_video(output)

# ----------------------------------------------------------

def face_swap(source, target, output):

    print("Trying CUDA...")

    ok = run_roop(

        "cuda",

        source,

        target,

        output

    )

    if ok:

        print("CUDA success")

        return True

    print("CUDA failed")

    print("Trying CPU...")

    ok = run_roop(

        "cpu",

        source,

        target,

        output

    )

    if ok:

        print("CPU success")

        return True

    print("CPU failed")

    return False
# ==========================================================
# START PROCESSING
# ==========================================================

rows = []

success_count = 0

failed_count = 0

skipped_count = 0

failed_videos = []

START_TIME = time.time()

# ==========================================================
# LOOP THROUGH VIDEOS
# ==========================================================

for index, video_path in enumerate(real_videos, start=1):

    video_name = os.path.basename(video_path).replace(".mp4", "")

    output_video = os.path.join(

        OUTPUT_DIR,

        video_name + "_partial.mp4"

    )

    print("\n")
    print("=" * 70)
    print(f"[{index}/{len(real_videos)}]")
    print(video_name)
    print("=" * 70)

    # ------------------------------------------------------
    # Skip completed videos
    # ------------------------------------------------------

    if is_good_video(output_video):

        print("Already generated.")

        skipped_count += 1

        duration = get_duration(output_video)

        language = "sinhala"

        if "tamil" in video_name.lower():

            language = "tamil"

        rows.append({

            "video_id": video_name + "_partial",

            "video_path": os.path.abspath(output_video),

            "language": language,

            "total_duration": round(duration,2),

            "fake_start_sec": -1,

            "fake_end_sec": -1,

            "fake_duration": -1,

            "donor_face":"previously_generated"

        })

        continue

    # ------------------------------------------------------
    # Video duration
    # ------------------------------------------------------

    duration = get_duration(video_path)

    print("Duration :", duration)

    if duration < MIN_VIDEO_DURATION:

        print("Video too short.")

        skipped_count += 1

        continue

    language = "sinhala"

    if "tamil" in video_name.lower():

        language = "tamil"

    success = False

    # ======================================================
    # RETRIES
    # ======================================================

    for attempt in range(RETRY_PER_VIDEO):

        print()

        print(f"Attempt {attempt+1}/{RETRY_PER_VIDEO}")

        donor = random.choice(donor_images)

        fake_length = random.choice([8,9,10,11,12])

        # ----------------------------------------------

        if attempt == 0:

            fake_start = round(

                random.uniform(

                    duration*0.20,

                    duration*0.40

                ),

                2

            )

        elif attempt == 1:

            fake_start = round(

                random.uniform(

                    duration*0.40,

                    duration*0.60

                ),

                2

            )

        else:

            fake_start = round(

                random.uniform(

                    duration*0.60,

                    duration*0.75

                ),

                2

            )

        fake_end = round(

            min(

                fake_start + fake_length,

                duration - 3

            ),

            2

        )

        print("Fake region :", fake_start, "->", fake_end)

        print("Fake length :", fake_end-fake_start)

        print("Donor :", os.path.basename(donor))

        before_video = os.path.join(

            TMP_DIR,

            video_name + "_before.mp4"

        )

        middle_video = os.path.join(

            TMP_DIR,

            video_name + "_middle.mp4"

        )

        swapped_video = os.path.join(

            TMP_DIR,

            video_name + "_swap.mp4"

        )

        after_video = os.path.join(

            TMP_DIR,

            video_name + "_after.mp4"

        )

        concat_txt = os.path.join(

            TMP_DIR,

            video_name + ".txt"

        )

        cleanup([

            before_video,

            middle_video,

            swapped_video,

            after_video,

            concat_txt

        ])

        # ----------------------------------------------
        # Extract BEFORE
        # ----------------------------------------------

        ok = ffmpeg([

            "-i",

            video_path,

            "-t",

            str(fake_start),

            "-c:v",

            "libx264",

            "-c:a",

            "aac",

            before_video

        ])

        if not ok:

            print("Failed extracting BEFORE.")

            continue

        # ----------------------------------------------
        # Extract MIDDLE
        # ----------------------------------------------

        ok = ffmpeg([

            "-i",

            video_path,

            "-ss",

            str(fake_start),

            "-to",

            str(fake_end),

            "-c:v",

            "libx264",

            "-c:a",

            "aac",

            middle_video

        ])

        if not is_good_video(middle_video):

            print("Middle extraction failed.")

            continue

        # ----------------------------------------------
        # Extract AFTER
        # ----------------------------------------------

        ok = ffmpeg([

            "-i",

            video_path,

            "-ss",

            str(fake_end),

            "-c:v",

            "libx264",

            "-c:a",

            "aac",

            after_video

        ])

        if not ok:

            print("Failed extracting AFTER.")

            cleanup([

                before_video,

                middle_video,

                after_video

            ])

            continue

        # ----------------------------------------------
        # Face Swap
        # ----------------------------------------------

        print()

        print("Running Roop...")

        ok = face_swap(

            donor,

            middle_video,

            swapped_video

        )

        if not ok:

            print("Face swap failed.")

            cleanup([

                before_video,

                middle_video,

                swapped_video,

                after_video

            ])

            continue

        print("Face swap completed.")

        # ----------------------------------------------
        # Validate swapped video
        # ----------------------------------------------

        original_middle_duration = get_duration(

            middle_video

        )

        swapped_duration = get_duration(

            swapped_video

        )

        if abs(

            original_middle_duration -

            swapped_duration

        ) > 1:

            print(

                "Duration mismatch."

            )

            cleanup([

                before_video,

                middle_video,

                swapped_video,

                after_video

            ])

            continue

        # ----------------------------------------------
        # Create concat list
        # ----------------------------------------------

        with open(

            concat_txt,

            "w",

            encoding="utf-8"

        ) as f:

            f.write(

                f"file '{os.path.abspath(before_video)}'\n"

            )

            f.write(

                f"file '{os.path.abspath(swapped_video)}'\n"

            )

            f.write(

                f"file '{os.path.abspath(after_video)}'\n"

            )

        # ----------------------------------------------
        # Merge video
        # ----------------------------------------------

        print()

        print("Merging segments...")

        ok = ffmpeg([

            "-f",

            "concat",

            "-safe",

            "0",

            "-i",

            concat_txt,

            "-c:v",

            "libx264",

            "-c:a",

            "aac",

            "-crf",

            "23",

            output_video

        ])

        cleanup([

            before_video,

            middle_video,

            swapped_video,

            after_video,

            concat_txt

        ])

        if not ok:

            print("Merge failed.")

            continue

        if not is_good_video(

            output_video

        ):

            print(

                "Merged output invalid."

            )

            remove_file(

                output_video

            )

            continue

        print()

        print("SUCCESS")

        print(output_video)

        success = True

        success_count += 1

        rows.append({

            "video_id":

                video_name + "_partial",

            "video_path":

                os.path.abspath(

                    output_video

                ),

            "language":

                language,

            "total_duration":

                round(duration,2),

            "fake_start_sec":

                fake_start,

            "fake_end_sec":

                fake_end,

            "fake_duration":

                round(

                    fake_end-fake_start,

                    2

                ),

            "donor_face":

                os.path.basename(

                    donor

                )

        })

        break

    # ------------------------------------------
    # Failed after retries
    # ------------------------------------------

    if not success:

        print()

        print("FAILED")

        failed_count += 1

        failed_videos.append(

            video_name

        )

# ==========================================================
# REBUILD GROUND TRUTH CSV
# ==========================================================

print("\n")
print("=" * 70)
print("REBUILDING GROUND TRUTH CSV")
print("=" * 70)

existing_videos = sorted(

    glob.glob(

        os.path.join(

            OUTPUT_DIR,

            "*_partial.mp4"

        )

    )

)

print(f"Found {len(existing_videos)} generated videos.")

existing_ids = {

    os.path.basename(v).replace("_partial.mp4", ""): v

    for v in existing_videos

}

csv_rows = []

# ----------------------------------------------------------
# Keep metadata collected during this run
# ----------------------------------------------------------

metadata_lookup = {

    row["video_id"].replace("_partial",""): row

    for row in rows

}

# ----------------------------------------------------------
# Rebuild CSV
# ----------------------------------------------------------

for video_id, video_path in existing_ids.items():

    if video_id in metadata_lookup:

        csv_rows.append(

            metadata_lookup[video_id]

        )

        continue

    duration = get_duration(video_path)

    language = "sinhala"

    if "tamil" in video_id.lower():

        language = "tamil"

    csv_rows.append({

        "video_id":

            video_id + "_partial",

        "video_path":

            os.path.abspath(video_path),

        "language":

            language,

        "total_duration":

            round(duration,2),

        "fake_start_sec":

            -1,

        "fake_end_sec":

            -1,

        "fake_duration":

            -1,

        "donor_face":

            "unknown"

    })

print(f"CSV Entries : {len(csv_rows)}")

# ----------------------------------------------------------
# Save CSV
# ----------------------------------------------------------

if len(csv_rows) > 0:

    csv_rows = sorted(

        csv_rows,

        key=lambda x: x["video_id"]

    )

    with open(

        GROUND_TRUTH_CSV,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[

                "video_id",

                "video_path",

                "language",

                "total_duration",

                "fake_start_sec",

                "fake_end_sec",

                "fake_duration",

                "donor_face"

            ]

        )

        writer.writeheader()

        writer.writerows(

            csv_rows

        )

    print()

    print("Ground truth CSV saved.")

    print(GROUND_TRUTH_CSV)

else:

    print()

    print("No videos generated.")

    print("CSV not created.")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

END_TIME = time.time()

TOTAL_TIME = END_TIME - START_TIME

TOTAL_PROCESSED = success_count + failed_count + skipped_count

print()
print("=" * 70)
print("PARTIAL FACE SWAP GENERATION COMPLETED")
print("=" * 70)

print(f"Total real videos      : {len(real_videos)}")
print(f"Successfully generated : {success_count}")
print(f"Already existed        : {skipped_count}")
print(f"Failed                 : {failed_count}")

print()

if TOTAL_PROCESSED > 0:

    success_rate = (success_count + skipped_count) / TOTAL_PROCESSED * 100

    print(f"Overall completion : {success_rate:.2f}%")

print()

print(f"Output folder : {os.path.abspath(OUTPUT_DIR)}")
print(f"Ground Truth  : {os.path.abspath(GROUND_TRUTH_CSV)}")

print()

print(
    "Total running time :",
    str(
        timedelta(
            seconds=int(TOTAL_TIME)
        )
    )
)

print("=" * 70)

# ==========================================================
# FAILED VIDEOS
# ==========================================================

if len(failed_videos) > 0:

    print()

    print("=" * 70)
    print("FAILED VIDEOS")
    print("=" * 70)

    for v in failed_videos:

        print(v)

    print("=" * 70)

else:

    print()

    print("No failed videos.")

# ==========================================================
# VERIFY OUTPUTS
# ==========================================================

print()

print("=" * 70)
print("VERIFYING GENERATED VIDEOS")
print("=" * 70)

generated = sorted(

    glob.glob(

        os.path.join(

            OUTPUT_DIR,

            "*_partial.mp4"

        )

    )

)

good = 0

bad = 0

for video in generated:

    if is_good_video(video):

        good += 1

    else:

        bad += 1

        print("Corrupted :", os.path.basename(video))

print()

print("Valid videos :", good)

print("Corrupted    :", bad)

print()

if os.path.exists(GROUND_TRUTH_CSV):

    with open(GROUND_TRUTH_CSV, "r", encoding="utf-8") as f:

        csv_rows = list(csv.reader(f))

    print(f"CSV rows (including header): {len(csv_rows)}")

print()

print("=" * 70)
print("SCRIPT FINISHED SUCCESSFULLY")
print("=" * 70)