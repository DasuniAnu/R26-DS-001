import subprocess
import os
import glob
import random
import sys
import time

# Load videos
real_videos = sorted(glob.glob("dataset/roop_input/*.mp4"))


#real_videos = real_videos[:2]

# Load donor faces
donor_faces = sorted(
    glob.glob("dataset/donor_faces/*.jpg") +
    glob.glob("dataset/donor_faces/*.JPG")
)

if not donor_faces:
    print("ERROR: No donor faces found")
    exit()

if not real_videos:
    print("ERROR: No videos found")
    exit()

print(f"Videos      : {len(real_videos)}")
print(f"Donors      : {len(donor_faces)}")

out_dir = os.path.abspath("dataset/fake_full")
os.makedirs(out_dir, exist_ok=True)

random.seed(42)

for video_path in real_videos:
    vid_id = os.path.basename(video_path).replace(".mp4", "")
    donor  = random.choice(donor_faces)

    abs_video  = os.path.abspath(video_path)
    abs_donor  = os.path.abspath(donor)
    abs_output = os.path.abspath(
        os.path.join(out_dir, vid_id + "_fake.mp4")
    )

    # SKIP if already exists (IMPORTANT)
    if os.path.exists(abs_output) and os.path.getsize(abs_output) > 100000:
        print(f" Already done: {vid_id} — skipping")
        continue

    print(f"\n{'='*40}")
    print(f"Video  : {vid_id}")
    print(f"Donor  : {os.path.basename(donor)}")
    print(f"Output : {abs_output}")
    print(f"{'='*40}")

    result = subprocess.run([
        sys.executable, "roop/run.py",
        "--target",             abs_video,
        "--source",             abs_donor,
        "--output",             abs_output,
        "--frame-processor",    "face_swapper",
        "--execution-provider", "cpu",
        "--many-faces",
        "--skip-audio"
    ])

    time.sleep(2)

    #  CHECK RESULT
    if result.returncode != 0:
        print(f" Roop crashed for {vid_id}")

    if os.path.exists(abs_output) and os.path.getsize(abs_output) > 100000:
        size_mb = os.path.getsize(abs_output) / 1e6
        print(f" SUCCESS — {size_mb:.1f} MB")
    else:
        print(f" FAILED — {vid_id}")