import subprocess
import glob
import os

# Source: your already-trimmed 10-min videos
source_videos = glob.glob("dataset/trimmed/*.mp4")

out_dir = "dataset/roop_input"
os.makedirs(out_dir, exist_ok=True)

for path in source_videos:
    vid_id = os.path.basename(path)
    output = f"{out_dir}/{vid_id}"

    if os.path.exists(output):
        print(f"Already done: {vid_id}")
        continue

    # Only 2 minutes for Roop — enough for 200+ fake windows
    subprocess.run([
        "ffmpeg", "-y",
        "-i", path,
        "-t", "120",
        "-c:v", "libx264",
        "-crf", "28",        # Higher compression = smaller file = faster Roop
        "-vf", "scale=480:-2",  # Downscale to 480p — your research uses 480p anyway
        "-c:a", "aac",
        output
    ], capture_output=True)

    if os.path.exists(output):
        size = os.path.getsize(output) / 1e6
        print(f"Done: {vid_id} → {size:.0f} MB")
    else:
        print(f"FAILED: {vid_id}")

print(f"\nAll videos prepared for Roop.")
print(f"Saved to: {out_dir}/")