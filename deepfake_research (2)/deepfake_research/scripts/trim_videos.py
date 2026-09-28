import subprocess
import glob
import os

# Collect all real videos
real_videos = (
    glob.glob("dataset/real_sinhala/*.mp4") +
    glob.glob("dataset/real_tamil/*.mp4")
)

# Output folder
trimmed_dir = "dataset/trimmed"
os.makedirs(trimmed_dir, exist_ok=True)

#  Process each video
for path in real_videos:
    vid_id = os.path.basename(path).replace(".mp4", "")
    output = os.path.join(trimmed_dir, vid_id + "_trim.mp4")

    # ⏭Skip if already trimmed
    if os.path.exists(output):
        print(f"Already trimmed: {vid_id}")
        continue

    print(f" Trimming: {vid_id}")

    #  Run FFmpeg (first 10 minutes)
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", path,
        "-t", "600",                 # 10 minutes
        "-c:v", "libx264",
        "-preset", "fast",           # faster processing
        "-c:a", "aac",
        output
    ], capture_output=True, text=True)

    #  Handle errors
    if result.returncode != 0:
        print(f" FAILED: {vid_id}")
        print(result.stderr)
        continue

    # Success check
    if os.path.exists(output):
        size = os.path.getsize(output) / 1e6
        print(f" Trimmed: {vid_id} → {size:.0f} MB")
    else:
        print(f" FAILED: {vid_id}")

# Final message
print("\n All videos processed.")
print(f" Saved to: {trimmed_dir}/")