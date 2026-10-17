import os
import glob
import random
import subprocess
import sys

# -----------------------------
# PATHS
# -----------------------------
videos = sorted(glob.glob("dataset/roop_input_v2/*.mp4"))
donors = sorted(glob.glob("dataset/donor_faces/*.*"))

out_dir = "dataset/fake_full_v2"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# SETTINGS
# -----------------------------
random.seed(42)

done = 0
failed = []

print("Videos :", len(videos))
print("Donors :", len(donors))

# -----------------------------
# LOOP
# -----------------------------
for v in videos:

    vid = os.path.basename(v).replace(".mp4", "")
    donor = random.choice(donors)

    output = os.path.join(out_dir, f"{vid}_fake.mp4")

    # Skip finished
    if os.path.exists(output) and os.path.getsize(output) > 500000:
        print(f"Already done: {vid}")
        done += 1
        continue

    print("\n" + "="*50)
    print("Processing:", vid)
    print("Donor     :", os.path.basename(donor))
    print("="*50)

    result = subprocess.run(
[
    sys.executable,
    "roop/run.py",
    "-s", donor,
    "-t", v,
    "-o", output,
    "--many-faces",
    "--skip-audio",
    "--execution-provider", "cpu"
]
)

    if result.returncode != 0:
        print("FAILED ERROR:")
        print(result.stderr)

    if os.path.exists(output) and os.path.getsize(output) > 500000:
        print("SUCCESS:", vid)
        done += 1
    else:
        print("FAILED:", vid)
        failed.append(vid)

print("\n" + "="*60)
print("DONE   :", done)
print("FAILED :", len(failed))
print(failed)
print("="*60)