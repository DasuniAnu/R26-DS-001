#!/usr/bin/env python3
"""
DATASET VERIFICATION SCRIPT
Checks: Video count, CSV completeness, Missing files
"""

import os
import glob
import csv
import json

FAKE_PARTIAL_DIR = "dataset/fake_partial_v2"
CSV_FILE = "dataset/ground_truth_partial_v2.csv"

print("=" * 60)
print("DATASET VERIFICATION REPORT")
print("=" * 60)

# Check videos
videos = sorted(glob.glob(f"{FAKE_PARTIAL_DIR}/*.mp4"))
print(f"\n✓ Partial videos found: {len(videos)}")

# Check CSV
csv_rows = []
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
    print(f"✓ CSV rows: {len(csv_rows)}")
else:
    print(f"✗ CSV NOT FOUND: {CSV_FILE}")

# Check for mismatches
print("\n--- MISSING MAPPINGS ---")
video_ids_in_csv = {row['video_id'] for row in csv_rows}
missing = []

for video_path in videos:
    video_name = os.path.basename(video_path).replace('.mp4', '')
    if video_name not in video_ids_in_csv:
        missing.append(video_name)
        print(f"  ✗ {video_name} - NO CSV ENTRY")

print(f"\nTotal missing: {len(missing)} / {len(videos)}")

# Verify file paths
print("\n--- PATH VERIFICATION ---")
broken_paths = []
for row in csv_rows:
    if 'video_path' in row and row['video_path']:
        if not os.path.exists(row['video_path']):
            broken_paths.append(row['video_id'])
            print(f"  ✗ {row['video_id']} - PATH BROKEN")

print(f"Total broken: {len(broken_paths)} / {len(csv_rows)}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
total_ok = len(videos) - len(missing)
print(f"Ready for preprocessing: {total_ok}/{len(videos)} videos")

if len(missing) == 0 and len(broken_paths) == 0:
    print("✅ DATASET READY - Proceed to preprocessing")
else:
    print("⚠️  DATASET INCOMPLETE - Fix issues above first")

# Save report
report = {
    "total_videos": len(videos),
    "csv_mapped": len(csv_rows),
    "missing_csv_entries": missing,
    "broken_paths": broken_paths,
    "ready_for_preprocessing": total_ok
}

with open("dataset_verification_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nReport saved: dataset_verification_report.json")
