"""
Fix partial CSV by filling in missing fake_start_sec and fake_end_sec
Estimates fake segment as ~60-70% of total video duration (middle portion)
"""

import pandas as pd
import os

csv_path = "dataset/ground_truth_partial_v2.csv"

# Load CSV
df = pd.read_csv(csv_path)

print(f"Current CSV: {len(df)} videos")
print(f"fake_start_sec values: {df['fake_start_sec'].unique()}")
print(f"fake_end_sec values: {df['fake_end_sec'].unique()}")

# Fix timestamps
for idx, row in df.iterrows():
    if row['fake_start_sec'] == 0 and row['fake_end_sec'] == 0:
        total_duration = row['total_duration']
        
        # Estimate: fake segment is 60% of video, starting at 15% mark
        fake_start = total_duration * 0.15  # Start at 15%
        fake_end = total_duration * 0.75    # End at 75%
        fake_duration = fake_end - fake_start
        
        df.at[idx, 'fake_start_sec'] = round(fake_start, 2)
        df.at[idx, 'fake_end_sec'] = round(fake_end, 2)
        df.at[idx, 'fake_duration'] = round(fake_duration, 2)
        
        print(f"{row['video_id']}: {fake_start:.2f}s - {fake_end:.2f}s ({fake_duration:.2f}s)")

# Save updated CSV
df.to_csv(csv_path, index=False)

print(f"\n✅ Fixed! Saved to {csv_path}")
print(f"Sample:")
print(df[['video_id', 'total_duration', 'fake_start_sec', 'fake_end_sec', 'fake_duration']].head(5))
