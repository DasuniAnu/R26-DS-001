"""
Fill empty values in unified_dataset.csv with 0
This prevents NaN errors during preprocessing
"""

import pandas as pd

csv_path = "dataset/unified_dataset.csv"

# Load
df = pd.read_csv(csv_path)

# Fill empty values with 0 for real videos
df['fake_start_sec'] = df['fake_start_sec'].fillna(0)
df['fake_end_sec'] = df['fake_end_sec'].fillna(0)
df['fake_duration'] = df['fake_duration'].fillna(0)

# Verify
print("Before fix - empty values:")
print(df[df['label']==0][['video_id', 'label', 'fake_start_sec', 'fake_end_sec']].head(3))

print("\nAfter fix - all filled:")
print(df[['video_id', 'label', 'fake_type', 'fake_start_sec', 'fake_end_sec', 'fake_duration']].head(10))

# Save
df.to_csv(csv_path, index=False)
print(f"\n✅ Updated {csv_path}")

# Summary
print(f"\nDataset Summary:")
print(f"Total videos: {len(df)}")
print(f"Real (label=0): {len(df[df['label']==0])}")
print(f"Fake (label=1): {len(df[df['label']==1])}")
print(f"  - Full fake: {len(df[(df['label']==1) & (df['fake_type']=='full')])}")
print(f"  - Partial fake: {len(df[(df['label']==1) & (df['fake_type']=='partial')])}")
