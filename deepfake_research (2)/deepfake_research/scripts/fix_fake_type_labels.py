"""
Fix fake_type labels: 'none' -> 'real' for real videos
"""

import pandas as pd

csv_path = "dataset/unified_dataset.csv"

# Load
df = pd.read_csv(csv_path)

# Replace 'none' with 'real' for real videos
df.loc[df['label'] == 0, 'fake_type'] = 'real'

print("Updated fake_type labels:")
print(df['fake_type'].value_counts())

print(f"\nSample real videos:")
print(df[df['label']==0][['video_id', 'label', 'fake_type', 'fake_start_sec']].head(5))

print(f"\nSample full fake:")
print(df[(df['label']==1) & (df['fake_type']=='full')][['video_id', 'label', 'fake_type', 'fake_start_sec', 'fake_end_sec']].head(3))

print(f"\nSample partial fake:")
print(df[(df['label']==1) & (df['fake_type']=='partial')][['video_id', 'label', 'fake_type', 'fake_start_sec', 'fake_end_sec']].head(3))

# Save
df.to_csv(csv_path, index=False)
print(f"\n✅ Updated {csv_path}")

# Final summary
print(f"\n📊 FINAL DATASET SUMMARY:")
print(f"Total videos: {len(df)}")
print(f"Real (fake_type='real'): {len(df[df['fake_type']=='real'])}")
print(f"Full Fake (fake_type='full'): {len(df[df['fake_type']=='full'])}")
print(f"Partial Fake (fake_type='partial'): {len(df[df['fake_type']=='partial'])}")
