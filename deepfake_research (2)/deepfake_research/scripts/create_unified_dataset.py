# Create this file: scripts/create_unified_dataset.py

import os
import csv
import pandas as pd
from pathlib import Path
import cv2

def get_video_duration(video_path):
    """Get video duration in seconds"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps > 0:
            return frame_count / fps
        return 0
    except:
        return 0

# Paths
REAL_SINHALA = "dataset/real_sinhala"
REAL_TAMIL = "dataset/real_tamil"
FAKE_FULL = "dataset/fake_full_v2"
FAKE_PARTIAL = "dataset/fake_partial_v2"
PARTIAL_CSV = "dataset/ground_truth_partial_v2.csv"

output_csv = "dataset/unified_dataset.csv"

data = []

# 1. REAL VIDEOS (label=0, no fake_start/fake_end)
print("Processing REAL Sinhala videos...")
for video in os.listdir(REAL_SINHALA):
    if video.endswith(('.mp4', '.avi', '.mov')):
        path = os.path.join(REAL_SINHALA, video)
        data.append({
            'video_id': Path(video).stem,
            'video_path': path,
            'language': 'Sinhala',
            'label': 0,  # Real
            'fake_type': 'none',
            'fake_start_sec': None,
            'fake_end_sec': None,
            'fake_duration': None
        })

print("Processing REAL Tamil videos...")
for video in os.listdir(REAL_TAMIL):
    if video.endswith(('.mp4', '.avi', '.mov')):
        path = os.path.join(REAL_TAMIL, video)
        data.append({
            'video_id': Path(video).stem,
            'video_path': path,
            'language': 'Tamil',
            'label': 0,  # Real
            'fake_type': 'none',
            'fake_start_sec': None,
            'fake_end_sec': None,
            'fake_duration': None
        })

# 2. FULL FAKE VIDEOS (label=1, fake_start=0, fake_end=video_duration)
print("Processing FULL FAKE videos...")
for video in os.listdir(FAKE_FULL):
    if video.endswith(('.mp4', '.avi', '.mov')):
        path = os.path.join(FAKE_FULL, video)
        # Extract language from filename
        lang = 'Sinhala' if 'sin' in video.lower() else 'Tamil'
        duration = get_video_duration(path)
        data.append({
            'video_id': Path(video).stem,
            'video_path': path,
            'language': lang,
            'label': 1,  # Fake
            'fake_type': 'full',
            'fake_start_sec': 0,  # Entire video is fake
            'fake_end_sec': round(duration, 2),  # Entire duration
            'fake_duration': round(duration, 2)
        })

# 3. PARTIAL FAKE VIDEOS (from existing CSV)
print("Processing PARTIAL FAKE videos...")
partial_df = pd.read_csv(PARTIAL_CSV)
for _, row in partial_df.iterrows():
    data.append({
        'video_id': row['video_id'],
        'video_path': row['video_path'],
        'language': row['language'],
        'label': 1,  # Fake
        'fake_type': 'partial',
        'fake_start_sec': row['fake_start_sec'],
        'fake_end_sec': row['fake_end_sec'],
        'fake_duration': row['fake_duration']
    })

# Create DataFrame & save
df = pd.DataFrame(data)
print(f"\n✅ Total videos: {len(df)}")
print(f"  - Real: {len(df[df['label']==0])}")
print(f"  - Full Fake: {len(df[(df['label']==1) & (df['fake_type']=='full')])}")
print(f"  - Partial Fake: {len(df[(df['label']==1) & (df['fake_type']=='partial')])}")

df.to_csv(output_csv, index=False)
print(f"✅ Saved to: {output_csv}")

# Verify
print(f"\nDataset Summary:")
print(df[['language', 'label', 'fake_type']].value_counts())
print(f"\nFirst few rows:")
print(df[['video_id', 'language', 'label', 'fake_type', 'fake_start_sec', 'fake_end_sec']].head(10))