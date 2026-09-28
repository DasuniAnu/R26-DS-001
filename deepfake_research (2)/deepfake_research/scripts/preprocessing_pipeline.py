# scripts/preprocessing_pipeline.py
# Fixed version — no MTCNN dependency, correct splits
#do not use MTCNN at all. 
# The dependency conflicts you just created by downgrading numpy and typing-extensions
#  will break other things in your environment. 
# The Haar cascade version in the script I gave you works fine and has no extra dependencies. 
# The script I gave you already uses Haar cascade — do not modify it to use MTCNN.

import cv2
import numpy as np
import os
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────
OUTPUT_DIR   = "dataset/processed"
CSV_PATH     = "dataset/unified_dataset.csv"
IMG_SIZE     = 224
FRAME_SKIP   = 5
MAX_FRAMES   = 20   # max frames per video

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Face detector (no TensorFlow needed) ─────────────
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# ── Load and fix CSV ──────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['language'] = df['language'].str.lower().str.strip()
df['label']    = df['label'].astype(int)

print(f"Total videos  : {len(df)}")
print(f"Real (0)      : {sum(df['label']==0)}")
print(f"Fake (1)      : {sum(df['label']==1)}")
print(f"Sinhala       : {sum(df['language']=='sinhala')}")
print(f"Tamil         : {sum(df['language']=='tamil')}")
print()

# ── Assign splits BEFORE processing ──────────────────
# Stratified 70/15/15
idx = df.index.tolist()
y   = df['label'].tolist()

idx_tr, idx_te, _, _ = train_test_split(
    idx, y, test_size=0.15, stratify=y, random_state=42)
y_tr = [y[i] for i in idx_tr]

idx_tr, idx_va, _, _ = train_test_split(
    idx_tr, y_tr, test_size=0.176,
    stratify=y_tr, random_state=42)

df['split'] = 'train'
df.loc[idx_va, 'split'] = 'val'
df.loc[idx_te, 'split'] = 'test'

print("Split counts:")
print(df['split'].value_counts())
print()

# ── Face extraction helper ────────────────────────────
def extract_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    faces = sorted(faces,
                   key=lambda x: x[2]*x[3],
                   reverse=True)
    x, y, w, h = faces[0]
    x, y  = max(0, x), max(0, y)
    x2    = min(frame.shape[1], x + w)
    y2    = min(frame.shape[0], y + h)
    crop  = frame[y:y2, x:x2]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


# ── Main processing loop ──────────────────────────────
X_train, y_train = [], []
X_val,   y_val   = [], []
X_test,  y_test  = [], []

skipped  = []
metadata = []

for idx2, row in df.iterrows():
    path    = row['video_path'].replace('/', os.sep).replace('\\', os.sep)
    vid_id  = row['video_id']
    label   = int(row['label'])
    split   = row['split']

    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        skipped.append(vid_id)
        continue

    cap      = cv2.VideoCapture(path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fidx     = 0
    frames   = []
    times    = []

    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx % FRAME_SKIP == 0:
            face = extract_face(frame)
            if face is not None:
                frames.append(face)
                times.append(round(fidx / fps, 3))
        fidx += 1
    cap.release()

    if len(frames) == 0:
        print(f"  NO FACES: {vid_id}")
        skipped.append(vid_id)
        continue

    # Add to correct split
    for face in frames:
        if split == 'train':
            X_train.append(face)
            y_train.append(label)
        elif split == 'val':
            X_val.append(face)
            y_val.append(label)
        else:
            X_test.append(face)
            y_test.append(label)

    metadata.append({
        'video_id':   vid_id,
        'label':      label,
        'split':      split,
        'language':   row['language'],
        'fake_type':  row.get('fake_type', ''),
        'num_frames': len(frames),
        'fake_start': float(row['fake_start_sec']) if str(row['fake_start_sec']) not in ['0','0.0',''] else None,
        'fake_end':   float(row['fake_end_sec'])   if str(row['fake_end_sec'])   not in ['0','0.0',''] else None,
        'frame_times': times
    })

    progress = idx2 + 1
    print(f"  [{progress}/{len(df)}] {vid_id} → {len(frames)} frames ({split})")

# ── Convert to arrays ─────────────────────────────────
print("\nBuilding arrays...")
X_train = np.array(X_train, dtype=np.float32)
X_val   = np.array(X_val,   dtype=np.float32)
X_test  = np.array(X_test,  dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
y_val   = np.array(y_val,   dtype=np.int32)
y_test  = np.array(y_test,  dtype=np.int32)

# ── Save ──────────────────────────────────────────────
print("Saving...")
np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/X_val.npy",   X_val)
np.save(f"{OUTPUT_DIR}/X_test.npy",  X_test)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/y_val.npy",   y_val)
np.save(f"{OUTPUT_DIR}/y_test.npy",  y_test)

with open(f"{OUTPUT_DIR}/metadata.json", "w",
          encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# Also save updated CSV with split column
df.to_csv("dataset/unified_dataset_splits.csv", index=False)

# ── Report ────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"PREPROCESSING COMPLETE")
print(f"{'='*50}")
print(f"X_train : {X_train.shape}  real={sum(y_train==0)}  fake={sum(y_train==1)}")
print(f"X_val   : {X_val.shape}    real={sum(y_val==0)}    fake={sum(y_val==1)}")
print(f"X_test  : {X_test.shape}   real={sum(y_test==0)}   fake={sum(y_test==1)}")
print(f"Skipped : {len(skipped)}")
print(f"Balance : {sum(y_train==1)/len(y_train)*100:.1f}% fake in train")
print(f"\nSaved to: {OUTPUT_DIR}/")
print(f"Upload X_train, X_val, X_test, y_train, y_val, y_test to Google Drive for training")