# scripts/preprocess.py
# FINAL UPDATED VERSION
# 3-Class Deepfake Dataset Preprocessing
# Supports:
# real / full_fake / partial_fake
# Sinhala + Tamil
# Face extraction + train/val/test .npy output

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CSV_PATH = "dataset/final_labels.csv"
OUTPUT_DIR = "dataset/processed"

IMG_SIZE = 224
FRAMES_PER_VIDEO = 20
FRAME_SKIP = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("Loaded dataset rows:", len(df))
print("=" * 60)

# --------------------------------------------------
# FACE DETECTOR
# --------------------------------------------------
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# OUTPUT ARRAYS
# --------------------------------------------------
X_train, y_train = [], []
X_val, y_val     = [], []
X_test, y_test   = [], []

# --------------------------------------------------
# FACE EXTRACTION FUNCTION
# --------------------------------------------------
def extract_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # largest face
    faces = sorted(
        faces,
        key=lambda x: x[2] * x[3],
        reverse=True
    )

    x, y, w, h = faces[0]

    face = frame[y:y+h, x:x+w]

    return face

# --------------------------------------------------
# PROCESS VIDEOS
# --------------------------------------------------
for _, row in tqdm(df.iterrows(), total=len(df)):

    video_path = row["video_path"]
    label      = int(row["label"])
    split      = row["split"]

    if not os.path.exists(video_path):
        print("Missing file:", video_path)
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open:", video_path)
        continue

    frame_id = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        # skip frames
        if frame_id % FRAME_SKIP != 0:
            continue

        face = extract_face(frame)

        if face is None:
            continue

        # resize
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        # BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # normalize
        face = face.astype(np.float32) / 255.0

        # append
        if split == "train":
            X_train.append(face)
            y_train.append(label)

        elif split == "val":
            X_val.append(face)
            y_val.append(label)

        else:
            X_test.append(face)
            y_test.append(label)

        saved += 1

        if saved >= FRAMES_PER_VIDEO:
            break

    cap.release()

# --------------------------------------------------
# TO NUMPY
# --------------------------------------------------
X_train = np.array(X_train, dtype=np.float32)
X_val   = np.array(X_val, dtype=np.float32)
X_test  = np.array(X_test, dtype=np.float32)

y_train = np.array(y_train, dtype=np.int32)
y_val   = np.array(y_val, dtype=np.int32)
y_test  = np.array(y_test, dtype=np.int32)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)

np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n" + "=" * 60)
print("PREPROCESS COMPLETE")
print("=" * 60)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("\nSaved to:", OUTPUT_DIR)
print("=" * 60)