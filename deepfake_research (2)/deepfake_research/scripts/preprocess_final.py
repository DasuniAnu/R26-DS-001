
import cv2
import numpy as np
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "dataset/processed_final"
CSV_PATH   = "dataset/unified_dataset_splits.csv"
IMG_SIZE   = 224
FRAME_SKIP = 5
MAX_FRAMES = 15   # HARD CAP — max 15 frames per video
WIN_SIZE   = 15   # frames per sequence window

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# Load CSV — splits already assigned
df = pd.read_csv(CSV_PATH)
df['language'] = df['language'].str.lower().str.strip()
df['label']    = df['label'].astype(int)

print(f"Total videos : {len(df)}")
print(f"Splits       : {df['split'].value_counts().to_dict()}")
print(f"Labels       : real={sum(df['label']==0)} fake={sum(df['label']==1)}")
print()

def extract_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(
        faces, key=lambda f: f[2]*f[3],
        reverse=True)[0]
    x, y = max(0,x), max(0,y)
    crop = frame[y:min(frame.shape[0],y+h),
                  x:min(frame.shape[1],x+w)]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(
        cv2.resize(crop, (IMG_SIZE, IMG_SIZE)),
        cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

# Storage — frame level
X_tr, y_tr = [], []
X_va, y_va = [], []
X_te, y_te = [], []

# Storage — sequence level (15-frame windows)
Xs_tr, ys_tr = [], []
Xs_va, ys_va = [], []
Xs_te, ys_te = [], []

metadata = []
skipped  = []
done     = 0

for i, row in df.iterrows():
    path  = str(row['video_path']).replace(
        '/', os.sep).replace('\\\\', os.sep)
    vid   = str(row['video_id'])
    label = int(row['label'])
    split = str(row['split'])

    if not os.path.exists(path):
        skipped.append(vid)
        continue

    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fidx  = 0
    faces = []
    times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # HARD STOP — never exceed MAX_FRAMES
        if len(faces) >= MAX_FRAMES:
            break

        if fidx % FRAME_SKIP == 0:
            face = extract_face(frame)
            if face is not None:
                faces.append(face)
                times.append(round(fidx / fps, 3))
        fidx += 1
    cap.release()

    if len(faces) == 0:
        skipped.append(vid)
        done += 1
        print(f"  [{done}/{len(df)}] NO FACES: {vid[:50]}")
        continue

    # Frame-level — add each face separately
    for face in faces:
        if split == 'train':
            X_tr.append(face); y_tr.append(label)
        elif split == 'val':
            X_va.append(face); y_va.append(label)
        else:
            X_te.append(face); y_te.append(label)

    # Sequence-level — only if enough frames for 1 window
    if len(faces) >= WIN_SIZE:
        window = faces[:WIN_SIZE]
        if split == 'train':
            Xs_tr.append(window); ys_tr.append(label)
        elif split == 'val':
            Xs_va.append(window); ys_va.append(label)
        else:
            Xs_te.append(window); ys_te.append(label)

    # Save metadata
    fake_s = row.get('fake_start_sec', 0)
    fake_e = row.get('fake_end_sec', 0)
    metadata.append({
        'video_id':    vid,
        'label':       label,
        'split':       split,
        'language':    row['language'],
        'fake_type':   str(row.get('fake_type', '')),
        'num_frames':  len(faces),
        'frame_times': times,
        'fake_start':  float(fake_s) if str(fake_s) not in ['0','0.0','nan',''] else None,
        'fake_end':    float(fake_e) if str(fake_e) not in ['0','0.0','nan',''] else None,
        'video_path':  path
    })

    done += 1
    seq_count = 1 if len(faces) >= WIN_SIZE else 0
    print(f"  [{done}/{len(df)}] {vid[:45]:<45} "
          f"→ {len(faces):2d} frames "
          f"{seq_count} seq ({split})")

# Convert to arrays
print("\nBuilding arrays...")

X_train_f = np.array(X_tr,  dtype=np.float32)
X_val_f   = np.array(X_va,  dtype=np.float32)
X_test_f  = np.array(X_te,  dtype=np.float32)
y_train_f = np.array(y_tr,  dtype=np.int32)
y_val_f   = np.array(y_va,  dtype=np.int32)
y_test_f  = np.array(y_te,  dtype=np.int32)

X_train_s = np.array(Xs_tr, dtype=np.float32)
X_val_s   = np.array(Xs_va, dtype=np.float32)
X_test_s  = np.array(Xs_te, dtype=np.float32)
y_train_s = np.array(ys_tr, dtype=np.int32)
y_val_s   = np.array(ys_va, dtype=np.int32)
y_test_s  = np.array(ys_te, dtype=np.int32)

# Save
print("Saving...")
np.save(f"{OUTPUT_DIR}/X_train_frames.npy", X_train_f)
np.save(f"{OUTPUT_DIR}/X_val_frames.npy",   X_val_f)
np.save(f"{OUTPUT_DIR}/X_test_frames.npy",  X_test_f)
np.save(f"{OUTPUT_DIR}/y_train_frames.npy", y_train_f)
np.save(f"{OUTPUT_DIR}/y_val_frames.npy",   y_val_f)
np.save(f"{OUTPUT_DIR}/y_test_frames.npy",  y_test_f)

np.save(f"{OUTPUT_DIR}/X_train_seq.npy",    X_train_s)
np.save(f"{OUTPUT_DIR}/X_val_seq.npy",      X_val_s)
np.save(f"{OUTPUT_DIR}/X_test_seq.npy",     X_test_s)
np.save(f"{OUTPUT_DIR}/y_train_seq.npy",    y_train_s)
np.save(f"{OUTPUT_DIR}/y_val_seq.npy",      y_val_s)
np.save(f"{OUTPUT_DIR}/y_test_seq.npy",     y_test_s)

with open(f"{OUTPUT_DIR}/metadata.json", "w",
          encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# Report
print(f"\n{'='*55}")
print(f"PREPROCESSING COMPLETE")
print(f"{'='*55}")
print(f"\nFRAME-LEVEL arrays (Condition A + B training):")
print(f"  X_train : {X_train_f.shape} "
      f"real={sum(y_train_f==0)} fake={sum(y_train_f==1)}")
print(f"  X_val   : {X_val_f.shape}")
print(f"  X_test  : {X_test_f.shape}")
bal = sum(y_train_f==1)/len(y_train_f)*100 if len(y_train_f) else 0
print(f"  Balance : {bal:.1f}% fake in train")

print(f"\nSEQUENCE arrays 15-frame (Condition C CBAM+GRU):")
print(f"  X_train : {X_train_s.shape} "
      f"real={sum(y_train_s==0)} fake={sum(y_train_s==1)}")
print(f"  X_val   : {X_val_s.shape}")
print(f"  X_test  : {X_test_s.shape}")

print(f"\nSkipped  : {len(skipped)} videos")
print(f"\nFiles in {OUTPUT_DIR}:")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    if fn.endswith('.npy'):
        size = os.path.getsize(
            f"{OUTPUT_DIR}/{fn}") / 1e6
        print(f"  {fn:<30} {size:.0f} MB")

print(f"\nNext step: upload all .npy files to")
print(f"Google Drive → deepfake_processed/ folder")