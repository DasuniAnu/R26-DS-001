import os
import sys
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    os.path.join(
        PROJECT_DIR,
        "scripts"
    )
)

from temporal_scoring import (
    load_model_cached,
    extract_temporal_window
)

VIDEO = os.path.join(
    PROJECT_DIR,
    "dataset",
    "dataset",
    "roop_input_v2",
    "sinhala_001.mp4"
)

MODEL = os.path.join(
    PROJECT_DIR,
    "models",
    "condition_c_v3_best.keras"
)

model = load_model_cached(MODEL)
detector = MTCNN()

import cv2

cap = cv2.VideoCapture(VIDEO)

frame_cache = {}

test_windows = [
    (20, 22),   # expected clearly real
    (35, 37),   # expected clearly real
    (50, 52),   # suspicious
    (51, 53),
    (52, 54),
    (53, 55),
    (74, 76),   # suspicious region
    (75, 77),
    (80, 82),
    (84, 86)
]

print("=" * 70)
print("DEBUGGING REAL VIDEO WINDOWS")
print("=" * 70)

for start, end in test_windows:

    sequence, face_count = extract_temporal_window(
        cap=cap,
        detector=detector,
        frame_cache=frame_cache,
        start_sec=start,
        end_sec=end,
        frames_per_window=15,
        image_size=224,
        minimum_detected_faces=8,
        mtcnn_confidence=0.8
    )

    if sequence is None:

        print(
            f"{start:5.1f}-{end:5.1f}s "
            f"| faces={face_count:2d} "
            f"| REJECTED"
        )

        continue

    batch = (
        sequence.astype(np.float32)
        /
        255.0
    )[None, ...]

    prediction = model.predict(
        batch,
        verbose=0
    )[0]

    print(
        f"{start:5.1f}-{end:5.1f}s "
        f"| faces={face_count:2d} "
        f"| REAL={prediction[0]:.4f} "
        f"| FAKE={prediction[1]:.4f}"
    )

cap.release()