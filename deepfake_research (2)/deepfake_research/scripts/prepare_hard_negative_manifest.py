# ============================================================
# scripts/prepare_hard_negative_manifest.py
# Create deterministic TRAIN / HOLDOUT split for
# new genuine REAL hard-negative videos
# ============================================================

from pathlib import Path
import random

import cv2
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

VIDEO_DIR = (
    PROJECT_DIR
    / "dataset"
    / "dataset"
    / "hard_negative_real"
)

OUTPUT_CSV = (
    PROJECT_DIR
    / "dataset"
    / "dataset"
    / "hard_negative_manifest_v3.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# 21 videos:
# 16 used for adaptation/fine-tuning
# 5 kept COMPLETELY unseen
HOLDOUT_COUNT = 5

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}


# ============================================================
# VALIDATE FOLDER
# ============================================================

if not VIDEO_DIR.exists():

    raise FileNotFoundError(
        f"Folder not found:\n{VIDEO_DIR}"
    )


videos = sorted(
    [
        path
        for path in VIDEO_DIR.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    ],
    key=lambda p: p.name.lower()
)


if len(videos) < 6:

    raise ValueError(
        "Need at least 6 hard-negative REAL videos "
        "to create train + holdout groups."
    )


print("=" * 75)
print("HARD-NEGATIVE REAL DATASET CHECK")
print("=" * 75)

print(
    "Folder:",
    VIDEO_DIR
)

print(
    "Videos found:",
    len(videos)
)


# ============================================================
# INSPECT VIDEOS
# ============================================================

records = []

for index, video_path in enumerate(
    videos,
    start=1
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    opened = cap.isOpened()

    if opened:

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if fps > 0:

            duration = (
                frame_count
                /
                fps
            )

        else:

            duration = 0.0

    else:

        fps = 0.0
        frame_count = 0
        width = 0
        height = 0
        duration = 0.0

    cap.release()

    valid = bool(
        opened
        and
        fps > 0
        and
        frame_count > 0
        and
        duration >= 2.0
    )

    records.append({

        "video_id":
            video_path.stem,

        "filename":
            video_path.name,

        "video_path":
            str(
                video_path.resolve()
            ),

        "label":
            0,

        "fake_type":
            "real_hard_negative",

        "fps":
            round(
                fps,
                3
            ),

        "frame_count":
            frame_count,

        "duration_sec":
            round(
                duration,
                3
            ),

        "width":
            width,

        "height":
            height,

        "valid_video":
            valid
    })

    status = (
        "OK"
        if valid
        else
        "INVALID"
    )

    print(
        f"[{index:02d}/{len(videos):02d}] "
        f"{video_path.name} | "
        f"{duration:.2f}s | "
        f"{width}x{height} | "
        f"{fps:.2f} FPS | "
        f"{status}"
    )


df = pd.DataFrame(
    records
)


# ============================================================
# REMOVE INVALID FILES FROM SPLITTING
# ============================================================

valid_df = df[
    df[
        "valid_video"
    ]
].copy()


invalid_df = df[
    ~df[
        "valid_video"
    ]
].copy()


print()
print("=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)

print(
    "Valid videos:",
    len(valid_df)
)

print(
    "Invalid videos:",
    len(invalid_df)
)


if len(valid_df) <= HOLDOUT_COUNT:

    raise ValueError(
        "Not enough valid videos "
        "for the requested holdout split."
    )


# ============================================================
# DETERMINISTIC VIDEO-LEVEL SPLIT
# ============================================================

video_ids = (
    valid_df[
        "video_id"
    ]
    .tolist()
)


random.Random(
    SEED
).shuffle(
    video_ids
)


holdout_ids = set(
    video_ids[
        :HOLDOUT_COUNT
    ]
)


valid_df[
    "hard_negative_split"
] = valid_df[
    "video_id"
].apply(
    lambda video_id:
        (
            "holdout"
            if video_id
            in holdout_ids
            else
            "train"
        )
)


# Invalid files are never used.
if not invalid_df.empty:

    invalid_df[
        "hard_negative_split"
    ] = "invalid"


final_df = pd.concat(
    [
        valid_df,
        invalid_df
    ],
    ignore_index=True
)


final_df = final_df.sort_values(
    "video_id"
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 75)
print("FINAL HARD-NEGATIVE SPLIT")
print("=" * 75)

print(
    final_df[
        "hard_negative_split"
    ].value_counts()
)

print()
print("TRAIN VIDEOS:")

print(
    final_df[
        final_df[
            "hard_negative_split"
        ]
        ==
        "train"
    ][
        "video_id"
    ].to_string(
        index=False
    )
)

print()
print("UNSEEN HOLDOUT VIDEOS:")

print(
    final_df[
        final_df[
            "hard_negative_split"
        ]
        ==
        "holdout"
    ][
        "video_id"
    ].to_string(
        index=False
    )
)

print()
print(
    "Manifest saved:",
    OUTPUT_CSV
)

print("=" * 75)