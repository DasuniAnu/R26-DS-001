# ============================================================
# scripts/build_master_dataset.py
#
# Build one master metadata CSV from:
#   roop_input_v2        -> REAL
#   fake_full_v2         -> FULL FACE-SWAP FAKE
#   fake_partial_v2      -> PARTIAL FAKE
#   fake_ai_generated    -> AI-GENERATED FAKE
#
# Output:
#   dataset/dataset/master_videos.csv
# ============================================================

import os
import cv2
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "dataset"
)

REAL_DIR = os.path.join(
    DATASET_DIR,
    "roop_input_v2"
)

FULL_FAKE_DIR = os.path.join(
    DATASET_DIR,
    "fake_full_v2"
)

PARTIAL_FAKE_DIR = os.path.join(
    DATASET_DIR,
    "fake_partial_v2"
)

AI_FAKE_DIR = os.path.join(
    DATASET_DIR,
    "fake_ai_generated"
)

PARTIAL_GT_CSV = os.path.join(
    DATASET_DIR,
    "ground_truth_partial_v2.csv"
)

OUTPUT_CSV = os.path.join(
    DATASET_DIR,
    "master_videos.csv"
)


# ============================================================
# SETTINGS
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v"
}


# ============================================================
# HELPERS
# ============================================================

def get_video_duration(video_path):
    """
    Return video duration in seconds.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    cap.release()

    if (
        fps is None
        or fps <= 0
        or total_frames is None
        or total_frames <= 0
    ):
        return None

    duration = total_frames / fps

    return round(
        float(duration),
        3
    )


def detect_language(filename):
    """
    Detect language from filename.

    Assumes filenames contain words such as:
        sinhala
        sin
        tamil
        tam

    If not detected:
        unknown
    """

    name = filename.lower()

    sinhala_tokens = [
        "sinhala",
        "sin_",
        "_sin",
        "sin-"
    ]

    tamil_tokens = [
        "tamil",
        "tam_",
        "_tam",
        "tam-"
    ]

    for token in sinhala_tokens:

        if token in name:
            return "sinhala"

    for token in tamil_tokens:

        if token in name:
            return "tamil"

    return "unknown"


def list_video_files(folder):
    """
    Return sorted video files from one folder.
    """

    if not os.path.exists(folder):

        print(
            f"WARNING: folder does not exist:\n{folder}"
        )

        return []

    files = []

    for filename in os.listdir(folder):

        full_path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(full_path):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in VIDEO_EXTENSIONS:
            files.append(full_path)

    return sorted(files)


def normalize_video_id(filename):
    """
    Remove extension and normalize filename.
    """

    return os.path.splitext(
        os.path.basename(filename)
    )[0]


# ============================================================
# LOAD PARTIAL GROUND TRUTH
# ============================================================

def load_partial_ground_truth():

    if not os.path.exists(
        PARTIAL_GT_CSV
    ):

        raise FileNotFoundError(
            f"Partial ground truth CSV not found:\n"
            f"{PARTIAL_GT_CSV}"
        )

    df = pd.read_csv(
        PARTIAL_GT_CSV
    )

    required_columns = [
        "video_id",
        "fake_start_sec",
        "fake_end_sec"
    ]

    missing = [
        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns in "
            "ground_truth_partial_v2.csv:\n"
            + ", ".join(missing)
        )

    # Make sure timestamps are numeric
    df["fake_start_sec"] = pd.to_numeric(
        df["fake_start_sec"],
        errors="coerce"
    )

    df["fake_end_sec"] = pd.to_numeric(
        df["fake_end_sec"],
        errors="coerce"
    )

    # Create lookup by video_id
    lookup = {}

    for row in df.itertuples():

        key = str(
            row.video_id
        ).strip().lower()

        lookup[key] = {

            "fake_start_sec":
                float(row.fake_start_sec),

            "fake_end_sec":
                float(row.fake_end_sec),

            "language":
                getattr(
                    row,
                    "language",
                    None
                ),

            "source":
                getattr(
                    row,
                    "source",
                    None
                )
        }

    return lookup


# ============================================================
# BUILD MASTER DATASET
# ============================================================

def build_master_dataset():

    print()
    print("=" * 70)
    print("BUILDING MASTER VIDEO DATASET")
    print("=" * 70)

    partial_lookup = (
        load_partial_ground_truth()
    )

    rows = []

    errors = []

    # ========================================================
    # REAL VIDEOS
    # ========================================================

    real_files = list_video_files(
        REAL_DIR
    )

    print()
    print(
        f"Real videos found: "
        f"{len(real_files)}"
    )

    for path in real_files:

        filename = os.path.basename(
            path
        )

        video_id = normalize_video_id(
            filename
        )

        duration = get_video_duration(
            path
        )

        if duration is None:

            errors.append({
                "video_path": path,
                "reason":
                    "Could not read duration"
            })

            continue

        rows.append({

            "video_id":
                video_id,

            "video_path":
                os.path.abspath(path),

            "language":
                detect_language(filename),

            "label":
                0,

            "fake_type":
                "real",

            "fake_start_sec":
                0.0,

            "fake_end_sec":
                0.0,

            "total_duration":
                duration,

            "source":
                "real",

            "split":
                ""
        })


    # ========================================================
    # FULL FACE-SWAP FAKE VIDEOS
    # ========================================================

    full_fake_files = list_video_files(
        FULL_FAKE_DIR
    )

    print(
        f"Full face-swap fake videos found: "
        f"{len(full_fake_files)}"
    )

    for path in full_fake_files:

        filename = os.path.basename(
            path
        )

        video_id = normalize_video_id(
            filename
        )

        duration = get_video_duration(
            path
        )

        if duration is None:

            errors.append({
                "video_path": path,
                "reason":
                    "Could not read duration"
            })

            continue

        rows.append({

            "video_id":
                video_id,

            "video_path":
                os.path.abspath(path),

            "language":
                detect_language(filename),

            "label":
                1,

            "fake_type":
                "face_swap",

            "fake_start_sec":
                0.0,

            "fake_end_sec":
                duration,

            "total_duration":
                duration,

            "source":
                "generated_face_swap",

            "split":
                ""
        })


    # ========================================================
    # PARTIAL FAKE VIDEOS
    # ========================================================

    partial_files = list_video_files(
        PARTIAL_FAKE_DIR
    )

    print(
        f"Partial fake videos found: "
        f"{len(partial_files)}"
    )

    unmatched_partial = []

    for path in partial_files:

        filename = os.path.basename(
            path
        )

        video_id = normalize_video_id(
            filename
        )

        duration = get_video_duration(
            path
        )

        if duration is None:

            errors.append({
                "video_path": path,
                "reason":
                    "Could not read duration"
            })

            continue

        key = video_id.lower()

        gt = partial_lookup.get(
            key
        )

        if gt is None:

            unmatched_partial.append(
                video_id
            )

            continue

        language = gt.get(
            "language"
        )

        if (
            language is None
            or pd.isna(language)
        ):

            language = detect_language(
                filename
            )

        source = gt.get(
            "source"
        )

        if (
            source is None
            or pd.isna(source)
        ):

            source = "partial_generated"

        fake_start = float(
            gt["fake_start_sec"]
        )

        fake_end = float(
            gt["fake_end_sec"]
        )

        # Safety checks
        fake_start = max(
            0.0,
            fake_start
        )

        fake_end = min(
            duration,
            fake_end
        )

        if fake_end <= fake_start:

            errors.append({

                "video_path":
                    path,

                "reason":
                    (
                        "Invalid partial timestamps: "
                        f"{fake_start} -> {fake_end}"
                    )
            })

            continue

        rows.append({

            "video_id":
                video_id,

            "video_path":
                os.path.abspath(path),

            "language":
                str(language).lower(),

            "label":
                1,

            "fake_type":
                "partial",

            "fake_start_sec":
                round(
                    fake_start,
                    3
                ),

            "fake_end_sec":
                round(
                    fake_end,
                    3
                ),

            "total_duration":
                duration,

            "source":
                source,

            "split":
                ""
        })


    # ========================================================
    # AI GENERATED VIDEOS
    # ========================================================

    ai_files = list_video_files(
        AI_FAKE_DIR
    )

    print(
        f"AI-generated videos found: "
        f"{len(ai_files)}"
    )

    for path in ai_files:

        filename = os.path.basename(
            path
        )

        video_id = normalize_video_id(
            filename
        )

        duration = get_video_duration(
            path
        )

        if duration is None:

            errors.append({
                "video_path": path,
                "reason":
                    "Could not read duration"
            })

            continue

        rows.append({

            "video_id":
                video_id,

            "video_path":
                os.path.abspath(path),

            "language":
                detect_language(filename),

            "label":
                1,

            "fake_type":
                "ai_generated",

            "fake_start_sec":
                0.0,

            "fake_end_sec":
                duration,

            "total_duration":
                duration,

            "source":
                "ai_generated",

            "split":
                ""
        })


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    master_df = pd.DataFrame(
        rows
    )

    if master_df.empty:

        raise RuntimeError(
            "No videos were added to "
            "the master dataset."
        )


    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    duplicate_ids = master_df[
        master_df.duplicated(
            subset=[
                "video_id",
                "fake_type"
            ],
            keep=False
        )
    ]

    if len(duplicate_ids) > 0:

        print()
        print(
            "WARNING: duplicate video IDs detected:"
        )

        print(
            duplicate_ids[
                [
                    "video_id",
                    "fake_type",
                    "video_path"
                ]
            ]
        )


    # ========================================================
    # SAVE MASTER CSV
    # ========================================================

    master_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MASTER DATASET CREATED")
    print("=" * 70)

    print(
        f"Output:\n{OUTPUT_CSV}"
    )

    print()
    print(
        f"Total videos: "
        f"{len(master_df)}"
    )

    print()
    print("Label counts:")

    print(
        master_df[
            "label"
        ].value_counts(
            dropna=False
        ).sort_index()
    )

    print()
    print("Fake type counts:")

    print(
        master_df[
            "fake_type"
        ].value_counts(
            dropna=False
        )
    )

    print()
    print("Language counts:")

    print(
        master_df[
            "language"
        ].value_counts(
            dropna=False
        )
    )


    # ========================================================
    # EXPECTED COUNTS
    # ========================================================

    print()
    print("=" * 70)
    print("EXPECTED VS ACTUAL")
    print("=" * 70)

    print(
        f"REAL       : "
        f"{sum(master_df['fake_type'] == 'real')}"
    )

    print(
        f"FACE SWAP  : "
        f"{sum(master_df['fake_type'] == 'face_swap')}"
    )

    print(
        f"PARTIAL    : "
        f"{sum(master_df['fake_type'] == 'partial')}"
    )

    print(
        f"AI GENERATED: "
        f"{sum(master_df['fake_type'] == 'ai_generated')}"
    )


    # ========================================================
    # PARTIAL CSV MATCHING CHECK
    # ========================================================

    if unmatched_partial:

        print()
        print(
            "=" * 70
        )

        print(
            "WARNING: PARTIAL VIDEOS NOT FOUND "
            "IN GROUND TRUTH CSV"
        )

        print(
            "=" * 70
        )

        for video_id in unmatched_partial:

            print(
                video_id
            )

        print(
            "\nThese videos were NOT included "
            "in master_videos.csv."
        )

    else:

        print()
        print(
            "All partial videos matched "
            "ground_truth_partial_v2.csv."
        )


    # ========================================================
    # VIDEO READ ERRORS
    # ========================================================

    if errors:

        print()
        print("=" * 70)
        print("VIDEO / METADATA ERRORS")
        print("=" * 70)

        for error in errors:

            print(
                error
            )

    else:

        print()
        print(
            "No video read errors found."
        )


    # ========================================================
    # UNKNOWN LANGUAGE
    # ========================================================

    unknown_df = master_df[
        master_df[
            "language"
        ] == "unknown"
    ]

    if len(unknown_df) > 0:

        print()
        print("=" * 70)
        print("UNKNOWN LANGUAGE VIDEOS")
        print("=" * 70)

        print(
            unknown_df[
                [
                    "video_id",
                    "fake_type"
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "This is not fatal. "
            "We can correct language automatically "
            "in the next step if naming patterns differ."
        )


    # ========================================================
    # PREVIEW
    # ========================================================

    print()
    print("=" * 70)
    print("FIRST 10 ROWS")
    print("=" * 70)

    print(
        master_df.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_master_dataset()