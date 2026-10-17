
# scripts/evaluate_temporal_v3.py
# CONDITION C v3 — HELD-OUT PARTIAL-VIDEO TEMPORAL EVALUATION
# ============================================================

import os
import time
import traceback

import numpy as np
import pandas as pd

from temporal_scoring import (
    sliding_window_temporal_scorer,
    temporal_iou,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "condition_c_v3_best.keras",
)

CONFIG_PATH = os.path.join(
    PROJECT_DIR,
    "config",
    "condition_c_v3_deployment_config.json",
)

TEST_CSV = os.path.join(
    PROJECT_DIR,
    "dataset",
    "dataset",
    "test_videos.csv",
)

OUTPUT_CSV = os.path.join(
    PROJECT_DIR,
    "dataset",
    "temporal_iou_results_v3_test.csv",
)

# If the evaluation is interrupted, rerunning this script will skip
# video_ids already saved in OUTPUT_CSV.
RESUME = True


# ============================================================
# VIDEO PATH RESOLUTION
# ============================================================

def resolve_video_path(row):
    """Prefer CSV video_path; otherwise try likely partial folders."""

    original = str(
        row.get("video_path", "") or ""
    ).strip()

    if original and os.path.exists(original):
        return original

    video_id = str(row["video_id"])

    dataset_root = os.path.join(
        PROJECT_DIR,
        "dataset",
        "dataset",
    )

    candidates = [
        os.path.join(
            dataset_root,
            "partial_fake_v2",
            video_id + ".mp4",
        ),
        os.path.join(
            dataset_root,
            "partial_fake",
            video_id + ".mp4",
        ),
        os.path.join(
            dataset_root,
            "partial_fakes",
            video_id + ".mp4",
        ),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return original


# ============================================================
# STARTUP CHECKS
# ============================================================

print("=" * 70)
print("CONDITION C v3 — HELD-OUT TEMPORAL EVALUATION")
print("=" * 70)

for path, name in [
    (MODEL_PATH, "Model"),
    (CONFIG_PATH, "Config"),
    (TEST_CSV, "Test CSV"),
]:
    print(f"{name}: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


df = pd.read_csv(TEST_CSV)

required = {
    "video_id",
    "fake_type",
    "fake_start_sec",
    "fake_end_sec",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        "Missing required test CSV columns: "
        + ", ".join(sorted(missing))
    )


# ============================================================
# HELD-OUT PARTIAL VIDEOS ONLY
# ============================================================

partial_df = df[
    df["fake_type"]
    .astype(str)
    .str.lower()
    .str.strip()
    .eq("partial")
].copy()

partial_df["fake_start_sec"] = pd.to_numeric(
    partial_df["fake_start_sec"],
    errors="coerce",
)

partial_df["fake_end_sec"] = pd.to_numeric(
    partial_df["fake_end_sec"],
    errors="coerce",
)

partial_df = partial_df[
    partial_df["fake_start_sec"].notna()
    & partial_df["fake_end_sec"].notna()
    & (
        partial_df["fake_end_sec"]
        > partial_df["fake_start_sec"]
    )
].copy()

partial_df = partial_df.reset_index(drop=True)

print(
    "Held-out partial test videos:",
    len(partial_df),
)


# ============================================================
# RESUME SUPPORT
# ============================================================

results = []
completed_video_ids = set()

if RESUME and os.path.exists(OUTPUT_CSV):
    try:
        previous_df = pd.read_csv(OUTPUT_CSV)

        if "video_id" in previous_df.columns:
            results = previous_df.to_dict(
                orient="records"
            )

            completed_video_ids = set(
                previous_df["video_id"]
                .astype(str)
                .tolist()
            )

            print(
                "Resume enabled — already completed:",
                len(completed_video_ids),
            )

    except Exception as error:
        print("Resume warning:", str(error))
        results = []
        completed_video_ids = set()


# ============================================================
# EVALUATION LOOP
# ============================================================

started = time.time()
total_partial = len(partial_df)

for row_index, row in partial_df.iterrows():
    video_id = str(row["video_id"])
    display_index = row_index + 1

    if (
        RESUME
        and video_id in completed_video_ids
    ):
        print()
        print(
            f"[{display_index}/{total_partial}] "
            f"{video_id} — already completed, skipping."
        )
        continue

    video_path = resolve_video_path(row)

    gt_start = float(row["fake_start_sec"])
    gt_end = float(row["fake_end_sec"])

    print()
    print("=" * 70)
    print(
        f"[{display_index}/{total_partial}] "
        f"{video_id}"
    )
    print("Path:", video_path)
    print(
        f"GT: {gt_start:.2f}s -> {gt_end:.2f}s"
    )

    row_result = {
        "video_id": video_id,
        "language": row.get("language", ""),
        "ground_truth_start": gt_start,
        "ground_truth_end": gt_end,
    }

    if (
        not video_path
        or not os.path.exists(video_path)
    ):
        print("SKIPPED: video file not found.")

        row_result.update({
            "verdict": "VIDEO_NOT_FOUND",
            "predicted_segments": "[]",
            "iou": np.nan,
            "fake_percentage": 0.0,
            "analysable_percentage": 0.0,
            "windows_scored": 0,
            "candidate_windows": 0,
            "fake_windows": 0,
            "raw_fake_windows": 0,
            "processing_seconds": np.nan,
            "mtcnn_seconds": np.nan,
            "model_inference_seconds": np.nan,
            "realtime_factor": np.nan,
            "cache_hits": 0,
            "cache_misses": 0,
            "cached_source_frames": 0,
            "error": "video file not found",
        })

        results.append(row_result)
        pd.DataFrame(results).to_csv(
            OUTPUT_CSV,
            index=False,
        )
        continue

    try:
        wall_start = time.time()

        result = sliding_window_temporal_scorer(
            video_path,
            MODEL_PATH,
            CONFIG_PATH,
        )

        wall_seconds = time.time() - wall_start

        predicted_segments = (
            result.get("fake_segments", []) or []
        )

        iou = temporal_iou(
            predicted_segments,
            gt_start,
            gt_end,
        )

        timing = result.get("timing", {}) or {}

        print("Verdict:", result.get("verdict"))
        print(
            "Fake coverage:",
            result.get("fake_percentage"),
            "%",
        )
        print("Predicted:", predicted_segments)
        print("IoU:", round(iou, 4))
        print(
            "Wall time:",
            f"{wall_seconds:.2f}s",
        )

        row_result.update({
            "verdict": result.get(
                "verdict",
                "UNKNOWN",
            ),
            "predicted_segments": str(
                predicted_segments
            ),
            "iou": float(iou),
            "fake_percentage": result.get(
                "fake_percentage",
                0.0,
            ),
            "analysable_percentage": result.get(
                "analysable_percentage",
                0.0,
            ),
            "windows_scored": result.get(
                "windows_scored",
                0,
            ),
            "candidate_windows": result.get(
                "candidate_windows",
                0,
            ),
            "fake_windows": result.get(
                "fake_windows",
                0,
            ),
            "raw_fake_windows": result.get(
                "raw_fake_windows",
                0,
            ),
            "processing_seconds": timing.get(
                "total_seconds",
                wall_seconds,
            ),
            "mtcnn_seconds": timing.get(
                "mtcnn_seconds",
                np.nan,
            ),
            "model_inference_seconds": timing.get(
                "model_inference_seconds",
                np.nan,
            ),
            "realtime_factor": timing.get(
                "realtime_factor",
                np.nan,
            ),
            "cache_hits": result.get(
                "cache_hits",
                0,
            ),
            "cache_misses": result.get(
                "cache_misses",
                0,
            ),
            "cached_source_frames": result.get(
                "cached_source_frames",
                0,
            ),
            "error": result.get("error", ""),
        })

    except Exception as error:
        traceback.print_exc()

        row_result.update({
            "verdict": "ERROR",
            "predicted_segments": "[]",
            "iou": np.nan,
            "fake_percentage": 0.0,
            "analysable_percentage": 0.0,
            "windows_scored": 0,
            "candidate_windows": 0,
            "fake_windows": 0,
            "raw_fake_windows": 0,
            "processing_seconds": np.nan,
            "mtcnn_seconds": np.nan,
            "model_inference_seconds": np.nan,
            "realtime_factor": np.nan,
            "cache_hits": 0,
            "cache_misses": 0,
            "cached_source_frames": 0,
            "error": str(error),
        })

    results.append(row_result)

    # Save after EVERY video, so a disconnect does not lose progress.
    pd.DataFrame(results).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("Checkpoint saved:", OUTPUT_CSV)


# ============================================================
# FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

if "iou" in results_df.columns:
    valid_ious = (
        pd.to_numeric(
            results_df["iou"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=np.float32)
    )
else:
    valid_ious = np.array(
        [],
        dtype=np.float32,
    )

print()
print("=" * 70)
print("FINAL CONDITION C v3 TEMPORAL RESULTS")
print("=" * 70)

if len(valid_ious):
    print("Videos evaluated:", len(valid_ious))
    print(
        "Mean IoU:",
        f"{np.mean(valid_ious):.4f}",
    )
    print(
        "Median IoU:",
        f"{np.median(valid_ious):.4f}",
    )
    print(
        "Minimum IoU:",
        f"{np.min(valid_ious):.4f}",
    )
    print(
        "Maximum IoU:",
        f"{np.max(valid_ious):.4f}",
    )
    print(
        "Std Dev:",
        f"{np.std(valid_ious):.4f}",
    )
    print(
        "Strong (>=0.50):",
        int(np.sum(valid_ious >= 0.50)),
    )
    print(
        "Moderate (0.30-0.49):",
        int(
            np.sum(
                (valid_ious >= 0.30)
                & (valid_ious < 0.50)
            )
        ),
    )
    print(
        "Weak (<0.30):",
        int(np.sum(valid_ious < 0.30)),
    )
else:
    print("No valid IoU results generated.")

if (
    not results_df.empty
    and "processing_seconds" in results_df.columns
):
    processing = pd.to_numeric(
        results_df["processing_seconds"],
        errors="coerce",
    ).dropna()

    if len(processing):
        print()
        print(
            "Mean processing/video:",
            f"{processing.mean():.2f}s",
        )
        print(
            "Median processing/video:",
            f"{processing.median():.2f}s",
        )

if (
    not results_df.empty
    and "realtime_factor" in results_df.columns
):
    rtf = pd.to_numeric(
        results_df["realtime_factor"],
        errors="coerce",
    ).dropna()

    if len(rtf):
        print(
            "Mean real-time factor:",
            f"{rtf.mean():.2f}x",
        )

print(
    "Total evaluation runtime:",
    f"{(time.time() - started) / 60:.2f} minutes",
)
print("Saved:", OUTPUT_CSV)
print("=" * 70)