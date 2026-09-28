# =========================================================
# scripts/evaluate_temporal_v3.py
# Condition C v3 - Held-out Partial-Video Temporal Evaluation
# =========================================================

import ast
import os
import time
import traceback

import numpy as np
import pandas as pd

from temporal_scoring import (
    sliding_window_temporal_scorer,
    temporal_iou
)

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "condition_c_v3_best.keras"
)

CONFIG_PATH = os.path.join(
    PROJECT_DIR,
    "config",
    "condition_c_v3_deployment_config.json"
)

TEST_CSV = os.path.join(
    PROJECT_DIR,
    "dataset",
    "dataset",
    "test_videos.csv"
)

OUTPUT_CSV = os.path.join(
    PROJECT_DIR,
    "dataset",
    "temporal_iou_results_v3_test.csv"
)


def resolve_video_path(row):
    """
    Prefer the CSV video_path if it exists. Otherwise try the final
    dataset partial-video folder using video_id.
    """
    original = str(row.get("video_path", "") or "").strip()

    if original and os.path.exists(original):
        return original

    video_id = str(row["video_id"])

    candidates = [
        os.path.join(
            PROJECT_DIR,
            "dataset",
            "dataset",
            "partial_fake_v2",
            video_id + ".mp4"
        ),
        os.path.join(
            PROJECT_DIR,
            "dataset",
            "dataset",
            "partial_fake",
            video_id + ".mp4"
        ),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return original


print("=" * 70)
print("CONDITION C v3 — HELD-OUT TEMPORAL EVALUATION")
print("=" * 70)

for path, name in [
    (MODEL_PATH, "Model"),
    (CONFIG_PATH, "Config"),
    (TEST_CSV, "Test CSV")
]:
    print(f"{name}: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found:\n{path}")

df = pd.read_csv(TEST_CSV)

required = {
    "video_id",
    "fake_type",
    "fake_start_sec",
    "fake_end_sec"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        "Missing required test CSV columns: "
        + ", ".join(sorted(missing))
    )

partial_df = df[
    df["fake_type"]
    .astype(str)
    .str.lower()
    .eq("partial")
].copy()

partial_df["fake_start_sec"] = pd.to_numeric(
    partial_df["fake_start_sec"],
    errors="coerce"
)

partial_df["fake_end_sec"] = pd.to_numeric(
    partial_df["fake_end_sec"],
    errors="coerce"
)

partial_df = partial_df[
    partial_df["fake_start_sec"].notna()
    &
    partial_df["fake_end_sec"].notna()
    &
    (
        partial_df["fake_end_sec"]
        >
        partial_df["fake_start_sec"]
    )
].copy()

print("Held-out partial test videos:", len(partial_df))

results = []
ious = []
started = time.time()

for index, (_, row) in enumerate(
    partial_df.iterrows(),
    start=1
):
    video_id = str(row["video_id"])
    video_path = resolve_video_path(row)

    gt_start = float(row["fake_start_sec"])
    gt_end = float(row["fake_end_sec"])

    print()
    print("=" * 70)
    print(f"[{index}/{len(partial_df)}] {video_id}")
    print("Path:", video_path)
    print(f"GT: {gt_start:.2f}s -> {gt_end:.2f}s")

    if not video_path or not os.path.exists(video_path):
        print("SKIPPED: video file not found.")
        results.append({
            "video_id": video_id,
            "language": row.get("language", ""),
            "verdict": "VIDEO_NOT_FOUND",
            "ground_truth_start": gt_start,
            "ground_truth_end": gt_end,
            "predicted_segments": "[]",
            "iou": np.nan,
            "fake_percentage": 0.0,
            "analysable_percentage": 0.0,
            "windows_scored": 0,
            "fake_windows": 0
        })
        continue

    try:
        result = sliding_window_temporal_scorer(
            video_path,
            MODEL_PATH,
            CONFIG_PATH
        )

        predicted_segments = result.get(
            "fake_segments",
            []
        )

        iou = temporal_iou(
            predicted_segments,
            gt_start,
            gt_end
        )

        ious.append(iou)

        print("Verdict:", result.get("verdict"))
        print(
            "Fake coverage:",
            result.get("fake_percentage"),
            "%"
        )
        print("Predicted:", predicted_segments)
        print("IoU:", round(iou, 4))

        results.append({
            "video_id": video_id,
            "language": row.get("language", ""),
            "verdict": result.get("verdict", "UNKNOWN"),
            "ground_truth_start": gt_start,
            "ground_truth_end": gt_end,
            "predicted_segments": str(predicted_segments),
            "iou": float(iou),
            "fake_percentage": result.get(
                "fake_percentage",
                0.0
            ),
            "analysable_percentage": result.get(
                "analysable_percentage",
                0.0
            ),
            "windows_scored": result.get(
                "windows_scored",
                0
            ),
            "fake_windows": result.get(
                "fake_windows",
                0
            )
        })

    except Exception as exc:
        traceback.print_exc()

        results.append({
            "video_id": video_id,
            "language": row.get("language", ""),
            "verdict": "ERROR",
            "ground_truth_start": gt_start,
            "ground_truth_end": gt_end,
            "predicted_segments": "[]",
            "iou": np.nan,
            "fake_percentage": 0.0,
            "analysable_percentage": 0.0,
            "windows_scored": 0,
            "fake_windows": 0,
            "error": str(exc)
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

valid_ious = results_df["iou"].dropna().to_numpy(
    dtype=np.float32
)

print()
print("=" * 70)
print("FINAL CONDITION C v3 TEMPORAL RESULTS")
print("=" * 70)

if len(valid_ious):
    print("Videos evaluated:", len(valid_ious))
    print("Mean IoU:", f"{np.mean(valid_ious):.4f}")
    print("Median IoU:", f"{np.median(valid_ious):.4f}")
    print("Minimum IoU:", f"{np.min(valid_ious):.4f}")
    print("Maximum IoU:", f"{np.max(valid_ious):.4f}")
    print("Std Dev:", f"{np.std(valid_ious):.4f}")
    print(
        "Strong (>=0.50):",
        int(np.sum(valid_ious >= 0.50))
    )
    print(
        "Moderate (0.30-0.49):",
        int(
            np.sum(
                (valid_ious >= 0.30)
                &
                (valid_ious < 0.50)
            )
        )
    )
    print(
        "Weak (<0.30):",
        int(np.sum(valid_ious < 0.30))
    )
else:
    print("No valid IoU results generated.")

print(
    "Runtime:",
    f"{(time.time() - started) / 60:.2f} minutes"
)

print("Saved:", OUTPUT_CSV)
print("=" * 70)