import os
import pandas as pd

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "dataset"
)

VIDEO_ID = "sinhala_001"

for filename in [
    "train_videos.csv",
    "val_videos.csv",
    "test_videos.csv"
]:
    path = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(path)

    match = df[
        df["video_id"].astype(str)
        ==
        VIDEO_ID
    ]

    if not match.empty:
        print("=" * 70)
        print("FOUND IN:", filename)
        print("=" * 70)

        print(
            match[
                [
                    "video_id",
                    "language",
                    "label",
                    "fake_type",
                    "split"
                ]
            ].to_string(index=False)
        )