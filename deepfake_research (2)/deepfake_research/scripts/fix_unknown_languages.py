import os
import re
import pandas as pd


# ============================================================
# PATHS
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

MASTER_CSV = os.path.join(
    DATASET_DIR,
    "master_videos.csv"
)


# ============================================================
# UNICODE DETECTION
# ============================================================

def contains_sinhala(text):
    # Sinhala Unicode block: 0D80–0DFF
    return bool(
        re.search(
            r"[\u0D80-\u0DFF]",
            str(text)
        )
    )


def contains_tamil(text):
    # Tamil Unicode block: 0B80–0BFF
    return bool(
        re.search(
            r"[\u0B80-\u0BFF]",
            str(text)
        )
    )


# ============================================================
# LANGUAGE GUESS
# ============================================================

def infer_language(video_id):

    text = str(
        video_id
    ).strip()

    lower = text.lower()

    # --------------------------------------------------------
    # 1. Unicode is strongest evidence
    # --------------------------------------------------------

    if contains_sinhala(text):
        return "sinhala"

    if contains_tamil(text):
        return "tamil"

    # --------------------------------------------------------
    # 2. Explicit English language words
    # --------------------------------------------------------

    if "sinhala" in lower:
        return "sinhala"

    if "tamil" in lower:
        return "tamil"

    # --------------------------------------------------------
    # 3. Sri Lankan / Lanka AI titles
    #    Default to Sinhala unless better evidence exists
    # --------------------------------------------------------

    sri_lanka_tokens = [
        "sri lanka",
        "srilanka",
        "ai lanka",
        "cine ai lanka",
        "lanka"
    ]

    for token in sri_lanka_tokens:

        if token in lower:
            return "sinhala"

    # --------------------------------------------------------
    # 4. Still ambiguous
    # --------------------------------------------------------

    return "unknown"


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(MASTER_CSV):

        raise FileNotFoundError(
            f"master_videos.csv not found:\n{MASTER_CSV}"
        )

    df = pd.read_csv(
        MASTER_CSV
    )

    print("=" * 70)
    print("FIX UNKNOWN LANGUAGE LABELS")
    print("=" * 70)

    unknown_mask = (
        df["language"]
        .astype(str)
        .str.lower()
        ==
        "unknown"
    )

    unknown_df = df[
        unknown_mask
    ].copy()

    print(
        f"Unknown videos before fix: "
        f"{len(unknown_df)}"
    )

    print()

    changed = 0

    for idx, row in unknown_df.iterrows():

        video_id = row[
            "video_id"
        ]

        inferred = infer_language(
            video_id
        )

        print(
            f"{video_id}\n"
            f"  -> {inferred}"
        )

        if inferred != "unknown":

            df.at[
                idx,
                "language"
            ] = inferred

            changed += 1

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
        MASTER_CSV,
        index=False
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    remaining_unknown = df[
        df["language"]
        .astype(str)
        .str.lower()
        ==
        "unknown"
    ]

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Automatically corrected: "
        f"{changed}"
    )

    print(
        f"Still unknown: "
        f"{len(remaining_unknown)}"
    )

    print()

    if len(
        remaining_unknown
    ) > 0:

        print(
            "Still need manual checking:"
        )

        print(
            remaining_unknown[
                [
                    "video_id",
                    "video_path"
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "Final language counts:"
    )

    print(
        df[
            "language"
        ].value_counts()
    )

    print()
    print(
        "Saved:"
    )

    print(
        MASTER_CSV
    )

    print("=" * 70)


if __name__ == "__main__":
    main()