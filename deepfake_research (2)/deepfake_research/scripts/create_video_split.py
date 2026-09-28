# ============================================================
# scripts/create_video_split.py
#
# Creates VIDEO-LEVEL train / validation / test splits
# from:
#   dataset/dataset/master_videos.csv
#
# IMPORTANT:
# - Splits ORIGINAL VIDEOS before any frame/window extraction.
# - Tries to preserve balance across:
#       language
#       fake_type
#
# Output:
#   master_videos.csv   -> updated with split column
#   train_videos.csv
#   val_videos.csv
#   test_videos.csv
# ============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


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

TRAIN_CSV = os.path.join(
    DATASET_DIR,
    "train_videos.csv"
)

VAL_CSV = os.path.join(
    DATASET_DIR,
    "val_videos.csv"
)

TEST_CSV = os.path.join(
    DATASET_DIR,
    "test_videos.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_master_dataset():

    if not os.path.exists(MASTER_CSV):

        raise FileNotFoundError(
            f"master_videos.csv not found:\n"
            f"{MASTER_CSV}"
        )

    df = pd.read_csv(
        MASTER_CSV
    )

    required_columns = [
        "video_id",
        "video_path",
        "language",
        "label",
        "fake_type"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + ", ".join(missing)
        )

    return df


# ============================================================
# FIX / CHECK LANGUAGE
# ============================================================

def clean_language_values(df):

    df = df.copy()

    df["language"] = (
        df["language"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_languages = {
        "sinhala",
        "tamil"
    }

    unknown_mask = ~df[
        "language"
    ].isin(
        valid_languages
    )

    unknown_count = int(
        unknown_mask.sum()
    )

    if unknown_count > 0:

        print()
        print("=" * 70)
        print("WARNING: UNKNOWN LANGUAGE VIDEOS")
        print("=" * 70)

        print(
            df.loc[
                unknown_mask,
                [
                    "video_id",
                    "fake_type",
                    "language"
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            f"Unknown language count: "
            f"{unknown_count}"
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "These videos will still be split, "
            "but they cannot be stratified by "
            "Sinhala/Tamil correctly."
        )

        print(
            "It is better to correct them first "
            "in master_videos.csv."
        )

    return df


# ============================================================
# CREATE STRATIFICATION KEY
# ============================================================

def create_stratification_key(df):

    df = df.copy()

    # Example:
    # sinhala_real
    # tamil_face_swap
    # sinhala_partial
    # tamil_ai_generated

    df["stratify_key"] = (

        df["language"].astype(str)
        +
        "_"
        +
        df["fake_type"].astype(str)
    )

    return df


# ============================================================
# CHECK STRATIFICATION COUNTS
# ============================================================

def check_stratification_counts(df):

    counts = (
        df["stratify_key"]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 70)
    print("STRATIFICATION GROUP COUNTS")
    print("=" * 70)

    print(
        counts
    )

    print()

    small_groups = counts[
        counts < 4
    ]

    if len(small_groups) > 0:

        print(
            "WARNING:"
        )

        print(
            "Some language + fake_type groups "
            "have fewer than 4 videos."
        )

        print(
            small_groups
        )

        print()

        print(
            "For these groups, perfect stratification "
            "may not be possible."
        )

    return counts


# ============================================================
# SPLIT DATASET
# ============================================================

def create_split(df):

    # --------------------------------------------------------
    # FIRST SPLIT:
    # train = 70%
    # temp  = 30%
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(

        df,

        test_size=(
            VAL_RATIO
            +
            TEST_RATIO
        ),

        random_state=RANDOM_STATE,

        shuffle=True,

        stratify=df[
            "stratify_key"
        ]
    )

    # --------------------------------------------------------
    # SECOND SPLIT:
    # temp -> validation + test
    #
    # Since VAL = 15 and TEST = 15,
    # split temp 50 / 50
    # --------------------------------------------------------

    relative_test_ratio = (

        TEST_RATIO
        /
        (
            VAL_RATIO
            +
            TEST_RATIO
        )
    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=relative_test_ratio,

        random_state=RANDOM_STATE,

        shuffle=True,

        stratify=temp_df[
            "stratify_key"
        ]
    )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# FALLBACK SPLIT
# ============================================================

def create_fallback_split(df):

    """
    Used only if strict stratification fails.

    Falls back to stratifying by fake_type only.
    """

    print()
    print("=" * 70)
    print("STRICT STRATIFICATION FAILED")
    print("=" * 70)

    print(
        "Falling back to stratification by fake_type only."
    )

    train_df, temp_df = train_test_split(

        df,

        test_size=(
            VAL_RATIO
            +
            TEST_RATIO
        ),

        random_state=RANDOM_STATE,

        shuffle=True,

        stratify=df[
            "fake_type"
        ]
    )

    relative_test_ratio = (

        TEST_RATIO
        /
        (
            VAL_RATIO
            +
            TEST_RATIO
        )
    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=relative_test_ratio,

        random_state=RANDOM_STATE,

        shuffle=True,

        stratify=temp_df[
            "fake_type"
        ]
    )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# ADD SPLIT COLUMN
# ============================================================

def assign_split_column(
        full_df,
        train_df,
        val_df,
        test_df):

    full_df = full_df.copy()

    full_df["split"] = ""

    train_ids = set(
        train_df["video_path"]
    )

    val_ids = set(
        val_df["video_path"]
    )

    test_ids = set(
        test_df["video_path"]
    )

    full_df.loc[
        full_df[
            "video_path"
        ].isin(
            train_ids
        ),
        "split"
    ] = "train"

    full_df.loc[
        full_df[
            "video_path"
        ].isin(
            val_ids
        ),
        "split"
    ] = "val"

    full_df.loc[
        full_df[
            "video_path"
        ].isin(
            test_ids
        ),
        "split"
    ] = "test"

    return full_df


# ============================================================
# VALIDATE SPLIT
# ============================================================

def validate_split(
        train_df,
        val_df,
        test_df):

    print()
    print("=" * 70)
    print("VALIDATING SPLIT")
    print("=" * 70)

    train_paths = set(
        train_df["video_path"]
    )

    val_paths = set(
        val_df["video_path"]
    )

    test_paths = set(
        test_df["video_path"]
    )

    overlap_train_val = (
        train_paths
        &
        val_paths
    )

    overlap_train_test = (
        train_paths
        &
        test_paths
    )

    overlap_val_test = (
        val_paths
        &
        test_paths
    )

    if (
        overlap_train_val
        or
        overlap_train_test
        or
        overlap_val_test
    ):

        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            "the same video appears "
            "in multiple splits."
        )

    print(
        "No video overlap detected."
    )

    print(
        "VIDEO-LEVEL SPLIT IS SAFE."
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_split_summary(
        name,
        df):

    print()
    print("=" * 70)
    print(
        f"{name.upper()} SET"
    )
    print("=" * 70)

    print(
        f"Videos: {len(df)}"
    )

    print()

    print(
        "Labels:"
    )

    print(
        df[
            "label"
        ].value_counts(
            dropna=False
        ).sort_index()
    )

    print()

    print(
        "Fake types:"
    )

    print(
        df[
            "fake_type"
        ].value_counts(
            dropna=False
        )
    )

    print()

    print(
        "Languages:"
    )

    print(
        df[
            "language"
        ].value_counts(
            dropna=False
        )
    )

    print()

    print(
        "Language + fake type:"
    )

    print(
        pd.crosstab(
            df["language"],
            df["fake_type"]
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CREATE VIDEO-LEVEL TRAIN / VAL / TEST SPLIT")
    print("=" * 70)

    print(
        f"Master CSV:\n"
        f"{MASTER_CSV}"
    )

    df = load_master_dataset()

    print()
    print(
        f"Total videos loaded: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = clean_language_values(
        df
    )

    # --------------------------------------------------------
    # STRATIFICATION KEY
    # --------------------------------------------------------

    df = create_stratification_key(
        df
    )

    check_stratification_counts(
        df
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    try:

        train_df, val_df, test_df = (
            create_split(
                df
            )
        )

        print()
        print(
            "Used stratification by:"
        )

        print(
            "language + fake_type"
        )

    except ValueError as e:

        print()
        print(
            "Reason strict stratification failed:"
        )

        print(
            str(e)
        )

        train_df, val_df, test_df = (
            create_fallback_split(
                df
            )
        )

    # --------------------------------------------------------
    # VALIDATE NO LEAKAGE
    # --------------------------------------------------------

    validate_split(
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # REMOVE TEMP COLUMN
    # --------------------------------------------------------

    for split_df in [
        train_df,
        val_df,
        test_df
    ]:

        if (
            "stratify_key"
            in split_df.columns
        ):

            split_df.drop(
                columns=[
                    "stratify_key"
                ],
                inplace=True
            )

    # --------------------------------------------------------
    # UPDATE MASTER CSV
    # --------------------------------------------------------

    updated_master = (
        assign_split_column(
            df,
            train_df,
            val_df,
            test_df
        )
    )

    if (
        "stratify_key"
        in updated_master.columns
    ):

        updated_master.drop(
            columns=[
                "stratify_key"
            ],
            inplace=True
        )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    unassigned = updated_master[
        updated_master[
            "split"
        ] == ""
    ]

    if len(unassigned) > 0:

        raise RuntimeError(

            f"{len(unassigned)} videos "
            "were not assigned to "
            "train/val/test."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    updated_master.to_csv(
        MASTER_CSV,
        index=False
    )

    train_df.to_csv(
        TRAIN_CSV,
        index=False
    )

    val_df.to_csv(
        VAL_CSV,
        index=False
    )

    test_df.to_csv(
        TEST_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_split_summary(
        "train",
        train_df
    )

    print_split_summary(
        "validation",
        val_df
    )

    print_split_summary(
        "test",
        test_df
    )

    print()
    print("=" * 70)
    print("FINAL SPLIT SUMMARY")
    print("=" * 70)

    total = len(
        updated_master
    )

    print(
        f"Total : {total}"
    )

    print(
        f"Train : "
        f"{len(train_df)} "
        f"({len(train_df)/total*100:.1f}%)"
    )

    print(
        f"Val   : "
        f"{len(val_df)} "
        f"({len(val_df)/total*100:.1f}%)"
    )

    print(
        f"Test  : "
        f"{len(test_df)} "
        f"({len(test_df)/total*100:.1f}%)"
    )

    print()
    print(
        "Updated master CSV:"
    )

    print(
        MASTER_CSV
    )

    print()
    print(
        "Separate split CSVs:"
    )

    print(
        TRAIN_CSV
    )

    print(
        VAL_CSV
    )

    print(
        TEST_CSV
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":

    main()