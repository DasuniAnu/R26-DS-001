"""Prepare grouped train/validation/test CSV files from the research CSV dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.default_config import DataConfig, TrainingConfig
from utils.consistency import (
    CONSISTENT_LABEL,
    INCONSISTENT_LABEL,
    compute_consistency_features,
    consistency_label_from_synthetic_pattern,
    format_consistency_text_input,
    mismatch_type_from_synthetic_pattern,
    normalize_consistency_label,
    normalize_mismatch_type,
)
from utils.data_utils import (
    build_binary_label_mapping,
    clean_text,
    is_meaningful_text,
    normalize_label,
)
from utils.logging_utils import setup_logging
from utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

SYNTHETIC_REQUIRED_COLUMNS = {
    "video_id",
    "group_id",
    "title",
    "description",
    "transcript",
    "thumbnail_text",
    "thumbnail_prompt",
    "label",
    "topic",
    "consistency_pattern",
}

CONTENT_MATCH_REQUIRED_COLUMNS = {
    "video_id",
    "title",
    "description",
    "transcript",
    "label",
}

REAL_REQUIRED_COLUMNS = {
    "video_id",
    "url",
    "title",
    "description",
    "transcript",
    "topic",
    "original_label",
    "binary_label",
    "label_source",
    "annotator_notes",
    "group_id",
}

CONSISTENCY_COLUMNS = [
    "consistency_label",
    "mismatch_type",
    "support_rationale",
    "generated_negative",
    "source_title_video_id",
    "source_transcript_video_id",
    "title_transcript_similarity",
    "description_transcript_similarity",
    "title_description_similarity",
]
OPTIONAL_REAL_COLUMNS = ["url", "label_source", "annotator_notes", *CONSISTENCY_COLUMNS]


def resolve_csv_path(input_path: str | Path) -> Path:
    """Resolve the requested CSV path and support the root-level fallback filename."""

    path = Path(input_path)
    if path.exists():
        return path

    fallback = Path(path.name)
    if fallback.exists():
        LOGGER.warning("Input %s was not found; using %s instead.", path, fallback)
        return fallback

    raise FileNotFoundError(f"Could not find dataset file: {input_path}")


def validate_schema(df: pd.DataFrame) -> str:
    """Accept supported training CSV schemas."""

    columns = set(df.columns)
    if REAL_REQUIRED_COLUMNS.issubset(columns):
        return "real"
    if SYNTHETIC_REQUIRED_COLUMNS.issubset(columns):
        return "synthetic"
    if CONTENT_MATCH_REQUIRED_COLUMNS.issubset(columns):
        return "content_match"

    synthetic_missing = sorted(SYNTHETIC_REQUIRED_COLUMNS.difference(columns))
    content_match_missing = sorted(CONTENT_MATCH_REQUIRED_COLUMNS.difference(columns))
    real_missing = sorted(REAL_REQUIRED_COLUMNS.difference(columns))
    raise ValueError(
        "Dataset must match the real-video, legacy synthetic, or content-match schema. "
        f"Missing for real schema: {real_missing}. "
        f"Missing for synthetic schema: {synthetic_missing}. "
        f"Missing for content-match schema: {content_match_missing}."
    )


def prepare_text_dataframe(
    raw_df: pd.DataFrame,
    dataset_kind: str,
    generate_hard_negatives: bool = False,
) -> pd.DataFrame:
    """Apply consistency labels, feature text, and optional hard negatives."""

    df = raw_df.copy()
    if dataset_kind == "content_match":
        for column in OPTIONAL_REAL_COLUMNS:
            if column not in df.columns:
                df[column] = ""
        if "group_id" not in df.columns:
            df["group_id"] = ""
        df["group_id"] = [
            clean_text(group_id) or f"content-match-{clean_text(video_id)}"
            for group_id, video_id in zip(df["group_id"], df["video_id"], strict=False)
        ]
        if "topic" not in df.columns:
            df["topic"] = "General"
        df["original_label"] = df["label"].map(lambda value: "Misleading" if clean_text(value) == "1" else "Reliable")
        df["label"] = df["label"].map(lambda value: "False" if clean_text(value) == "1" else "Not False")
        df["binary_label"] = df["label"]
        df["consistency_label"] = df["label"].map(
            lambda value: INCONSISTENT_LABEL if value == "False" else CONSISTENT_LABEL
        )
        df["mismatch_type"] = df["label"].map(
            lambda value: "title_transcript_mismatch" if value == "False" else "consistent"
        )
        df["support_rationale"] = df["label"].map(
            lambda value: (
                "Synthetic content-match row: metadata and transcript are intentionally mismatched."
                if value == "False"
                else "Synthetic content-match row: metadata and transcript describe the same content."
            )
        )
        df["generated_negative"] = df["label"].map(lambda value: "true" if value == "False" else "false")
        df["source_title_video_id"] = df["video_id"]
        df["source_transcript_video_id"] = df["video_id"]
        df["label_source"] = "Synthetic content-match dataset"
        df["annotator_notes"] = df["support_rationale"]
        df["url"] = df["video_id"].map(lambda value: f"https://synthetic.local/watch?v={clean_text(value)}")
    elif dataset_kind == "synthetic":
        df["original_label"] = df["label"]
        for column in OPTIONAL_REAL_COLUMNS:
            if column not in df.columns:
                df[column] = ""
        df["consistency_label"] = df["consistency_pattern"].map(consistency_label_from_synthetic_pattern)
        df["mismatch_type"] = df["consistency_pattern"].map(mismatch_type_from_synthetic_pattern)
        df["support_rationale"] = df["consistency_pattern"].map(
            lambda value: f"Synthetic consistency pattern: {clean_text(value) or 'unknown'}."
        )
        df["generated_negative"] = "false"
        df["source_title_video_id"] = df["video_id"]
        df["source_transcript_video_id"] = df["video_id"]
        df["label"] = df["consistency_label"].map(
            lambda value: "False" if value == INCONSISTENT_LABEL else "Not False"
        )
    else:
        if "binary_label" not in df.columns:
            df["binary_label"] = ""

    for column in [
        "video_id",
        "url",
        "group_id",
        "title",
        "description",
        "transcript",
        "label",
        "original_label",
        "topic",
        "label_source",
        "annotator_notes",
        *CONSISTENCY_COLUMNS,
    ]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].map(clean_text)

    df["original_label"] = df["original_label"].map(normalize_label)
    if dataset_kind == "real":
        df["consistency_label"] = [
            normalize_consistency_label(consistency_label, binary_label)
            for consistency_label, binary_label in zip(df["consistency_label"], df["binary_label"], strict=False)
        ]
        df["mismatch_type"] = [
            normalize_mismatch_type(mismatch_type, consistency_label)
            for mismatch_type, consistency_label in zip(df["mismatch_type"], df["consistency_label"], strict=False)
        ]
        df["generated_negative"] = df["generated_negative"].map(lambda value: "true" if value.lower() == "true" else "false")
        df["source_title_video_id"] = [
            source or video_id
            for source, video_id in zip(df["source_title_video_id"], df["video_id"], strict=False)
        ]
        df["source_transcript_video_id"] = [
            source or video_id
            for source, video_id in zip(df["source_transcript_video_id"], df["video_id"], strict=False)
        ]
        df["label"] = df["consistency_label"].map(
            lambda value: "False" if value == INCONSISTENT_LABEL else "Not False"
        )
    df["label"] = df["label"].map(clean_text)
    df["binary_label"] = df["label"]
    df["support_rationale"] = [
        rationale
        or (
            "Title, description, and transcript are treated as aligned."
            if consistency_label == CONSISTENT_LABEL
            else "Title or description is not sufficiently supported by the transcript."
        )
        for rationale, consistency_label in zip(df["support_rationale"], df["consistency_label"], strict=False)
    ]
    df = add_consistency_feature_columns(df)
    if generate_hard_negatives:
        df = add_same_topic_hard_negatives(df)
    df["text"] = [
        format_consistency_text_input(title, description, transcript)
        for title, description, transcript in zip(
            df["title"],
            df["description"],
            df["transcript"],
            strict=False,
        )
    ]

    before = len(df)
    has_any_text_input = (
        df["title"].map(is_meaningful_text)
        | df["description"].map(is_meaningful_text)
        | df["transcript"].map(is_meaningful_text)
    )
    df = df[
        df["group_id"].map(bool)
        & df["label"].map(bool)
        & has_any_text_input
        & df["text"].map(bool)
    ].copy()
    df = df.drop_duplicates(subset=["title", "description", "transcript", "label"])
    df = df.reset_index(drop=True)
    LOGGER.info("Cleaned dataset: %s rows -> %s rows", before, len(df))
    return df


def add_consistency_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_rows = [
        compute_consistency_features(title, description, transcript).as_dict()
        for title, description, transcript in zip(df["title"], df["description"], df["transcript"], strict=False)
    ]
    for column in [
        "title_transcript_similarity",
        "description_transcript_similarity",
        "title_description_similarity",
    ]:
        df[column] = [f"{features[column]:.3f}" for features in feature_rows]
    return df


def add_same_topic_hard_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Create same-topic title/description-vs-transcript mismatch rows."""

    eligible = df[
        (df["generated_negative"] != "true")
        & (df["consistency_label"] == CONSISTENT_LABEL)
        & df["title"].map(is_meaningful_text)
        & df["transcript"].map(is_meaningful_text)
    ].copy()
    generated_rows: list[dict[str, str]] = []
    for topic, topic_df in eligible.groupby("topic", sort=True):
        rows = topic_df.sort_values("video_id").to_dict(orient="records")
        if len(rows) < 2:
            continue
        for index, source in enumerate(rows):
            target = rows[(index + 1) % len(rows)]
            if clean_text(source["video_id"]) == clean_text(target["video_id"]):
                continue
            mismatch_kind = "title_transcript_mismatch" if index % 2 == 0 else "description_transcript_mismatch"
            generated = dict(source)
            generated["video_id"] = f"{source['video_id']}__mismatch__{target['video_id']}"
            generated["url"] = clean_text(source.get("url", ""))
            generated["transcript"] = target["transcript"]
            if mismatch_kind == "description_transcript_mismatch":
                generated["description"] = source["description"] or source["title"]
            generated["label"] = "False"
            generated["binary_label"] = "False"
            generated["original_label"] = "Misleading"
            generated["consistency_label"] = INCONSISTENT_LABEL
            generated["mismatch_type"] = "same_topic_swap"
            generated["support_rationale"] = (
                "Generated hard negative: title/description and transcript come from different "
                f"{topic} videos, so transcript support should fail."
            )
            generated["generated_negative"] = "true"
            generated["source_title_video_id"] = source["video_id"]
            generated["source_transcript_video_id"] = target["video_id"]
            generated["group_id"] = f"generated-{clean_text(topic).lower()}-{source['video_id']}-{target['video_id']}"
            generated_rows.append(generated)
    if not generated_rows:
        return df
    combined = pd.concat([df, pd.DataFrame(generated_rows)], ignore_index=True)
    return add_consistency_feature_columns(combined)


def stratified_group_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create an approximate 70/15/15 split without splitting group_id values."""

    from sklearn.model_selection import StratifiedGroupKFold

    group_count = df["group_id"].nunique()
    if group_count < 20:
        raise ValueError("At least 20 groups are required for the 70/15/15 grouped split.")

    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    df = df.copy()
    df["fold"] = -1
    for fold, (_, fold_idx) in enumerate(
        splitter.split(df, y=df["label"], groups=df["group_id"])
    ):
        df.loc[fold_idx, "fold"] = fold

    train_df = df[df["fold"].between(0, 13)].drop(columns=["fold"]).reset_index(drop=True)
    val_df = df[df["fold"].between(14, 16)].drop(columns=["fold"]).reset_index(drop=True)
    test_df = df[df["fold"].between(17, 19)].drop(columns=["fold"]).reset_index(drop=True)
    return train_df, val_df, test_df


def assert_no_group_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Stop preparation/training if a group_id appears in multiple splits."""

    split_groups = {
        "train": set(train_df["group_id"]),
        "validation": set(val_df["group_id"]),
        "test": set(test_df["group_id"]),
    }
    overlaps = {
        "train_validation": sorted(split_groups["train"] & split_groups["validation"]),
        "train_test": sorted(split_groups["train"] & split_groups["test"]),
        "validation_test": sorted(split_groups["validation"] & split_groups["test"]),
    }
    leaking = {name: groups for name, groups in overlaps.items() if groups}
    if leaking:
        raise RuntimeError(f"Group leakage detected. Stop training. Overlaps: {leaking}")


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Summarize full dataset size, labels, topics, and group count."""

    return {
        "dataset_size": int(len(df)),
        "group_count": int(df["group_id"].nunique()),
        "label_distribution": {
            str(label): int(count)
            for label, count in df["label"].value_counts().sort_index().items()
        },
        "original_label_distribution": {
            str(label): int(count)
            for label, count in df["original_label"].value_counts().sort_index().items()
            if str(label)
        },
        "topic_distribution": {
            str(topic): int(count)
            for topic, count in df["topic"].value_counts().sort_index().items()
        },
    }


def summarize_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Summarize split sizes, label distributions, topic distributions, and group counts."""

    splits = {"train": train_df, "validation": val_df, "test": test_df}
    return {
        split_name: {
            "size": int(len(split_df)),
            "group_count": int(split_df["group_id"].nunique()),
            "label_distribution": {
                str(label): int(count)
                for label, count in split_df["label"].value_counts().sort_index().items()
            },
            "topic_distribution": {
                str(topic): int(count)
                for topic, count in split_df["topic"].value_counts().sort_index().items()
            },
        }
        for split_name, split_df in splits.items()
    }


def log_summary(dataset_summary: dict, split_summary: dict) -> None:
    LOGGER.info("Dataset size: %s", dataset_summary["dataset_size"])
    LOGGER.info("Label distribution: %s", dataset_summary["label_distribution"])
    LOGGER.info("Topic distribution: %s", dataset_summary["topic_distribution"])
    LOGGER.info("Group count: %s", dataset_summary["group_count"])
    for split_name, summary in split_summary.items():
        LOGGER.info("%s size: %s", split_name, summary["size"])
        LOGGER.info("%s group count: %s", split_name, summary["group_count"])
        LOGGER.info("%s label distribution: %s", split_name, summary["label_distribution"])


def prepare_dataset(input_path: str | Path, data_config: DataConfig, training_config: TrainingConfig) -> None:
    """Load the new CSV, clean lightly, split by group, and save artifacts."""

    csv_path = resolve_csv_path(input_path)
    LOGGER.info("Loading dataset from %s", csv_path)
    raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    dataset_kind = validate_schema(raw_df)
    LOGGER.info("Dataset schema: %s", dataset_kind)
    if raw_df.empty:
        raise ValueError(
            "The dataset file exists but has no labeled videos yet. "
            "Add real Sinhala YouTube rows to data/real_sinhala_youtube_false_content.csv first. "
            "Required labels are binary_label=False or binary_label=Not False, with group_id set for each claim/event."
        )
    if dataset_kind == "synthetic":
        LOGGER.warning(
            "Using synthetic data for binary model development only. "
            "Do not report these metrics as real-world accuracy."
        )
    df = prepare_text_dataframe(
        raw_df,
        dataset_kind,
        generate_hard_negatives=dataset_kind != "content_match",
    )
    if df.empty:
        raise ValueError(
            "No usable training rows remained after cleaning. "
            "Each row needs a group_id, a valid binary_label, and meaningful title, description, or transcript text."
        )

    label_to_id = build_binary_label_mapping(df["label"])
    LOGGER.info("Label mapping: %s", label_to_id)

    train_df, val_df, test_df = stratified_group_split(df, seed=training_config.seed)
    assert_no_group_leakage(train_df, val_df, test_df)

    dataset_summary = summarize_dataset(df)
    split_summary = summarize_splits(train_df, val_df, test_df)
    log_summary(dataset_summary, split_summary)

    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    training_config.model_dir.mkdir(parents=True, exist_ok=True)
    for split_df, path in [
        (train_df, data_config.train_csv),
        (val_df, data_config.val_csv),
        (test_df, data_config.test_csv),
    ]:
        split_df.to_csv(path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved %s rows to %s", len(split_df), path)

    for mapping_path in [
        training_config.output_dir / "label_mapping.json",
        training_config.model_dir / "label_mapping.json",
    ]:
        mapping_path.write_text(json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Saved label mapping to %s", mapping_path)

    summary = {
        "dataset_kind": dataset_kind,
        "label_policy": {
            "False": ["Inconsistent title/description/transcript support", "Clickbait or same-topic mismatch"],
            "Not False": ["Consistent title/description/transcript support"],
        },
        "benchmark_warning": (
            "Generated-negative and synthetic metrics are for development only; real-world "
            "accuracy requires a human-reviewed consistency benchmark with generated_negative=false."
        ),
        "dataset": dataset_summary,
        "splits": split_summary,
    }
    summary_path = training_config.output_dir / "data_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved data summary to %s", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare grouped Sinhala YouTube false-content CSV.")
    parser.add_argument(
        "--input",
        default=str(DataConfig().source_csv),
        help=(
            "Path to either the real-video CSV or "
            "data/sinhala_youtube_false_content_synthetic_1000.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    training_config = TrainingConfig()
    set_seed(training_config.seed)
    prepare_dataset(args.input, DataConfig(), training_config)


if __name__ == "__main__":
    main()
