"""Train the fast text model with high-weight real consistency samples."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.default_config import BINARY_LABEL_ORDER, DataConfig, TrainingConfig
from evaluation.metrics import compute_metrics, save_metrics
from training.prepare_data import prepare_dataset, prepare_text_dataframe, validate_schema
from training.train_tfidf_content_match import ARTIFACT_NAME, evaluate_split, load_label_mapping, make_pipeline
from utils.data_utils import clean_text
from utils.logging_utils import setup_logging
from utils.seed import set_seed


LOGGER = logging.getLogger(__name__)


def read_prepared_split(path: Path, label_to_id: dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {"text", "label"}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required prepared columns: {sorted(missing)}")
    df = df[df["label"].isin(label_to_id)].copy()
    df["label_id"] = df["label"].map(label_to_id).astype(int)
    return df


def load_real_samples(path: Path, label_to_id: dict[str, int]) -> pd.DataFrame:
    raw_df = pd.read_csv(path, encoding="utf-8-sig")
    dataset_kind = validate_schema(raw_df)
    if dataset_kind != "real":
        raise ValueError(f"{path} must use the real labeled schema, got {dataset_kind}.")
    prepared = prepare_text_dataframe(raw_df, dataset_kind="real", generate_hard_negatives=False)
    prepared = prepared[prepared["label"].isin(label_to_id)].copy()
    prepared = collapse_repeated_video_ids(prepared)
    prepared["label_id"] = prepared["label"].map(label_to_id).astype(int)
    prepared["is_real_youtube"] = ~prepared["video_id"].astype(str).str.startswith("human-sim-")
    return prepared


def collapse_repeated_video_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one majority-label annotation per video and exclude unresolved ties."""

    selected_rows: list[pd.Series] = []
    conflicting_ids: list[str] = []
    tied_ids: list[str] = []
    for video_id, group in df.groupby("video_id", sort=False):
        label_counts = group["label"].value_counts()
        highest_count = int(label_counts.max())
        majority_labels = sorted(label_counts[label_counts == highest_count].index.astype(str))
        if len(majority_labels) != 1:
            tied_ids.append(clean_text(video_id))
            continue

        majority_label = majority_labels[0]
        if len(label_counts) > 1:
            conflicting_ids.append(clean_text(video_id))
        candidates = group[group["label"] == majority_label].copy()
        candidates["_evidence_length"] = candidates.apply(
            lambda row: sum(
                len(clean_text(row.get(column, "")))
                for column in ("title", "description", "transcript", "support_rationale", "annotator_notes")
            ),
            axis=1,
        )
        selected_rows.append(candidates.sort_values("_evidence_length", ascending=False).iloc[0].drop(labels="_evidence_length"))

    if conflicting_ids:
        LOGGER.warning(
            "Resolved %s conflicting repeated video id(s) by majority label: %s",
            len(conflicting_ids),
            ", ".join(conflicting_ids),
        )
    if tied_ids:
        LOGGER.warning(
            "Excluded %s repeated video id(s) with tied labels: %s",
            len(tied_ids),
            ", ".join(tied_ids),
        )
    if not selected_rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def summarize_real_samples(df: pd.DataFrame) -> dict:
    source_counts = df.get("label_source", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
    return {
        "rows": int(len(df)),
        "labels": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "unique_video_ids": int(df["video_id"].nunique()),
        "duplicate_video_rows": int(len(df) - df["video_id"].nunique()),
        "real_youtube_rows": int(df["is_real_youtube"].sum()) if "is_real_youtube" in df.columns else None,
        "human_sim_rows": int((~df["is_real_youtube"]).sum()) if "is_real_youtube" in df.columns else None,
        "label_sources": {str(k): int(v) for k, v in source_counts.items()},
    }


def apparent_metrics(model, df: pd.DataFrame, label_names: list[str]) -> dict:
    texts = df["text"].fillna("").astype(str)
    labels = df["label_id"].astype(int).tolist()
    predictions = model.predict(texts).astype(int).tolist()
    probabilities = model.predict_proba(texts)
    false_index = label_names.index("False")
    metrics = compute_metrics(labels, predictions, label_names, y_score=probabilities[:, false_index].tolist())
    metrics["rows"] = int(len(df))
    metrics["classification_report_text"] = classification_report(
        labels,
        predictions,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    metrics["confusion_matrix_text"] = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(label_names))),
    ).tolist()
    return metrics


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_config = DataConfig()
    training_config = TrainingConfig(seed=args.seed)

    if args.prepare_synthetic:
        LOGGER.info("Preparing synthetic/content-match base dataset from %s", args.synthetic_input)
        prepare_dataset(args.synthetic_input, data_config, training_config)

    label_to_id = load_label_mapping(training_config.model_dir / "label_mapping.json")
    label_names = [label for label, _idx in sorted(label_to_id.items(), key=lambda item: item[1])]
    if set(label_names) != set(BINARY_LABEL_ORDER):
        raise ValueError(f"This trainer expects binary labels {BINARY_LABEL_ORDER}, got {label_names}")

    synthetic_train = read_prepared_split(data_config.train_csv, label_to_id)
    synthetic_val = read_prepared_split(data_config.val_csv, label_to_id)
    synthetic_test = read_prepared_split(data_config.test_csv, label_to_id)
    real_samples = load_real_samples(Path(args.real_input), label_to_id)

    if real_samples.empty:
        raise ValueError(f"No usable real labeled samples found in {args.real_input}.")

    train_df = pd.concat([synthetic_train, real_samples], ignore_index=True)
    sample_weights = [float(args.synthetic_weight)] * len(synthetic_train)
    for is_real_youtube in real_samples["is_real_youtube"].tolist():
        sample_weights.append(float(args.real_weight if is_real_youtube else args.reviewed_synthetic_weight))

    model = make_pipeline(args)
    LOGGER.info("Synthetic train rows: %s", len(synthetic_train))
    LOGGER.info("Real/calibration rows: %s", len(real_samples))
    LOGGER.info("Real sample summary: %s", summarize_real_samples(real_samples))
    model.fit(train_df["text"].fillna("").astype(str), train_df["label_id"].astype(int), classifier__sample_weight=sample_weights)

    validation_metrics = evaluate_split(
        model,
        synthetic_val["text"].fillna("").astype(str),
        synthetic_val["label_id"].astype(int).tolist(),
        label_names,
    )
    test_metrics = evaluate_split(
        model,
        synthetic_test["text"].fillna("").astype(str),
        synthetic_test["label_id"].astype(int).tolist(),
        label_names,
    )
    real_metrics = apparent_metrics(model, real_samples, label_names)

    metrics = {
        "model_type": "sklearn_tfidf_logistic_regression_real_calibrated",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "synthetic_train_rows": int(len(synthetic_train)),
            "synthetic_validation_rows": int(len(synthetic_val)),
            "synthetic_test_rows": int(len(synthetic_test)),
            "real_calibration": summarize_real_samples(real_samples),
            "weights": {
                "synthetic": args.synthetic_weight,
                "real_youtube": args.real_weight,
                "human_reviewed_synthetic": args.reviewed_synthetic_weight,
            },
        },
        "label_mapping": label_to_id,
        "synthetic_validation": validation_metrics,
        "synthetic_test": test_metrics,
        "real_calibration_apparent": real_metrics,
        "benchmark_warning": (
            "Real calibration metrics are apparent training-set metrics because there are too few real rows "
            "for a reliable holdout. Collect more real rows before claiming real-world accuracy."
        ),
    }

    training_config.model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = training_config.model_dir / ARTIFACT_NAME
    joblib.dump(
        {
            "model": model,
            "label_to_id": label_to_id,
            "id_to_label": {idx: label for label, idx in label_to_id.items()},
            "label_order": label_names,
            "model_type": metrics["model_type"],
            "created_at": metrics["trained_at"],
        },
        artifact_path,
    )
    metrics_path = training_config.output_dir / "sklearn_text_real_calibrated_metrics.json"
    save_metrics(metrics, metrics_path)
    LOGGER.info("Saved real-calibrated text model to %s", artifact_path)
    LOGGER.info("Saved real-calibrated metrics to %s", metrics_path)
    LOGGER.info("Synthetic test macro F1: %.4f", test_metrics["macro_f1"])
    LOGGER.info("Real apparent macro F1: %.4f", real_metrics["macro_f1"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF model with high-weight real samples.")
    parser.add_argument("--real-input", default=str(DataConfig().data_dir / "real_sinhala_youtube_false_content.csv"))
    parser.add_argument(
        "--synthetic-input",
        default=str(DataConfig().data_dir / "sinhala_youtube_content_match_dataset.csv"),
    )
    parser.add_argument("--prepare-synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    parser.add_argument("--real-weight", type=float, default=150.0)
    parser.add_argument("--reviewed-synthetic-weight", type=float, default=50.0)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-features", type=int, default=60000)
    parser.add_argument("--char-features", type=int, default=120000)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    train(parse_args())


if __name__ == "__main__":
    main()
