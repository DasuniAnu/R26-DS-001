"""Train a fast TF-IDF classifier for Sinhala title/description/transcript matching."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.default_config import BINARY_FALSE_LABEL, BINARY_LABEL_ORDER, DataConfig, TrainingConfig
from evaluation.metrics import compute_metrics, save_metrics
from training.prepare_data import prepare_dataset
from utils.logging_utils import setup_logging
from utils.seed import set_seed


LOGGER = logging.getLogger(__name__)
ARTIFACT_NAME = "sklearn_text_classifier.joblib"


def load_label_mapping(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Label mapping not found: {path}. Run prepare_data first.")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_split_files(data_config: DataConfig) -> None:
    missing = [
        path
        for path in (data_config.train_csv, data_config.val_csv, data_config.test_csv)
        if not path.exists()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing prepared split file(s): {names}")


def load_split(path: Path, label_to_id: dict[str, int]) -> tuple[pd.Series, list[int]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain text and label columns. Run training/prepare_data.py first.")
    unknown = sorted(set(df["label"].astype(str)).difference(label_to_id))
    if unknown:
        raise ValueError(f"{path} contains labels missing from label_mapping.json: {unknown}")
    return df["text"].fillna("").astype(str), df["label"].map(label_to_id).astype(int).tolist()


def make_pipeline(args: argparse.Namespace) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=args.min_df,
                    max_features=args.word_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=args.min_df,
                    max_features=args.char_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def evaluate_split(model: Pipeline, texts: pd.Series, labels: list[int], label_names: list[str]) -> dict:
    predictions = model.predict(texts).astype(int).tolist()
    probabilities = model.predict_proba(texts)
    false_index = label_names.index(BINARY_FALSE_LABEL)
    false_scores = probabilities[:, false_index].tolist()
    return compute_metrics(labels, predictions, label_names, y_score=false_scores)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_config = DataConfig()
    training_config = TrainingConfig(seed=args.seed)

    if args.prepare_data:
        LOGGER.info("Preparing dataset from %s", args.input)
        prepare_dataset(args.input, data_config, training_config)

    assert_split_files(data_config)
    label_to_id = load_label_mapping(training_config.model_dir / "label_mapping.json")
    label_names = [label for label, _idx in sorted(label_to_id.items(), key=lambda item: item[1])]
    if set(label_names) != set(BINARY_LABEL_ORDER):
        raise ValueError(f"This trainer expects binary labels {BINARY_LABEL_ORDER}, got {label_names}")

    train_texts, train_labels = load_split(data_config.train_csv, label_to_id)
    val_texts, val_labels = load_split(data_config.val_csv, label_to_id)
    test_texts, test_labels = load_split(data_config.test_csv, label_to_id)

    LOGGER.info("Training rows: %s", len(train_texts))
    LOGGER.info("Validation rows: %s", len(val_texts))
    LOGGER.info("Test rows: %s", len(test_texts))
    LOGGER.info("Label mapping: %s", label_to_id)

    model = make_pipeline(args)
    model.fit(train_texts, train_labels)

    val_metrics = evaluate_split(model, val_texts, val_labels, label_names)
    test_metrics = evaluate_split(model, test_texts, test_labels, label_names)
    metrics = {
        "model_type": "sklearn_tfidf_logistic_regression",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "train_rows": len(train_texts),
            "validation_rows": len(val_texts),
            "test_rows": len(test_texts),
        },
        "label_mapping": label_to_id,
        "validation": val_metrics,
        "test": test_metrics,
        "benchmark_warning": (
            "These metrics are from the synthetic/generated content-match dataset. "
            "Use a human-reviewed real-video benchmark before claiming real-world accuracy."
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
            "model_type": "sklearn_tfidf_logistic_regression",
            "created_at": metrics["trained_at"],
        },
        artifact_path,
    )
    metrics_path = training_config.output_dir / "sklearn_text_metrics.json"
    save_metrics(metrics, metrics_path)
    LOGGER.info("Saved sklearn text model to %s", artifact_path)
    LOGGER.info("Saved sklearn metrics to %s", metrics_path)
    LOGGER.info("Validation macro F1: %.4f", val_metrics["macro_f1"])
    LOGGER.info("Test macro F1: %.4f", test_metrics["macro_f1"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fast sklearn text classifier for content matching.")
    parser.add_argument(
        "--input",
        default=str(DataConfig().data_dir / "sinhala_youtube_content_match_dataset.csv"),
        help="Content-match CSV to prepare before training.",
    )
    parser.add_argument("--prepare-data", action="store_true", help="Regenerate train/validation/test splits first.")
    parser.add_argument("--seed", type=int, default=42)
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
