"""Evaluate the text-only XLM-RoBERTa baseline on the test split."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.default_config import BINARY_LABEL_ORDER, DataConfig, TrainingConfig
from data.text_dataset import SinhalaTextDataset
from evaluation.metrics import compute_metrics, save_confusion_matrix_plot, save_metrics
from models.text_classifier import load_text_model_for_inference
from services.predictor import choose_binary_label, fold_binary_probabilities
from utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def log_split_report(data_config: DataConfig) -> None:
    for split_name, path in [
        ("train", data_config.train_csv),
        ("val", data_config.val_csv),
        ("test", data_config.test_csv),
    ]:
        df = pd.read_csv(path, encoding="utf-8-sig")
        distribution = df["label"].value_counts().sort_index().to_dict()
        LOGGER.info("%s size: %s", split_name, len(df))
        LOGGER.info("%s group count: %s", split_name, df["group_id"].nunique())
        LOGGER.info("%s label distribution: %s", split_name, distribution)


def assert_no_group_leakage(data_config: DataConfig) -> None:
    """Stop evaluation if any group_id appears in more than one split."""

    split_frames = {
        "train": pd.read_csv(data_config.train_csv, encoding="utf-8-sig"),
        "validation": pd.read_csv(data_config.val_csv, encoding="utf-8-sig"),
        "test": pd.read_csv(data_config.test_csv, encoding="utf-8-sig"),
    }
    split_groups = {name: set(df["group_id"]) for name, df in split_frames.items()}
    overlaps = {
        "train_validation": split_groups["train"] & split_groups["validation"],
        "train_test": split_groups["train"] & split_groups["test"],
        "validation_test": split_groups["validation"] & split_groups["test"],
    }
    leaking = {name: sorted(groups) for name, groups in overlaps.items() if groups}
    if leaking:
        raise RuntimeError(f"Group leakage detected. Stop evaluation. Overlaps: {leaking}")


def evaluate(args: argparse.Namespace) -> None:
    data_config = DataConfig(max_length=args.max_length)
    training_config = TrainingConfig(batch_size=args.batch_size)
    model_dir = Path(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Evaluating text-only baseline. Image branch is disabled.")
    assert_no_group_leakage(data_config)
    LOGGER.info("No group_id leakage found across train/validation/test.")
    log_split_report(data_config)

    model, tokenizer, id_to_label = load_text_model_for_inference(
        model_dir,
        device,
        local_files_only=args.local_files_only,
    )
    label_to_id = {label: idx for idx, label in enumerate(BINARY_LABEL_ORDER)}
    labels = list(BINARY_LABEL_ORDER)
    test_csv = data_config.test_csv
    temp_test_file = None
    if args.human_only:
        test_frame = pd.read_csv(data_config.test_csv, encoding="utf-8-sig").fillna("")
        if "generated_negative" in test_frame.columns:
            test_frame = test_frame[test_frame["generated_negative"].astype(str).str.lower() != "true"].copy()
        if test_frame.empty:
            raise ValueError("No human-reviewed benchmark rows found after excluding generated negatives.")
        temp_test_file = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
        test_frame.to_csv(temp_test_file.name, index=False, encoding="utf-8-sig")
        temp_test_file.close()
        test_csv = Path(temp_test_file.name)
        LOGGER.info("Human-only benchmark rows: %s", len(test_frame))
    test_dataset = SinhalaTextDataset(test_csv, tokenizer, label_to_id, data_config)
    test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False)

    y_true: list[int] = []
    y_pred: list[int] = []
    false_scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            labels_tensor = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch)
            probabilities = torch.softmax(logits, dim=1)
            preds = []
            for row in probabilities:
                binary_probabilities = fold_binary_probabilities(row, id_to_label)
                predicted_label, _ = choose_binary_label(binary_probabilities)
                preds.append(label_to_id[predicted_label])
                false_scores.append(binary_probabilities.get("False", 0.0))
            y_true.extend(labels_tensor.cpu().tolist())
            y_pred.extend(preds)

    metrics = compute_metrics(y_true, y_pred, labels, y_score=false_scores)
    metrics_path = training_config.metrics_path
    confusion_path = training_config.confusion_matrix_path
    report_path = training_config.output_dir / "classification_report.txt"
    save_metrics(metrics, metrics_path)
    save_confusion_matrix_plot(y_true, y_pred, labels, confusion_path)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
    )
    report_path.write_text(report_text, encoding="utf-8")

    LOGGER.info("Accuracy: %.4f", metrics["accuracy"])
    LOGGER.info("Macro F1: %.4f", metrics["macro_f1"])
    LOGGER.info("Weighted F1: %.4f", metrics["weighted_f1"])
    LOGGER.info("Confusion matrix: %s", metrics["confusion_matrix"])
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved confusion matrix to %s", confusion_path)
    LOGGER.info("Saved classification report to %s", report_path)
    if temp_test_file is not None:
        Path(temp_test_file.name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text-only XLM-RoBERTa baseline.")
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="Evaluate only human-reviewed rows by excluding generated_negative=true examples.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    evaluate(parse_args())


if __name__ == "__main__":
    main()
