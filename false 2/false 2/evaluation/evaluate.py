"""Evaluate a trained multimodal classifier on the test split."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.default_config import DataConfig, TrainingConfig
from data.dataset import SinhalaYouTubeDataset
from evaluation.metrics import compute_metrics, save_confusion_matrix_plot, save_metrics
from models.multimodal_classifier import load_model_for_inference
from utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def evaluate(args: argparse.Namespace) -> None:
    data_config = DataConfig(max_length=args.max_length)
    training_config = TrainingConfig(batch_size=args.batch_size)
    model_dir = Path(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    model, tokenizer, image_processor, id_to_label, _ = load_model_for_inference(
        model_dir,
        device,
        local_files_only=args.local_files_only,
    )
    label_to_id = {label: idx for idx, label in id_to_label.items()}
    labels = [id_to_label[idx] for idx in sorted(id_to_label)]

    test_dataset = SinhalaYouTubeDataset(
        data_config.test_csv,
        tokenizer,
        image_processor,
        label_to_id,
        data_config,
    )
    test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False)

    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            labels_tensor = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels_tensor.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, labels)
    save_metrics(metrics, training_config.metrics_path)
    save_confusion_matrix_plot(y_true, y_pred, labels, training_config.confusion_matrix_path)

    LOGGER.info("Accuracy: %.4f", metrics["accuracy"])
    LOGGER.info("Macro F1: %.4f", metrics["macro_f1"])
    LOGGER.info("Weighted F1: %.4f", metrics["weighted_f1"])
    LOGGER.info("Saved metrics to %s", training_config.metrics_path)
    LOGGER.info("Saved confusion matrix to %s", training_config.confusion_matrix_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    evaluate(parse_args())


if __name__ == "__main__":
    main()
