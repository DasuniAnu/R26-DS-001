"""Train a text-only XLM-RoBERTa baseline end-to-end."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.default_config import DataConfig, TrainingConfig
from data.text_dataset import SinhalaTextDataset
from evaluation.metrics import compute_metrics, save_metrics
from models.text_classifier import XLMRTextClassifier, save_text_model_artifacts
from utils.logging_utils import setup_logging
from utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def load_label_mapping(model_dir: Path) -> dict[str, int]:
    mapping_path = model_dir.parent / "label_mapping.json"
    if not mapping_path.exists():
        mapping_path = model_dir / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError("Run training/prepare_data.py before text baseline training.")
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def assert_no_group_leakage(data_config: DataConfig) -> None:
    """Stop training if any group_id appears in more than one split."""

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
        raise RuntimeError(f"Group leakage detected. Stop training. Overlaps: {leaking}")


def log_split_report(data_config: DataConfig) -> None:
    """Log dataset sizes, label distributions, topic distributions, and group counts."""

    for split_name, path in [
        ("train", data_config.train_csv),
        ("validation", data_config.val_csv),
        ("test", data_config.test_csv),
    ]:
        df = pd.read_csv(path, encoding="utf-8-sig")
        label_distribution = df["label"].value_counts().sort_index().to_dict()
        topic_distribution = df["topic"].value_counts().sort_index().to_dict()
        LOGGER.info("%s size: %s", split_name, len(df))
        LOGGER.info("%s group count: %s", split_name, df["group_id"].nunique())
        LOGGER.info("%s label distribution: %s", split_name, label_distribution)
        LOGGER.info("%s topic distribution: %s", split_name, topic_distribution)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: XLMRTextClassifier,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, list[int], list[int]]:
    """Run a training or evaluation epoch."""

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    if is_training:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(data_loader, desc="train" if is_training else "eval")
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**batch)
                raw_loss = criterion(logits, labels)
            if is_training:
                loss = raw_loss / gradient_accumulation_steps
                should_step = step % gradient_accumulation_steps == 0 or step == len(data_loader)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    if should_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                if should_step and scheduler is not None:
                    scheduler.step()
            else:
                raw_loss = criterion(logits, labels)

        total_loss += raw_loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        progress.set_postfix(loss=raw_loss.item())

    average_loss = total_loss / max(len(data_loader.dataset), 1)
    return average_loss, all_labels, all_preds


def train(args: argparse.Namespace) -> None:
    data_config = DataConfig(max_length=args.max_length)
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
    )
    set_seed(training_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    micro_batch_size = args.micro_batch_size
    if micro_batch_size is None:
        micro_batch_size = 1 if device.type == "cpu" else training_config.batch_size
    if micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be >= 1.")
    gradient_accumulation_steps = max(1, math.ceil(training_config.batch_size / micro_batch_size))
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Mixed precision enabled: %s", use_amp)
    LOGGER.info("Effective batch size: %s", training_config.batch_size)
    LOGGER.info("Micro batch size: %s", micro_batch_size)
    LOGGER.info("Gradient accumulation steps: %s", gradient_accumulation_steps)
    LOGGER.info("Training text-only XLM-R baseline with Sinhala text kept unchanged.")
    assert_no_group_leakage(data_config)
    LOGGER.info("No group_id leakage found across train/validation/test.")
    log_split_report(data_config)

    label_to_id = load_label_mapping(training_config.model_dir)
    labels = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]

    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model,
        local_files_only=args.local_files_only,
    )
    train_dataset = SinhalaTextDataset(data_config.train_csv, tokenizer, label_to_id, data_config)
    val_dataset = SinhalaTextDataset(data_config.val_csv, tokenizer, label_to_id, data_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    model = XLMRTextClassifier(
        num_labels=len(label_to_id),
        model_name=args.text_model,
        dropout=args.dropout,
        local_files_only=args.local_files_only,
    ).to(device)
    if args.gradient_checkpointing and hasattr(model.text_encoder, "gradient_checkpointing_enable"):
        model.text_encoder.gradient_checkpointing_enable()
        LOGGER.info("Gradient checkpointing enabled.")
    LOGGER.info("Fine-tuning all XLM-R parameters end-to-end.")

    optimizer_name = args.optimizer
    if optimizer_name == "auto":
        optimizer_name = "sgd" if device.type == "cpu" else "adamw"
    if optimizer_name == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif optimizer_name == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    LOGGER.info("Optimizer: %s", optimizer_name)
    total_steps = math.ceil(len(train_loader) / gradient_accumulation_steps) * training_config.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    text_output_dir = training_config.model_dir
    best_false_recall = -1.0
    best_macro_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, training_config.epochs + 1):
        LOGGER.info("Epoch %s/%s", epoch, training_config.epochs)
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            scaler=scaler,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        val_loss, val_true, val_pred = run_epoch(model, val_loader, None, None, criterion, device)
        val_metrics = compute_metrics(val_true, val_pred, labels)
        false_recall = float(
            val_metrics["classification_report"].get("False", {}).get("recall", 0.0)
        )
        LOGGER.info("Train loss: %.4f | Val loss: %.4f", train_loss, val_loss)
        LOGGER.info(
            "Val accuracy: %.4f | Val macro F1: %.4f | Val False recall: %.4f",
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            false_recall,
        )
        LOGGER.info("Val confusion matrix: %s", val_metrics["confusion_matrix"])

        improved = false_recall > best_false_recall or (
            false_recall == best_false_recall and val_metrics["macro_f1"] > best_macro_f1
        )
        if improved:
            best_false_recall = false_recall
            best_macro_f1 = val_metrics["macro_f1"]
            bad_epochs = 0
            save_text_model_artifacts(model, tokenizer, label_to_id, text_output_dir)
            save_metrics(val_metrics, training_config.output_dir / "validation_metrics.json")
            LOGGER.info("Saved best text model to %s", text_output_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= training_config.patience:
                LOGGER.info("Early stopping after %s epochs without improvement.", bad_epochs)
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text-only XLM-RoBERTa baseline.")
    parser.add_argument("--text-model", default="xlm-roberta-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Per-step batch size. Defaults to 1 on CPU and --batch-size on GPU.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--optimizer",
        choices=["auto", "adamw", "adafactor", "sgd"],
        default="auto",
        help="auto uses SGD on CPU to reduce memory and AdamW on GPU.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce activation memory while fine-tuning the full encoder.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    train(parse_args())


if __name__ == "__main__":
    main()
