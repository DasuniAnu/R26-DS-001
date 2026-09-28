"""Train the multimodal Sinhala YouTube false-content classifier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.default_config import DataConfig, ModelConfig, TrainingConfig
from data.dataset import SinhalaYouTubeDataset
from evaluation.metrics import compute_metrics, save_metrics
from models.multimodal_classifier import (
    MultimodalClassifier,
    load_tokenizer_and_image_processor,
    save_model_artifacts,
)
from utils.logging_utils import setup_logging
from utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def load_label_mapping(model_dir: Path) -> dict[str, int]:
    mapping_path = model_dir / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError("Run training/prepare_data.py before training.")
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: MultimodalClassifier,
    data_loader: DataLoader,
    optimizer: AdamW | None,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Run one train or evaluation epoch."""

    is_training = optimizer is not None
    model.train(is_training)
    if model.model_config.freeze_encoders:
        model.text_encoder.eval()
        model.image_encoder.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    progress = tqdm(data_loader, desc="train" if is_training else "eval")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")

        with torch.set_grad_enabled(is_training):
            logits = model(**batch)
            loss = criterion(logits, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        progress.set_postfix(loss=loss.item())

    average_loss = total_loss / max(len(data_loader.dataset), 1)
    return average_loss, all_labels, all_preds


def train(args: argparse.Namespace) -> None:
    freeze_encoders = not args.fine_tune_encoders
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-3 if freeze_encoders else 2e-5

    data_config = DataConfig(max_length=args.max_length)
    model_config = ModelConfig(
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        fusion_hidden_size=args.fusion_hidden_size,
        dropout=args.dropout,
        freeze_encoders=freeze_encoders,
        local_files_only=args.local_files_only,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
    )
    set_seed(training_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Using learning rate: %s", training_config.learning_rate)

    label_to_id = load_label_mapping(training_config.model_dir)
    labels = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]

    tokenizer, image_processor = load_tokenizer_and_image_processor(model_config)
    train_dataset = SinhalaYouTubeDataset(
        data_config.train_csv,
        tokenizer,
        image_processor,
        label_to_id,
        data_config,
    )
    val_dataset = SinhalaYouTubeDataset(
        data_config.val_csv,
        tokenizer,
        image_processor,
        label_to_id,
        data_config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    model = MultimodalClassifier(num_labels=len(label_to_id), model_config=model_config).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    LOGGER.info(
        "Training %s parameters; encoders are %s.",
        sum(parameter.numel() for parameter in trainable_parameters),
        "frozen" if model_config.freeze_encoders else "trainable",
    )
    optimizer = AdamW(
        trainable_parameters,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, training_config.epochs + 1):
        LOGGER.info("Epoch %s/%s", epoch, training_config.epochs)
        train_loss, train_true, train_pred = run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_true, val_pred = run_epoch(model, val_loader, None, criterion, device)
        val_metrics = compute_metrics(val_true, val_pred, labels)

        LOGGER.info("Train loss: %.4f | Val loss: %.4f", train_loss, val_loss)
        LOGGER.info("Val macro F1: %.4f | Val accuracy: %.4f", val_metrics["macro_f1"], val_metrics["accuracy"])

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            bad_epochs = 0
            save_model_artifacts(
                model,
                tokenizer,
                image_processor,
                label_to_id,
                training_config.model_dir,
                save_full_model=args.save_full_model,
            )
            save_metrics(val_metrics, training_config.output_dir / "val_metrics.json")
            LOGGER.info("Saved new best model to %s", training_config.model_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= training_config.patience:
                LOGGER.info("Early stopping after %s epochs without improvement.", bad_epochs)
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal classifier.")
    parser.add_argument("--text-model", default="xlm-roberta-base")
    parser.add_argument("--image-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Defaults to 1e-3 for frozen encoders and 2e-5 when fine-tuning encoders.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fusion-hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files. Helpful when internet/DNS is unavailable.",
    )
    parser.add_argument(
        "--fine-tune-encoders",
        action="store_true",
        help="Train XLM-R and image encoder weights too. This is slower and needs a large checkpoint.",
    )
    parser.add_argument(
        "--save-full-model",
        action="store_true",
        help="Save the full encoder+classifier state_dict as pytorch_model.bin.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    train(parse_args())


if __name__ == "__main__":
    main()
