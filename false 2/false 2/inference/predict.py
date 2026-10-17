"""Predict a Sinhala YouTube false-content label with the text-only XLM-R model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.default_config import DataConfig
from models.text_classifier import load_text_model_for_inference
from services.predictor import (
    analyze_input_evidence,
    build_prediction_warnings,
    choose_binary_label,
    fold_binary_probabilities,
    review_recommendation,
)
from utils.data_utils import format_text_input, is_meaningful_text


def predict(args: argparse.Namespace) -> dict:
    if not any(is_meaningful_text(value) for value in (args.title, args.description, args.transcript)):
        raise ValueError(
            "Enter meaningful title, description, or transcript text before predicting. "
            "Placeholders such as '...' are not enough for classification."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, id_to_label = load_text_model_for_inference(
        args.model_dir,
        device,
        local_files_only=args.local_files_only,
    )
    data_config = DataConfig(max_length=args.max_length)

    text = format_text_input(args.title, args.description, args.transcript)
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=data_config.max_length,
        return_tensors="pt",
    )
    inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }

    with torch.no_grad():
        logits = model(**inputs)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    class_probabilities = fold_binary_probabilities(probabilities, id_to_label)
    predicted_label, confidence = choose_binary_label(class_probabilities)
    evidence_signals = analyze_input_evidence(args.title, args.description, args.transcript)
    return {
        "predicted_label": predicted_label,
        "label": predicted_label,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "evidence_signals": evidence_signals,
        "warnings": list(build_prediction_warnings(confidence, evidence_signals, id_to_label)),
        "review_recommendation": review_recommendation(predicted_label, confidence, evidence_signals),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with the text-only Sinhala XLM-R model.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        result = predict(parse_args())
    except ValueError as exc:
        result = {"error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
