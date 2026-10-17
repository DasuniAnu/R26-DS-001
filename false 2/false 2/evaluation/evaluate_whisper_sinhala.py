"""Evaluate Sinhala Whisper transcripts against human reference text."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.youtube_client import audio_transcript_status, transcribe_audio_with_whisper
from utils.data_utils import clean_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sinhala_asr_manifest.csv"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "whisper_sinhala_eval.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "whisper_sinhala_eval_predictions.csv"
PUNCTUATION_RE = re.compile(r"[^\w\s\u0d80-\u0dff]", flags=re.UNICODE)


def normalize_text(value: str) -> str:
    text = clean_text(value).lower()
    text = PUNCTUATION_RE.sub(" ", text)
    return clean_text(text)


def edit_distance(reference: list[str], prediction: list[str]) -> int:
    rows = len(reference) + 1
    cols = len(prediction) + 1
    previous = list(range(cols))
    for row in range(1, rows):
        current = [row] + [0] * (cols - 1)
        for col in range(1, cols):
            substitution_cost = 0 if reference[row - 1] == prediction[col - 1] else 1
            current[col] = min(
                previous[col] + 1,
                current[col - 1] + 1,
                previous[col - 1] + substitution_cost,
            )
        previous = current
    return previous[-1]


def wer(reference: str, prediction: str) -> float:
    reference_words = normalize_text(reference).split()
    prediction_words = normalize_text(prediction).split()
    if not reference_words:
        return 0.0 if not prediction_words else 1.0
    return edit_distance(reference_words, prediction_words) / len(reference_words)


def cer(reference: str, prediction: str) -> float:
    reference_chars = list(normalize_text(reference).replace(" ", ""))
    prediction_chars = list(normalize_text(prediction).replace(" ", ""))
    if not reference_chars:
        return 0.0 if not prediction_chars else 1.0
    return edit_distance(reference_chars, prediction_chars) / len(reference_chars)


def resolve_audio_path(value: str) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if split != "all":
        rows = [row for row in rows if clean_text(row.get("split", "")).lower() == split]
    return rows


def evaluate_manifest(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path, args.split)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows found for split '{args.split}' in {manifest_path}.")

    status = audio_transcript_status()
    if not status.get("whisper_backend"):
        raise RuntimeError(f"Whisper backend is not ready: {status}")

    output_rows = []
    total_word_errors = 0
    total_words = 0
    total_char_errors = 0
    total_chars = 0

    for index, row in enumerate(rows, start=1):
        audio_path = resolve_audio_path(row.get("audio_path", ""))
        reference = clean_text(row.get("transcript", ""))
        if not audio_path.exists():
            raise FileNotFoundError(f"Row {index}: audio file does not exist: {audio_path}")
        if not reference:
            raise ValueError(f"Row {index}: reference transcript is empty.")

        prediction, warnings = transcribe_audio_with_whisper(audio_path)
        row_wer = wer(reference, prediction)
        row_cer = cer(reference, prediction)
        reference_words = normalize_text(reference).split()
        reference_chars = list(normalize_text(reference).replace(" ", ""))
        word_errors = edit_distance(reference_words, normalize_text(prediction).split())
        char_errors = edit_distance(reference_chars, list(normalize_text(prediction).replace(" ", "")))
        total_word_errors += word_errors
        total_words += len(reference_words)
        total_char_errors += char_errors
        total_chars += len(reference_chars)

        output_rows.append(
            {
                "audio_path": str(audio_path),
                "split": clean_text(row.get("split", "")),
                "source": clean_text(row.get("source", "")),
                "reference": reference,
                "prediction": prediction,
                "wer": row_wer,
                "cer": row_cer,
                "warnings": " ".join(warnings),
            }
        )
        print(f"[{index}/{len(rows)}] WER={row_wer:.3f} CER={row_cer:.3f} {audio_path.name}")

    summary = {
        "manifest": str(manifest_path),
        "split": args.split,
        "samples": len(output_rows),
        "wer": total_word_errors / total_words if total_words else 0.0,
        "cer": total_char_errors / total_chars if total_chars else 0.0,
        "runtime_status": status,
        "predictions_csv": str(Path(args.output_csv)),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with Path(args.output_csv).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sinhala Whisper ASR WER/CER from a manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default="test", choices=("train", "validation", "test", "all"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate_manifest(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
