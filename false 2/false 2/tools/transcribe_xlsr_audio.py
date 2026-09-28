"""Transcribe one audio file with the Sinhala XLS-R CTC model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor


DEFAULT_MODEL = "janiduchamika/wav2vec2-xls-r-300m-sinhala-general-185k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe Sinhala audio with XLS-R.")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def decode_audio_16k(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    audio_path = Path(args.audio_path).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForCTC.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    audio = decode_audio_16k(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    print(processor.batch_decode(predicted_ids)[0].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
