import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCTC, AutoProcessor


DEFAULT_MODEL = "janiduchamika/wav2vec2-xls-r-300m-sinhala-general-185k"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe Sinhala audio with a Wav2Vec2 XLS-R CTC model."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--metadata",
        default="slr52_metadata.csv",
        help="CSV with audio_path and optional reference columns.",
    )
    parser.add_argument(
        "--audio-path",
        default="",
        help="Transcribe one audio file instead of a metadata CSV.",
    )
    parser.add_argument("--output-csv", default="xlsr_transcripts.csv")
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached Hugging Face files and do not contact the network.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use half precision on CUDA. Leave off if outputs look unstable.",
    )
    return parser.parse_args()


def decode_audio_16k(path: Path) -> np.ndarray:
    cmd = [
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
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def transcribe(audio_path: Path, processor, model, device):
    audio = decode_audio_16k(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0].strip()


def load_model(model_name, device, use_fp16, local_files_only=False):
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForCTC.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device)
    if device.type == "cuda" and use_fp16:
        model.half()
    model.eval()
    return processor, model


def single_audio(args, root, processor, model, device):
    audio_path = Path(args.audio_path)
    if not audio_path.is_absolute():
        audio_path = root / audio_path
    text = transcribe(audio_path, processor, model, device)
    print(text)


def batch_csv(args, root, processor, model, device):
    metadata_path = root / args.metadata
    out_path = root / args.output_csv
    df = pd.read_csv(metadata_path, encoding="utf-8")
    if "audio_path" not in df.columns:
        raise SystemExit("metadata CSV must contain an audio_path column")

    if args.start_index:
        df = df.iloc[args.start_index :].copy()
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    fieldnames = [
        "audio_path",
        "predicted_transcription",
        "reference_transcription",
        "speaker_id",
        "utterance_id",
        "part",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in df.reset_index(drop=True).iterrows():
            relative_audio = str(row["audio_path"])
            audio_path = root / relative_audio
            prediction = transcribe(audio_path, processor, model, device)
            writer.writerow(
                {
                    "audio_path": relative_audio,
                    "predicted_transcription": prediction,
                    "reference_transcription": row.get("transcription", ""),
                    "speaker_id": row.get("speaker_id", ""),
                    "utterance_id": row.get("utterance_id", ""),
                    "part": row.get("part", ""),
                }
            )
            handle.flush()
            print(f"[{idx + 1}/{len(df)}] {relative_audio} -> {prediction}")

    print(f"wrote: {out_path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    root = Path.cwd()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("warning: CPU inference works, but this 300M model will be slow over all 34k files.")

    processor, model = load_model(args.model_name, device, args.fp16, args.local_files_only)
    if args.audio_path:
        single_audio(args, root, processor, model, device)
    else:
        batch_csv(args, root, processor, model, device)


if __name__ == "__main__":
    main()
