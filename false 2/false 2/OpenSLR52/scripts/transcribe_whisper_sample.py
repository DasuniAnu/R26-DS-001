import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe one audio file with a Whisper checkpoint.")
    parser.add_argument("--model-dir", default="runs/whisper-sinhala-tiny/final")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--language", default="sinhalese")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--cpu", action="store_true")
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)

    audio = decode_audio_16k(Path(args.audio_path))
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(text)


if __name__ == "__main__":
    main()
