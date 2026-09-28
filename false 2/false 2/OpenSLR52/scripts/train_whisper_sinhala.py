import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Sinhala SLR52 CSV splits.")
    parser.add_argument("--model-name", default="openai/whisper-tiny")
    parser.add_argument("--train-csv", default="metadata/train.csv")
    parser.add_argument("--eval-csv", default="metadata/eval.csv")
    parser.add_argument("--output-dir", default="runs/whisper-sinhala-tiny")
    parser.add_argument("--language", default="sinhalese")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-label-length", type=int, default=448)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Optional limit for smoke tests. 0 uses all training rows.",
    )
    parser.add_argument(
        "--limit-eval",
        type=int,
        default=256,
        help="Eval row limit to keep validation quick. 0 uses all eval rows.",
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
    audio = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
    return audio


class SpeechCsvDataset(Dataset):
    def __init__(self, csv_path, processor, root, max_label_length, limit=0):
        self.root = Path(root)
        self.processor = processor
        self.max_label_length = max_label_length
        df = pd.read_csv(csv_path, encoding="utf-8")
        if limit and limit > 0:
            df = df.head(limit)
        self.rows = df.to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio_path = self.root / row["audio_path"]
        audio = decode_audio_16k(audio_path)
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_attention_mask=False,
        ).input_features[0]
        labels = self.processor.tokenizer(
            str(row["transcription"]),
            max_length=self.max_label_length,
            truncation=True,
        ).input_ids
        return {"input_features": features, "labels": labels}


class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def evaluate_loss(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            losses.append(loss.detach().float().cpu().item())
    model.train()
    return float(np.mean(losses)) if losses else math.nan


def save_checkpoint(model, processor, output_dir, step_name):
    path = Path(output_dir) / step_name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"saved: {path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path.cwd()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda_ok = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if cuda_ok else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("warning: CPU training works for smoke tests, but full fine-tuning will be very slow.")

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language,
        task=args.task,
    )
    model.config.suppress_tokens = []
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    train_ds = SpeechCsvDataset(
        args.train_csv,
        processor,
        root,
        args.max_label_length,
        limit=args.limit_train,
    )
    eval_ds = SpeechCsvDataset(
        args.eval_csv,
        processor,
        root,
        args.max_label_length,
        limit=args.limit_eval,
    )
    collator = WhisperCollator(processor)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    progress = tqdm(total=args.max_steps, desc="training")
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(**batch).loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += loss.detach().float().cpu().item()

            if (global_step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{running_loss:.4f}")

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                eval_loss = evaluate_loss(model, eval_loader, device)
                print(f"step={global_step} train_loss_window={running_loss:.4f} eval_loss={eval_loss:.4f}")
                running_loss = 0.0

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, processor, output_dir, f"checkpoint-{global_step}")

            if global_step >= args.max_steps:
                break

    progress.close()
    save_checkpoint(model, processor, output_dir, "final")


if __name__ == "__main__":
    main()
