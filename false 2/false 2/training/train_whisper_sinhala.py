"""Fine-tune Whisper for Sinhala ASR from a local manifest.

Expected manifest columns:
audio_path, transcript, split, source, duration_seconds
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.validate_asr_manifest import PROJECT_ROOT, load_manifest, validate_asr_manifest


DEFAULT_MODEL = "openai/whisper-small"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "whisper_sinhala"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sinhala_asr_manifest.csv"


def import_training_dependencies():
    """Import optional Hugging Face training dependencies with a clear error."""

    try:
        import evaluate
        import torch
        from datasets import Audio, Dataset, DatasetDict
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install ASR training dependencies first: "
            "pip install datasets evaluate jiwer accelerate soundfile"
        ) from exc

    return {
        "Audio": Audio,
        "Dataset": Dataset,
        "DatasetDict": DatasetDict,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
        "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
        "WhisperProcessor": WhisperProcessor,
        "evaluate": evaluate,
        "torch": torch,
    }


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def manifest_to_dataset_dict(dataframe: pd.DataFrame, dataset_cls, dataset_dict_cls, audio_cls):
    """Create a DatasetDict split by train/validation/test."""

    split_datasets = {}
    prepared = dataframe.copy()
    prepared["split"] = prepared["split"].astype(str).str.lower()
    prepared["audio_path"] = prepared["audio_path"].apply(
        lambda value: str(Path(value)) if Path(value).is_absolute() else str(PROJECT_ROOT / str(value))
    )
    prepared = prepared.rename(columns={"audio_path": "audio"})

    for split_name, split_frame in prepared.groupby("split"):
        records = split_frame[["audio", "transcript"]].to_dict(orient="records")
        split_datasets[split_name] = dataset_cls.from_list(records).cast_column(
            "audio",
            audio_cls(sampling_rate=16000),
        )

    return dataset_dict_cls(split_datasets)


def train(args: argparse.Namespace) -> None:
    deps = import_training_dependencies()
    dataframe = load_manifest(args.manifest)
    errors = validate_asr_manifest(
        dataframe,
        base_dir=PROJECT_ROOT,
        max_duration_seconds=args.max_duration_seconds,
    )
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"ASR manifest validation failed:\n{details}")

    processor = deps["WhisperProcessor"].from_pretrained(
        args.model,
        language=args.language,
        task=args.task,
    )
    model = deps["WhisperForConditionalGeneration"].from_pretrained(args.model)
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    dataset = manifest_to_dataset_dict(
        dataframe,
        deps["Dataset"],
        deps["DatasetDict"],
        deps["Audio"],
    )

    def prepare_example(example):
        audio = example["audio"]
        example["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["transcript"]).input_ids
        return example

    dataset = dataset.map(
        prepare_example,
        remove_columns=next(iter(dataset.values())).column_names,
        num_proc=args.num_proc,
    )

    wer_metric = deps["evaluate"].load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=pred_text, references=label_text)}

    output_dir = Path(args.output_dir)
    training_args = deps["Seq2SeqTrainingArguments"](
        output_dir=str(output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=deps["torch"].cuda.is_available(),
        evaluation_strategy="epoch" if "validation" in dataset else "no",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=args.logging_steps,
        report_to=[],
        load_best_model_at_end="validation" in dataset,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = deps["Seq2SeqTrainer"](
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=WhisperDataCollator(processor=processor),
        compute_metrics=compute_metrics if "validation" in dataset else None,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    if "test" in dataset:
        metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
        print(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Sinhala transcription.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--language", default="sinhalese")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
