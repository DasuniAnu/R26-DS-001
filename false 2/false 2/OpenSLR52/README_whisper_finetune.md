# Sinhala SLR52 Whisper Fine-Tuning

This folder contains a local fine-tuning setup for the downloaded OpenSLR52 Sinhala subset.

## Data

The metadata file is:

```powershell
slr52_metadata.csv
```

Each audio path maps to a transcription by filename stem:

```text
data/part0/asr_sinhala/data/00/0000f47c22.flac -> utt_spk_text.tsv utterance_id 0000f47c22
```

## Prepare Splits

```powershell
python .\scripts\prepare_whisper_splits.py
```

For a tiny smoke-test split:

```powershell
python .\scripts\prepare_whisper_splits.py --limit 200
```

## Fine-Tune

Your current PyTorch install is CPU-only. To use the NVIDIA GPU, install a CUDA-enabled
PyTorch wheel first. For your driver-reported CUDA 12.6 support, use:

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Default tiny model:

```powershell
python .\scripts\train_whisper_sinhala.py --model-name openai/whisper-tiny --max-steps 2000
```

For a quick CPU smoke test:

```powershell
python .\scripts\train_whisper_sinhala.py --model-name openai/whisper-tiny --max-steps 2 --limit-train 4 --limit-eval 2 --cpu
```

For better quality after CUDA-enabled PyTorch is installed, try:

```powershell
python .\scripts\train_whisper_sinhala.py --model-name openai/whisper-base --max-steps 4000 --batch-size 1 --grad-accum-steps 16
```

## Transcribe One File

```powershell
python .\scripts\transcribe_whisper_sample.py --model-dir .\runs\whisper-sinhala-tiny\final --audio-path .\data\part0\asr_sinhala\data\00\0000f47c22.flac
```

## Alternative: Use Sinhala XLS-R ASR Directly

You can use `janiduchamika/wav2vec2-xls-r-300m-sinhala-general-185k` to generate
transcripts without fine-tuning Whisper first:

```powershell
python .\scripts\transcribe_xlsr_sinhala.py --audio-path .\data\part0\asr_sinhala\data\00\0000f47c22.flac --local-files-only
```

Batch over the metadata CSV:

```powershell
python .\scripts\transcribe_xlsr_sinhala.py --metadata .\slr52_metadata.csv --output-csv .\xlsr_transcripts.csv --limit 100 --local-files-only
```

Remove `--limit 100` to transcribe the full downloaded subset.

## Notes

- Whisper's language name in Hugging Face is usually `sinhalese`, even though the language is commonly called Sinhala.
- The current installed PyTorch is CPU-only. Full fine-tuning will be slow until CUDA-enabled PyTorch is installed.
- Original ZIP files are not needed for training but are preserved.
