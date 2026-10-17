"""Lazy Sinhala speech-to-text support used only by YouTube analysis."""

import os
import threading

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor


_processor = None
_model = None
_model_lock = threading.Lock()
MODEL_ID = os.environ.get(
    "SINHALA_ASR_MODEL",
    "janiduchamika/wav2vec2-xls-r-300m-sinhala-general-185k",
)
SAMPLE_RATE = 16000
# Heuristic screening only: these are not calibrated probabilities of correctness.
MIN_TOKEN_SCORE = 0.65
MAX_WEAK_TOKEN_FRACTION = 0.25


def _load_model():
    """Load the fine-tuned Sinhala XLS-R CTC model once, on first use."""
    global _processor, _model
    if _processor is None or _model is None:
        with _model_lock:
            if _processor is None or _model is None:
                print(f"[ASR] Loading Sinhala XLS-R model: {MODEL_ID}")
                # Prefer the cached checkpoint. This avoids an unnecessary
                # Hugging Face network check every time the backend restarts.
                try:
                    _processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
                    _model = AutoModelForCTC.from_pretrained(MODEL_ID, local_files_only=True)
                except OSError:
                    _processor = AutoProcessor.from_pretrained(MODEL_ID)
                    _model = AutoModelForCTC.from_pretrained(MODEL_ID)
                _model.eval()
                _model.to("cpu")
                print("[ASR] Sinhala XLS-R model loaded and cached.")
    return _processor, _model


def decode_segment(logits, processor, blank_id, frame_start=0, frame_end=None):
    """Collapse CTC runs before selecting the central interval.

    Assign each run by its midpoint, preserving boundary context without
    including entire neighbouring transcripts. Blank frames never inflate scores.
    """
    probabilities = torch.softmax(logits.float(), dim=-1)
    scores, ids = probabilities.max(dim=-1)
    ids, scores = ids.tolist(), scores.tolist()
    frame_end = len(ids) if frame_end is None else frame_end
    tokens, token_scores = [], []
    start = 0
    while start < len(ids):
        end = start + 1
        while end < len(ids) and ids[end] == ids[start]:
            end += 1
        midpoint = (start + end - 1) / 2
        if ids[start] != blank_id and frame_start <= midpoint < frame_end:
            tokens.append(ids[start])
            token_scores.append(max(scores[start:end]))
        start = end
    text = processor.tokenizer.decode(
        tokens, group_tokens=False, skip_special_tokens=True,
    ).strip()
    score = float(np.mean(token_scores)) if token_scores else 0.0
    weak_fraction = (sum(s < MIN_TOKEN_SCORE for s in token_scores) / len(token_scores)
                     if token_scores else 1.0)
    needs_review = not text or score < MIN_TOKEN_SCORE or weak_fraction > MAX_WEAK_TOKEN_FRACTION
    return {
        "text": text,
        "score": round(score, 4),
        "needs_review": needs_review,
        "reason": ("Recognition is uncertain; excluded from text hate classification."
                   if needs_review else ""),
    }


def transcribe_sinhala_audio(audio: np.ndarray, core_start=0, core_end=None) -> dict:
    """Transcribe original 16 kHz audio with context; return only the core interval."""
    if audio is None or len(audio) == 0:
        return {"text": "", "score": 0.0, "needs_review": True, "reason": "No audio samples."}
    core_end = len(audio) if core_end is None else core_end
    if not 0 <= core_start < core_end <= len(audio):
        raise ValueError("Invalid transcription interval")
    if not np.isfinite(audio).all():
        return {"text": "", "score": 0.0, "needs_review": True, "reason": "Invalid audio samples."}
    if np.max(np.abs(audio[core_start:core_end])) < 1e-7:
        return {"text": "", "score": 0.0, "needs_review": True, "reason": "Silent audio interval."}

    processor, model = _load_model()
    try:
        inputs = processor(
            audio.astype(np.float32),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        # Wav2Vec2 emits one frame per convolutional stride. Use its actual
        # receptive-field centre instead of scaling by the padded clip length.
        stride, receptive_field = 1, 1
        for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride):
            receptive_field += (kernel - 1) * stride
            stride *= step
        offset = (receptive_field - 1) / 2
        return decode_segment(
            logits, processor, model.config.pad_token_id,
            (core_start - offset) / stride, (core_end - offset) / stride,
        )
    except Exception as exc:
        raise RuntimeError(f"Sinhala transcription failed: {exc}") from exc
