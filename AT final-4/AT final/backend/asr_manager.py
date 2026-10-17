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


def transcribe_sinhala_audio(audio: np.ndarray) -> str:
    """Transcribe one 16 kHz mono YouTube segment as Sinhala text with CTC."""
    if audio is None or len(audio) == 0:
        return ""

    processor, model = _load_model()
    try:
        inputs = processor(
            audio.astype(np.float32),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = model(input_values=inputs["input_values"]).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        raise RuntimeError(f"Sinhala transcription failed: {exc}") from exc
