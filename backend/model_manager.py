# =============================================================
# MODEL MANAGER — LAZY LOADING + CACHING
#
# - CPU-only inference
# - Models loaded on first request, cached afterwards
# - Designed for 12GB RAM laptops
# =============================================================

import os
import gc
import torch
import numpy as np
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
)

from preprocessing import preprocess
from audio_utils import load_audio, SAMPLE_RATE, MAX_LENGTH


# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

TEXT_MODEL_PATH   = MODELS_DIR / "model_text_hatespeech" / "text_model_zip"
AUDIO_HATE_PATH   = MODELS_DIR / "model_audio_hatespeech"
AUDIO_FAKE_PATH   = MODELS_DIR / "model_audio_deepfake"

DEVICE = torch.device("cpu")

# ── Cache ────────────────────────────────────────────────────
_cache = {}


def _load_text_model():
    """Load XLM-RoBERTa text hate speech model (lazy)."""
    if "text_tokenizer" in _cache and "text_model" in _cache:
        return _cache["text_tokenizer"], _cache["text_model"]

    print("[ModelManager] Loading text hate speech model...")
    tokenizer = AutoTokenizer.from_pretrained(str(TEXT_MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(TEXT_MODEL_PATH),
        num_labels=2,
        id2label={0: "NOT", 1: "OFF"},
        label2id={"NOT": 0, "OFF": 1},
        ignore_mismatched_sizes=False,
    )
    model.eval()
    model.to(DEVICE)

    _cache["text_tokenizer"] = tokenizer
    _cache["text_model"]     = model
    print("[ModelManager] Text model loaded and cached.")
    return tokenizer, model


def _load_audio_hate_model():
    """Load Wav2Vec2 audio hate speech model (lazy)."""
    if "audio_hate_processor" in _cache and "audio_hate_model" in _cache:
        return _cache["audio_hate_processor"], _cache["audio_hate_model"]

    print("[ModelManager] Loading audio hate speech model...")
    processor = Wav2Vec2Processor.from_pretrained(str(AUDIO_HATE_PATH))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        str(AUDIO_HATE_PATH)
    )
    model.eval()
    model.to(DEVICE)

    _cache["audio_hate_processor"] = processor
    _cache["audio_hate_model"]     = model
    print("[ModelManager] Audio hate model loaded and cached.")
    return processor, model


def _load_audio_fake_model():
    """Load Wav2Vec2 audio deepfake model (lazy)."""
    if "audio_fake_processor" in _cache and "audio_fake_model" in _cache:
        return _cache["audio_fake_processor"], _cache["audio_fake_model"]

    print("[ModelManager] Loading audio deepfake model...")
    processor = Wav2Vec2Processor.from_pretrained(str(AUDIO_FAKE_PATH))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        str(AUDIO_FAKE_PATH)
    )
    model.eval()
    model.to(DEVICE)

    _cache["audio_fake_processor"] = processor
    _cache["audio_fake_model"]     = model
    print("[ModelManager] Audio deepfake model loaded and cached.")
    return processor, model


# =============================================================
# PUBLIC INFERENCE FUNCTIONS
# =============================================================

def predict_text(raw_text: str) -> dict:
    """
    Run text hate speech detection.

    Returns:
        {
            "input_text"       : original text,
            "processed_text"   : after preprocessing,
            "script_type"      : "unicode" | "romanized",
            "prediction"       : "OFF" | "NOT",
            "is_offensive"     : True/False,
            "confidence"       : 0.0 - 1.0,
            "probabilities"    : {"NOT": float, "OFF": float},
        }
    """
    # Preprocess — exact same pipeline as training
    processed_text, script_type = preprocess(raw_text)

    # Load model (lazy)
    tokenizer, model = _load_text_model()

    # Tokenize
    encoding = tokenizer(
        processed_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Inference
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
        )
        logits = outputs.logits
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = int(np.argmax(probs))

    id2label = {0: "NOT", 1: "OFF"}
    label    = id2label[pred]

    return {
        "input_text"     : raw_text,
        "processed_text" : processed_text,
        "script_type"    : script_type,
        "prediction"     : label,
        "is_offensive"   : label == "OFF",
        "confidence"     : float(probs[pred]),
        "probabilities"  : {
            "NOT": float(probs[0]),
            "OFF": float(probs[1]),
        },
    }


def predict_audio(filepath: str) -> dict:
    """
    Run both audio models on one audio file.

    Returns:
        {
            "hate_speech"             : True/False,
            "hate_label"              : "OFF" | "NOT",
            "hate_confidence"         : 0.0 - 1.0,
            "hate_probabilities"      : {"NOT": float, "OFF": float},
            "audio_authentic"         : True/False,
            "authenticity_label"      : "REAL" | "FAKE",
            "authenticity_confidence" : 0.0 - 1.0,
            "authenticity_probabilities": {"REAL": float, "FAKE": float},
        }
    """
    # Load audio — exact same preprocessing as training
    audio = load_audio(filepath)

    output = {}

    # ── Hate Speech Model ────────────────────────────────────
    h_proc, h_model = _load_audio_hate_model()

    inp = h_proc(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=MAX_LENGTH,
        truncation=True,
    )

    with torch.no_grad():
        logits = h_model(
            input_values=inp["input_values"].to(DEVICE)
        ).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred  = int(np.argmax(probs))

    output["hate_speech"]     = bool(pred == 1)
    output["hate_label"]      = "OFF" if pred == 1 else "NOT"
    output["hate_confidence"] = float(probs[pred])
    output["hate_probabilities"] = {
        "NOT": float(probs[0]),
        "OFF": float(probs[1]),
    }

    # ── Deepfake Model ───────────────────────────────────────
    f_proc, f_model = _load_audio_fake_model()

    inp = f_proc(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=MAX_LENGTH,
        truncation=True,
    )

    with torch.no_grad():
        logits = f_model(
            input_values=inp["input_values"].to(DEVICE)
        ).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred  = int(np.argmax(probs))

    output["audio_authentic"]     = bool(pred == 0)
    output["authenticity_label"]  = "REAL" if pred == 0 else "FAKE"
    output["authenticity_confidence"] = float(probs[pred])
    output["authenticity_probabilities"] = {
        "REAL": float(probs[0]),
        "FAKE": float(probs[1]),
    }

    return output
