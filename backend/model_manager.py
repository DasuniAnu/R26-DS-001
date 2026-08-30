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
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
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
    """Load a Wav2Vec2 or WavLM audio deepfake classifier (lazy)."""
    if "audio_fake_processor" in _cache and "audio_fake_model" in _cache:
        return _cache["audio_fake_processor"], _cache["audio_fake_model"]

    print("[ModelManager] Loading audio deepfake model...")
    # Auto classes support both the current Wav2Vec2 checkpoint and a future
    # WavLMForAudioClassification checkpoint in the same model folder.
    processor = AutoFeatureExtractor.from_pretrained(str(AUDIO_FAKE_PATH))
    model = AutoModelForAudioClassification.from_pretrained(str(AUDIO_FAKE_PATH))

    labels = {str(value).upper() for value in model.config.id2label.values()}
    if not {"REAL", "FAKE"}.issubset(labels):
        raise ValueError(
            "Deepfake model must define REAL and FAKE labels in config.json id2label."
        )
    model.eval()
    model.to(DEVICE)

    _cache["audio_fake_processor"] = processor
    _cache["audio_fake_model"]     = model
    print(f"[ModelManager] {model.config.model_type} deepfake model loaded and cached.")
    return processor, model


def _deepfake_label_indices(model):
    """Read REAL/FAKE class positions from the active model configuration."""
    labels = {str(label).upper(): int(index) for index, label in model.config.id2label.items()}
    return labels["REAL"], labels["FAKE"]


# =============================================================
# AUDIO FEATURE EXTRACTION & EXPLANATION GENERATION
# =============================================================

def _compute_audio_features(audio: np.ndarray) -> dict:
    """Compute interpretable audio signal features for explanations."""
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    silence_thresh = 0.02
    silent_n = int(np.sum(np.abs(audio) < silence_thresh))
    silence_ratio = float(silent_n / len(audio)) if len(audio) > 0 else 1.0

    if rms > 0.15:
        energy_level = "High"
    elif rms > 0.05:
        energy_level = "Medium"
    else:
        energy_level = "Low"

    return {
        "peak_amplitude": round(peak, 4),
        "rms_energy": round(rms, 4),
        "zero_crossing_rate": round(zcr, 4),
        "silence_ratio": round(silence_ratio, 3),
        "has_speech": silence_ratio < 0.7,
        "energy_level": energy_level,
    }


def _generate_explanations(
    hate_probs, hate_conf, hate_label,
    fake_probs, fake_conf, fake_label,
    audio_features,
) -> dict:
    """Generate human-readable explanations for both model predictions."""
    hate_margin = abs(float(hate_probs[1]) - float(hate_probs[0]))
    fake_margin = abs(float(fake_probs[1]) - float(fake_probs[0]))
    energy = audio_features["energy_level"]
    has_speech = audio_features["has_speech"]
    decisive = "decisive" if hate_margin > 0.5 else "moderate" if hate_margin > 0.2 else "narrow"

    # ── Hate Speech Explanation ──────────────────────────────
    h_factors = []
    if not has_speech:
        h_factors.append("No significant speech detected (mostly silence, noise, or music)")
        h_factors.append("Skipped classification to prevent false positives")
        h_summary = "Classified as NO SPEECH. Audio lacks sufficient vocal content for reliable analysis."
    elif hate_label == "UNCERTAIN":
        h_factors.append("Audio contains speech, but model confidence is too low (< 60%)")
        h_factors.append(f"Offensive prob: {hate_probs[1]:.1%} vs Non-offensive: {hate_probs[0]:.1%}")
        h_factors.append("Avoided definitive classification due to ambiguity")
        h_summary = f"Classified as UNCERTAIN ({hate_conf:.1%} confidence). Needs manual review."
    elif hate_label == "OFF":
        if hate_conf > 0.85:
            h_factors.append("Strong offensive vocal patterns detected in tone and prosody")
            h_factors.append(f"High model confidence ({hate_conf:.1%}) — clear hate speech indicators")
        else:
            h_factors.append("Moderate offensive speech patterns detected")
        if energy == "High":
            h_factors.append("Elevated vocal intensity (common in aggressive speech)")
        h_summary = (
            f"Classified as OFFENSIVE ({hate_conf:.1%} confidence). "
            f"Offensive probability {hate_probs[1]:.1%} vs non-offensive {hate_probs[0]:.1%}."
        )
    else:
        h_factors.append("No significant hate speech patterns in vocal features")
        if hate_conf > 0.85:
            h_factors.append("High confidence — speech is clearly non-offensive")
        elif energy == "Low":
            h_factors.append("Calm vocal tone consistent with non-offensive speech")
        h_summary = (
            f"Classified as NOT OFFENSIVE ({hate_conf:.1%} confidence). "
            f"Offensive probability only {hate_probs[1]:.1%}."
        )

    # ── Deepfake Explanation ─────────────────────────────────
    f_factors = []
    if not has_speech:
        f_factors.append("No significant speech detected (mostly silence, noise, or music)")
        f_summary = "Classified as NO SPEECH. Audio lacks sufficient vocal content for deepfake analysis."
    elif fake_label == "UNCERTAIN":
        f_factors.append("Audio contains speech, but model confidence is too low (< 60%)")
        f_factors.append(f"Real prob: {fake_probs[0]:.1%} vs Fake: {fake_probs[1]:.1%}")
        f_summary = f"Classified as UNCERTAIN ({fake_conf:.1%} confidence). Needs manual review."
    elif fake_label == "REAL":
        f_factors.append("Natural speech characteristics detected")
        if fake_conf > 0.85:
            f_factors.append(f"High confidence ({fake_conf:.1%}) — consistent natural voice")
        else:
            f_factors.append(f"Moderate confidence ({fake_conf:.1%}) — mostly natural patterns")
        f_summary = (
            f"Classified as AUTHENTIC ({fake_conf:.1%} confidence). "
            f"Real probability {fake_probs[0]:.1%} vs fake {fake_probs[1]:.1%}."
        )
    else:
        f_factors.append("Potential audio manipulation artifacts detected")
        if fake_conf > 0.85:
            f_factors.append(f"High confidence ({fake_conf:.1%}) — strong synthetic audio indicators")
        else:
            f_factors.append(f"Moderate confidence ({fake_conf:.1%}) — some manipulation indicators")
        f_summary = (
            f"Classified as FAKE ({fake_conf:.1%} confidence). "
            f"Fake probability {fake_probs[1]:.1%} vs real {fake_probs[0]:.1%}."
        )

    return {
        "hate_explanation": {
            "factors": h_factors,
            "summary": h_summary,
            "confidence_level": "High" if hate_conf > 0.85 else "Moderate" if hate_conf >= 0.60 else "Low",
            "margin": round(hate_margin, 4),
        },
        "fake_explanation": {
            "factors": f_factors,
            "summary": f_summary,
            "confidence_level": "High" if fake_conf > 0.85 else "Moderate" if fake_conf >= 0.60 else "Low",
            "margin": round(fake_margin, 4),
        },
        "audio_features": audio_features,
    }


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


def predict_text_batch(raw_texts: list[str], batch_size: int = 16) -> list[dict]:
    """Run text hate-speech prediction for many inputs efficiently.

    This is used by the YouTube comments endpoint.  It preserves the same
    preprocessing, labels, and response fields as :func:`predict_text`, but
    sends several comments through XLM-RoBERTa in one inference call.
    """
    if not raw_texts:
        return []

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    prepared = [preprocess(text) for text in raw_texts]
    tokenizer, model = _load_text_model()
    results = []
    id2label = {0: "NOT", 1: "OFF"}

    for start in range(0, len(raw_texts), batch_size):
        batch_texts = [item[0] for item in prepared[start:start + batch_size]]
        encoding = tokenizer(
            batch_texts,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE),
            )
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for offset, probs in enumerate(probabilities):
            index = start + offset
            pred = int(np.argmax(probs))
            label = id2label[pred]
            processed_text, script_type = prepared[index]
            results.append({
                "input_text": raw_texts[index],
                "processed_text": processed_text,
                "script_type": script_type,
                "prediction": label,
                "is_offensive": label == "OFF",
                "confidence": float(probs[pred]),
                "probabilities": {
                    "NOT": float(probs[0]),
                    "OFF": float(probs[1]),
                },
            })

    return results


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
    real_idx, fake_idx = _deepfake_label_indices(f_model)

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

    output["audio_authentic"]     = bool(pred == real_idx)
    output["authenticity_label"]  = "REAL" if pred == real_idx else "FAKE"
    output["authenticity_confidence"] = float(probs[pred])
    output["authenticity_probabilities"] = {
        "REAL": float(probs[real_idx]),
        "FAKE": float(probs[fake_idx]),
    }

    return output


def predict_audio_array(
    audio: "np.ndarray",
    deepfake_audio: "np.ndarray | None" = None,
) -> dict:
    """
    Run both audio models on a raw numpy array (16kHz, float32).
    Used for segmented YouTube analysis — no temp file needed.
    """
    output = {}
    audio_features = _compute_audio_features(audio)
    has_speech = audio_features["has_speech"]

    # ── Hate Speech ──────────────────────────────────────────
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
        logits = h_model(input_values=inp["input_values"].to(DEVICE)).logits
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = int(np.argmax(probs))

    conf = float(probs[pred])
    if not has_speech:
        output["hate_speech"] = False
        output["hate_label"] = "NO_SPEECH"
    elif conf < 0.60:
        output["hate_speech"] = False
        output["hate_label"] = "UNCERTAIN"
    else:
        output["hate_speech"] = bool(pred == 1)
        output["hate_label"] = "OFF" if pred == 1 else "NOT"

    output["hate_confidence"]    = conf
    output["hate_probabilities"] = {"NOT": float(probs[0]), "OFF": float(probs[1])}

    # ── Deepfake ─────────────────────────────────────────────
    # For YouTube, preserve the original waveform for deepfake detection.
    # Hate-speech inference and speech-presence checks continue to use audio.
    fake_waveform = deepfake_audio if deepfake_audio is not None else audio
    f_proc, f_model = _load_audio_fake_model()
    real_idx, fake_idx = _deepfake_label_indices(f_model)
    inp = f_proc(
        fake_waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=MAX_LENGTH,
        truncation=True,
    )
    with torch.no_grad():
        logits = f_model(input_values=inp["input_values"].to(DEVICE)).logits
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = int(np.argmax(probs))

    conf = float(probs[pred])
    if not has_speech:
        output["audio_authentic"] = True
        output["authenticity_label"] = "NO_SPEECH"
    else:
        # Deepfake output is binary for every segment that contains speech.
        # The confidence/probability values remain available for review.
        output["audio_authentic"] = bool(pred == real_idx)
        output["authenticity_label"] = "REAL" if pred == real_idx else "FAKE"

    output["authenticity_confidence"]     = conf
    output["authenticity_probabilities"]  = {
        "REAL": float(probs[real_idx]),
        "FAKE": float(probs[fake_idx]),
    }

    # ── Explanation ──────────────────────────────────────────
    output["explanation"] = _generate_explanations(
        hate_probs=[output["hate_probabilities"]["NOT"], output["hate_probabilities"]["OFF"]],
        hate_conf=output["hate_confidence"],
        hate_label=output["hate_label"],
        fake_probs=[output["authenticity_probabilities"]["REAL"], output["authenticity_probabilities"]["FAKE"]],
        fake_conf=output["authenticity_confidence"],
        fake_label=output["authenticity_label"],
        audio_features=audio_features,
    )

    return output
