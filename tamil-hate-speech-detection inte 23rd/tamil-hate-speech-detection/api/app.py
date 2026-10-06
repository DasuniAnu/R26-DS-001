import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import whisper
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.util
import joblib   # load SVM (.pkl files)
import noisereduce as nr
import soundfile as sf # read/write WAV files
import re # used in clean_text() to remove URLs, mentions, hashtags
import os #  creates temporary files to save uploaded audio temporarily
import tempfile
import json
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)  # allow browser extension content scripts to call the API

@app.after_request
def add_private_network_header(response):
    # Required for browser extensions to call localhost from YouTube pages
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Access-Control-Request-Private-Network'
    return response

# ── Paths ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)  # hate_detector.py / tamil_slur_lexicon.py live at project root

# ── Load .env (no third-party dependency — API keys must never be
# committed to git, so they live in a local, gitignored .env file instead) ──
def _load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_dotenv(os.path.join(BASE_DIR, ".env"))
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text_models", "xlmr_saved")
AUDIO_SCALER    = os.path.join(BASE_DIR, "models", "audio_models", "audio_scaler.pkl")
AUDIO_SVM       = os.path.join(BASE_DIR, "models", "audio_models", "audio_svm.pkl")
FUSION_AUDIO_MODEL = os.path.join(BASE_DIR, "models", "audio_models", "audio_model_combined_tuned.pkl")
AUDIO_DNN       = os.path.join(BASE_DIR, "models", "audio_models", "audio_dnn.pt")
FUSION_MODEL    = os.path.join(BASE_DIR, "models", "fusion_models", "fusion_final.pt")
FUSION_SCALER   = os.path.join(BASE_DIR, "models", "fusion_models", "fusion_scaler_final.pkl")

# Fallback ASR only — used when Sarvam AI fails. Stored on D: (not under the
# project folder) since it's a large download kept off the low-space C: drive.

LOGS_DIR          = os.path.join(BASE_DIR, "logs")
ANALYSIS_LOG_PATH = os.path.join(LOGS_DIR, "analysis_log.jsonl")

# A video is reported "HATE SPEECH DETECTED" once at least this fraction of
# its chunks are individually flagged HATE by youtube_detector.check_chunk().
YOUTUBE_HATE_CHUNK_RATIO = 0.15

def _log_chunk_analysis(url, chunk_index, result):
    os.makedirs(LOGS_DIR, exist_ok=True)
    entry = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "video_url":        url,
        "chunk_index":      chunk_index,
        "verdict":          result["verdict"],
        "n_flagged":        result.get("n_flagged", 0),
        "worst_confidence": result.get("worst_confidence", 0.0),
        "worst_sentence":   result.get("worst_sentence", ""),
    }
    with open(ANALYSIS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# YouTube-only fine-tuned text model + lexicon detector — used ONLY by /analyze_youtube
# and /debug_sentence. Every other route (/predict_fusion, /analyze_document) keeps
# using the original TEXT_MODEL_PATH above, unchanged.
YOUTUBE_TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text_models", "vigilai_v4_model (1)")

# Text-only detector — used by /predict and /predict_batch.
TEXT_ONLY_MODEL_PATH = os.path.join(BASE_DIR, "models", "text_models", "vigilai_v6_model")

# ── Sarvam AI API key (loaded from .env — see _load_dotenv() above) ────
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

# ── ElevenLabs Speech-to-Text API key (2nd choice, only if Sarvam fails) ──
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

# ── Google Cloud Speech-to-Text API key (3rd choice, only if Sarvam and ElevenLabs fail) ──
GOOGLE_SPEECH_API_KEY = os.environ.get("GOOGLE_SPEECH_API_KEY", "")

# ── Model definitions ──────────────────────────────────
class AudioDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(104, 256),   # 0
            nn.BatchNorm1d(256),   # 1
            nn.ReLU(),             # 2
            nn.Dropout(0.3),       # 3
            nn.Linear(256, 128),   # 4
            nn.BatchNorm1d(128),   # 5
            nn.ReLU(),             # 6
            nn.Dropout(0.3),       # 7
            nn.Linear(128, 64),    # 8
            nn.ReLU(),             # 9
            nn.Dropout(0.3),       # 10
            nn.Linear(64, 2),      # 11
        )
    def forward(self, x):
        return self.net(x)

class FusionDNN(nn.Module):
    # Input: 768 (XLM-R CLS) + 104 (audio features) = 872
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(872, 256),   # 0
            nn.BatchNorm1d(256),   # 1
            nn.ReLU(),             # 2
            nn.Dropout(0.3),       # 3
            nn.Linear(256, 128),   # 4
            nn.ReLU(),             # 5
            nn.Dropout(0.3),       # 6
            nn.Linear(128, 2),     # 7
        )
    def forward(self, x):
        return self.net(x)

# ── Load models at startup ─────────────────────────────
print("Loading models...")

tokenizer  = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
text_model = AutoModelForSequenceClassification.from_pretrained(
    TEXT_MODEL_PATH, output_hidden_states=True
)
text_model.eval()

audio_scaler = joblib.load(AUDIO_SCALER)
audio_svm    = joblib.load(AUDIO_SVM)

# Used ONLY by /predict_fusion_audio (the 30% audio side of fusion). A full
# sklearn Pipeline (StandardScaler + SVC) — scaling is bundled in, so no
# separate scaler is applied before calling .predict_proba() on it.
fusion_audio_model = joblib.load(FUSION_AUDIO_MODEL)

audio_dnn = AudioDNN()
audio_dnn.load_state_dict(torch.load(AUDIO_DNN, map_location="cpu", weights_only=True))
audio_dnn.eval()

fusion_model = FusionDNN()
fusion_model.load_state_dict(torch.load(FUSION_MODEL, map_location="cpu", weights_only=True))
fusion_model.eval()

fusion_scaler = joblib.load(FUSION_SCALER)

from hate_detector import HateDetector
from tamil_slur_lexicon import classify as lexicon_classify
print(f"Loading YouTube-only HateDetector ({os.path.basename(YOUTUBE_TEXT_MODEL_PATH)} + slur lexicon)...")
youtube_detector = HateDetector(YOUTUBE_TEXT_MODEL_PATH)

print(f"Loading text-only HateDetector ({os.path.basename(TEXT_ONLY_MODEL_PATH)} + slur lexicon)...")
text_detector = HateDetector(TEXT_ONLY_MODEL_PATH)

print("Loading Whisper tiny (Sarvam AI fallback only)...")
whisper_model = whisper.load_model("tiny")

print("All models loaded!")

# ── Text cleaning ──────────────────────────────────────
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s\u0B80-\u0BFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Text prediction ────────────────────────────────────
# Uses text_detector (vigilai_v6_model + slur lexicon). /analyze_youtube keeps
# using youtube_detector (vigilai_v4_model) separately — see YOUTUBE_TEXT_MODEL_PATH.
# /predict_fusion keeps using the ORIGINAL xlmr_saved model via get_text_embedding()
# below, since FusionDNN + its scaler were trained on that model's embedding space
# and would be miscalibrated by any other one.
def predict_text(text, use_lexicon=False):
    cleaned = clean_text(text)
    result  = text_detector.check(cleaned, use_lexicon=use_lexicon)
    verdict = result["verdict"]  # HATE / UNCERTAIN / SAFE
    if verdict == "UNCERTAIN":   # Text tab is binary: HATE (>=0.75) vs SAFE (everything else)
        verdict = "SAFE"
    is_hate = verdict == "HATE"
    conf    = result["confidence"]  # hate-probability style value, 0-1

    return {
        "label":      "hate" if is_hate else "not_hate",
        "label_code": 1 if is_hate else 0,
        "confidence": round(conf * 100, 2) if is_hate else round((1 - conf) * 100, 2),
        "verdict":    verdict,
        "source":     result.get("source"),
        "text":       text,
        "clean_text": cleaned,
        "modality":   "text"
    }

# ── Text embedding (CLS token, 768-dim) ───────────────
def get_text_embedding(text):
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, padding="max_length", max_length=128
    )
    with torch.no_grad():
        outputs = text_model(**inputs)
        # CLS token from last hidden state → shape (768,)
        cls_embedding = outputs.hidden_states[-1][:, 0, :]
    return cls_embedding.squeeze(0).numpy()  # (768,)

# ── ASR: Sarvam AI (primary) → ElevenLabs (2nd) → Google STT (3rd) ────
# No Whisper fallback: its transcripts were too garbled to be usable — see
# the end of this function for what happens if all three ASR tiers fail.
def transcribe_audio(audio_array, sr):
    if SARVAM_API_KEY and SARVAM_API_KEY != "YOUR_SARVAM_KEY_HERE":
        try:
            import requests as req
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            trimmed = audio_array[:sr * 30]  # max 30s for Sarvam
            sf.write(tmp.name, trimmed, sr)
            tmp.close()
            # Single attempt, no retry: on a bad Sarvam stretch (observed:
            # failing most chunks) a retry-once just adds ~20-80s of dead
            # weight per chunk before falling back anyway. Straight fallback
            # to Google STT keeps YouTube analysis speed predictable.
            with open(tmp.name, 'rb') as f:
                response = req.post(
                    "https://api.sarvam.ai/speech-to-text",
                    headers={"api-subscription-key": SARVAM_API_KEY},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    # Sarvam deprecated saarika:v2.5 on 2026-08-15 — Saarika
                    # and Saaras are now unified under saaras:v3, selected
                    # via `mode` (transcribe here, since this was previously
                    # the transcription-only Saarika model, not translation).
                    # This was the real cause of the earlier flaky/slow/reset
                    # connections: the old model's infrastructure was already
                    # being wound down. Confirmed directly: saaras:v3 replies
                    # in ~18s vs. the old model's 26s-to-outright-failure.
                    data={"language_code": "ta-IN", "model": "saaras:v3", "mode": "transcribe"},
                    # Kept at 60s (up from the original 15s) as headroom —
                    # still useful even though saaras:v3 itself responds
                    # faster, since real network conditions can vary.
                    timeout=60
                )
            os.unlink(tmp.name)
            print(f"[Sarvam AI] status={response.status_code}", flush=True)
            if response.status_code == 200:
                text = response.json().get("transcript", "")
                if text.strip():
                    print(f"[Sarvam AI] {text[:50]}", flush=True)
                    return text.strip(), "Sarvam AI"
                else:
                    print(f"[Sarvam AI] empty transcript, response={response.json()}", flush=True)
            else:
                print(f"[Sarvam AI] error body={response.text}", flush=True)
        except Exception as e:
            print(f"[Sarvam AI failed] {type(e).__name__}: {e}", flush=True)

    if ELEVENLABS_API_KEY and ELEVENLABS_API_KEY != "YOUR_ELEVENLABS_KEY_HERE":
        try:
            import requests as req
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            trimmed = audio_array[:sr * 30]  # keep in sync with the 30s Sarvam limit
            sf.write(tmp.name, trimmed, sr)
            tmp.close()
            with open(tmp.name, 'rb') as f:
                response = req.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers={"xi-api-key": ELEVENLABS_API_KEY},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={"model_id": "scribe_v2", "language_code": "ta"},
                    timeout=30
                )
            os.unlink(tmp.name)
            print(f"[ElevenLabs] status={response.status_code}", flush=True)
            if response.status_code == 200:
                text = response.json().get("text", "")
                if text.strip():
                    print(f"[ElevenLabs] {text[:50]}", flush=True)
                    return text.strip(), "ElevenLabs"
                else:
                    print(f"[ElevenLabs] empty transcript, response={response.json()}", flush=True)
            else:
                print(f"[ElevenLabs] error body={response.text}", flush=True)
        except Exception as e:
            print(f"[ElevenLabs failed] {type(e).__name__}: {e}", flush=True)

    if GOOGLE_SPEECH_API_KEY:
        try:
            import requests as req
            import base64
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            trimmed = audio_array[:sr * 30]  # keep in sync with the 30s Sarvam limit
            sf.write(tmp.name, trimmed, sr)
            tmp.close()
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            os.unlink(tmp.name)
            response = req.post(
                f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_SPEECH_API_KEY}",
                json={
                    "config": {
                        "languageCode": "ta-IN",
                        "model": "latest_long",   # default model was slower AND less accurate
                        "useEnhanced": True,
                    },
                    "audio": {"content": base64.b64encode(audio_bytes).decode("ascii")},
                },
                timeout=15
            )
            print(f"[Google STT] status={response.status_code}", flush=True)
            if response.status_code == 200:
                results = response.json().get("results", [])
                text = " ".join(
                    r["alternatives"][0].get("transcript", "")
                    for r in results if r.get("alternatives")
                )
                if text.strip():
                    print(f"[Google STT] {text[:50]}", flush=True)
                    return text.strip(), "Google STT"
                else:
                    print(f"[Google STT] empty transcript, response={response.json()}", flush=True)
            else:
                print(f"[Google STT] error body={response.text}", flush=True)
        except Exception as e:
            print(f"[Google STT failed] {type(e).__name__}: {e}", flush=True)

    # No Whisper fallback: its transcripts were too garbled to be usable.
    # If Sarvam, ElevenLabs, and Google STT all fail/return nothing, treat
    # the chunk as having no usable speech rather than guessing with a
    # low-quality ASR — downstream (check_chunk / no-content handling)
    # already treats an empty transcript as SKIPPED / not-hate, which is
    # the desired behavior.
    print("[No transcript] Sarvam, ElevenLabs, and Google STT all unavailable", flush=True)
    return "", "No Speech Detected"

# ── Audio feature extraction (104 features) ───────────
def _safe(val, default=0.0):
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default

def extract_audio_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=16000, mono=True, duration=60)
    except Exception as e:
        print(f"[audio load ERROR] {e}", flush=True)
        return None

    if len(y) < 1000:
        return None

    f = []

    # MFCC — 80 features
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        for j in range(40):
            f.extend([_safe(np.mean(mfcc[j])), _safe(np.std(mfcc[j]))])
    except Exception:
        f.extend([0.0] * 80)

    # Pitch — 3 features
    try:
        pv = librosa.piptrack(y=y, sr=sr)[0]
        pv = pv[pv > 0]
        f.extend([
            _safe(np.mean(pv)) if len(pv) > 0 else 0.0,
            _safe(np.std(pv))  if len(pv) > 0 else 0.0,
            _safe(np.max(pv))  if len(pv) > 0 else 0.0
        ])
    except Exception:
        f.extend([0.0, 0.0, 0.0])

    # RMS — 3 features
    try:
        rms = librosa.feature.rms(y=y)
        f.extend([_safe(np.mean(rms)), _safe(np.std(rms)), _safe(np.max(rms))])
    except Exception:
        f.extend([0.0, 0.0, 0.0])

    # Spectral centroid — 2 features
    try:
        c = librosa.feature.spectral_centroid(y=y, sr=sr)
        f.extend([_safe(np.mean(c)), _safe(np.std(c))])
    except Exception:
        f.extend([0.0, 0.0])

    # Spectral bandwidth — 2 features
    try:
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        f.extend([_safe(np.mean(bw)), _safe(np.std(bw))])
    except Exception:
        f.extend([0.0, 0.0])

    # Spectral rolloff — 2 features
    try:
        ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
        f.extend([_safe(np.mean(ro)), _safe(np.std(ro))])
    except Exception:
        f.extend([0.0, 0.0])

    # ZCR — 3 features
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        f.extend([_safe(np.mean(zcr)), _safe(np.std(zcr)), _safe(np.max(zcr))])
    except Exception:
        f.extend([0.0, 0.0, 0.0])

    # Chroma — 2 features
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        f.extend([_safe(np.mean(chroma)), _safe(np.std(chroma))])
    except Exception:
        f.extend([0.0, 0.0])

    # Mel spectrogram — 2 features
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        f.extend([_safe(np.mean(mel)), _safe(np.std(mel))])
    except Exception:
        f.extend([0.0, 0.0])

    # Tonnetz — 2 features
    try:
        harmonic = librosa.effects.harmonic(y)
        tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
        f.extend([_safe(np.mean(tonnetz)), _safe(np.std(tonnetz))])
    except Exception:
        f.extend([0.0, 0.0])

    # Tempo — 1 feature
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        f.append(_safe(np.asarray(tempo).flat[0]))
    except Exception:
        f.append(0.0)

    # Spectral contrast — 2 features
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        f.extend([_safe(np.mean(contrast)), _safe(np.std(contrast))])
    except Exception:
        f.extend([0.0, 0.0])

    return np.array(f, dtype=np.float32).reshape(1, -1)


def extract_audio_features_fusion_v2(y, sr):
    """Feature order MUST match the training notebook for audio_model_combined_tuned.pkl
    exactly (audio-4.ipynb) — different from extract_audio_features() above. Used ONLY
    by /predict_fusion_audio, paired 1:1 with fusion_audio_model."""
    f = []

    # 1. MFCC — 80 (block layout: all 40 means, then all 40 stds)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    f.extend([_safe(v) for v in np.mean(mfcc, axis=1)])
    f.extend([_safe(v) for v in np.std(mfcc, axis=1)])

    # 2. Pitch — 3
    pv = librosa.piptrack(y=y, sr=sr)[0]
    pv = pv[pv > 0]
    f.extend([
        _safe(np.mean(pv)) if len(pv) > 0 else 0.0,
        _safe(np.std(pv))  if len(pv) > 0 else 0.0,
        _safe(np.max(pv))  if len(pv) > 0 else 0.0
    ])

    # 3. RMS — 3
    rms = librosa.feature.rms(y=y)
    f.extend([_safe(np.mean(rms)), _safe(np.std(rms)), _safe(np.max(rms))])

    # 4. ZCR — 3
    zcr = librosa.feature.zero_crossing_rate(y)
    f.extend([_safe(np.mean(zcr)), _safe(np.std(zcr)), _safe(np.max(zcr))])

    # 5. Spectral centroid / bandwidth / rolloff — 6
    for fn in (librosa.feature.spectral_centroid, librosa.feature.spectral_bandwidth, librosa.feature.spectral_rolloff):
        v = fn(y=y, sr=sr)
        f.extend([_safe(np.mean(v)), _safe(np.std(v))])

    # 6. Chroma — 2
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f.extend([_safe(np.mean(chroma)), _safe(np.std(chroma))])

    # 7. Mel — 2
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    f.extend([_safe(np.mean(mel)), _safe(np.std(mel))])

    # 8. Tempo — 1
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        f.append(_safe(np.asarray(tempo).flat[0]))
    except Exception:
        f.append(0.0)

    # 9. Spectral contrast — 2
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    f.extend([_safe(np.mean(contrast)), _safe(np.std(contrast))])

    # 10. Tonnetz — 2
    harmonic = librosa.effects.harmonic(y)
    tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
    f.extend([_safe(np.mean(tonnetz)), _safe(np.std(tonnetz))])

    return np.array(f, dtype=np.float32).reshape(1, -1)

# ── Routes ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message":   "Tamil Hate Speech Detection API",
        "author":    "Anutthara S.A.D (IT22217554)",
        "version":   "1.0",
        "endpoints": {
            "/predict":        "POST — predict text only (XLM-RoBERTa, 83.18%)",
            "/predict_batch":  "POST — predict multiple texts",
            "/predict_audio":  "POST — predict audio only (SVM, 85.86%)",
            "/predict_fusion": "POST — text + audio fusion (DNN, 96.24%)",
            "/health":         "GET  — API status"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "xlm-roberta-base"})

@app.route("/predict", methods=["POST"])
def predict_single():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide text field"}), 400
    use_lexicon = data.get("use_lexicon", False)
    result = predict_text(data["text"], use_lexicon=use_lexicon)
    return jsonify(result)

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Please provide texts field (list)"}), 400
    use_lexicon = data.get("use_lexicon", False)
    results    = [predict_text(t, use_lexicon=use_lexicon) for t in data["texts"]]
    hate_count = sum(1 for r in results if r["label"] == "hate")
    return jsonify({
        "total":      len(results),
        "hate_count": hate_count,
        "safe_count": len(results) - hate_count,
        "hate_pct":   round(hate_count / len(results) * 100, 1),
        "results":    results
    })

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Please upload audio file"}), 400
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    try:
        features = extract_audio_features(tmp_path)
    finally:
        os.unlink(tmp_path)
    if features is None:
        return jsonify({"error": "Could not process audio"}), 400
    features_scaled = audio_scaler.transform(features)
    pred = audio_svm.predict(features_scaled)[0]
    prob = audio_svm.predict_proba(features_scaled)[0]
    conf = prob[pred]
    return jsonify({
        "label":      "hate" if pred == 1 else "not_hate",
        "label_code": int(pred),
        "confidence": round(conf * 100, 2),
        "modality":   "audio"
    })

@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    """Late fusion: XLM-R CLS embedding (768) + audio features (104) → FusionDNN → 96.24%"""
    if "audio" not in request.files:
        return jsonify({"error": "Please upload audio file"}), 400
    text = request.form.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please provide text field in form data"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    try:
        audio_feats = extract_audio_features(tmp_path)
    finally:
        os.unlink(tmp_path)

    if audio_feats is None:
        return jsonify({"error": "Could not process audio"}), 400

    text_embedding = get_text_embedding(text)   # (768,)
    audio_flat     = audio_feats.flatten()       # (104,)
    combined       = np.concatenate([text_embedding, audio_flat]).reshape(1, -1)  # (1, 872)

    combined_scaled = fusion_scaler.transform(combined)
    tensor_input    = torch.tensor(combined_scaled, dtype=torch.float32)

    with torch.no_grad():
        fusion_model.eval()
        logits = fusion_model(tensor_input)
        probs  = torch.softmax(logits, dim=1)
        pred   = torch.argmax(probs, dim=1).item()
        conf   = probs[0][pred].item()

    return jsonify({
        "label":      "hate" if pred == 1 else "not_hate",
        "label_code": int(pred),
        "confidence": round(conf * 100, 2),
        "text":       text,
        "modality":   "fusion (text + audio)",
        "accuracy":   "96.24%"
    })

@app.route("/predict_fusion_audio", methods=["POST"])
def predict_fusion_audio():
    """WAV → denoise → normalize → Sarvam/Whisper ASR → vigilai_v4_model + fusion_audio_model
    (audio_model_combined_tuned.pkl) → 70/30 weighted fusion, HATE at >=0.75."""
    if "audio" not in request.files:
        return jsonify({"error": "Please upload audio file"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_path = tmp.name
        audio_file.save(raw_path)

    try:
        # ── Step 1: Load raw audio ─────────────────────
        y_raw, sr = librosa.load(raw_path, sr=16000, duration=60)
    finally:
        os.unlink(raw_path)

    if len(y_raw) < 1000:
        return jsonify({"error": "Audio too short to process"}), 400

    # ── Step 2: Denoise + normalize ────────────────────
    y_clean = nr.reduce_noise(y=y_raw, sr=sr, stationary=False, prop_decrease=0.8)
    y_clean = librosa.util.normalize(y_clean)

    # ── Step 3: ASR (Sarvam AI → Google STT → Whisper fallback) ────
    transcription, asr_model_used = transcribe_audio(y_clean, sr)
    if not transcription:
        transcription = "[no speech detected]"

    # ── Step 4: Text probability (vigilai_v4_model, via youtube_detector) ──
    cleaned        = clean_text(transcription)
    text_hate_prob = youtube_detector._model_prob(cleaned)

    # ── Step 5: Librosa feature extraction + fusion_audio_model ──
    # Feature order matches audio_model_combined_tuned.pkl's training notebook,
    # NOT the same order as extract_audio_features()/audio_svm used elsewhere.
    features    = extract_audio_features_fusion_v2(y_clean, sr)
    audio_proba = fusion_audio_model.predict_proba(features)[0]
    audio_hate_prob = float(audio_proba[1])

    # ── Step 6: Weighted fusion 70% text (v4) + 30% audio (SVM) ──
    fusion_hate_prob = 0.70 * text_hate_prob + 0.30 * audio_hate_prob
    label      = "hate" if fusion_hate_prob >= 0.75 else "not_hate"
    confidence = max(fusion_hate_prob, 1.0 - fusion_hate_prob) * 100

    return jsonify({
        "label":            label,
        "confidence":       round(confidence, 2),
        "transcription":    transcription,
        "asr_model":        asr_model_used,
        "text_hate_prob":   round(text_hate_prob * 100, 2),
        "audio_hate_prob":  round(audio_hate_prob * 100, 2),
        "fusion_hate_prob": round(fusion_hate_prob * 100, 2),
        "fusion_weights":   "70% text (v4) + 30% audio (SVM)",
        "fusion_threshold": 0.75,
        "modality":         "fusion (denoise → ASR → text + audio)"
    })

@app.route("/analyze_youtube", methods=["POST"])
def analyze_youtube():
    """Download YouTube audio → split 30s chunks → Sarvam AI/Whisper ASR → XLM-RoBERTa per chunk."""
    try:
        data = request.get_json(silent=True)
        if not data or "youtube_url" not in data:
            return jsonify({"error": "Please provide youtube_url"}), 400
        url = data["youtube_url"].strip()
        use_lexicon = data.get("use_lexicon", True)
        try:
            import yt_dlp
        except ImportError:
            return jsonify({"error": "yt-dlp not installed. Run: pip install yt-dlp"}), 500

        return _run_youtube_analysis(url, use_lexicon=use_lexicon)

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


def _run_youtube_analysis(url, use_lexicon=True):
    import yt_dlp
    # ── Step 1: Download audio ─────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "audio")
        FFMPEG_PATH = r"C:\Users\Admin\Downloads\ffmpeg-2026-07-02-git-95a888b9ca-essentials_build\ffmpeg-2026-07-02-git-95a888b9ca-essentials_build\bin"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_path + '.%(ext)s',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'ffmpeg_location': FFMPEG_PATH,
            'quiet': True, 'no_warnings': True,
            # yt-dlp's default client (android_vr) gets SABR-blocked (403) on many
            # videos; android/mweb still serve direct, downloadable audio formats.
            'extractor_args': {'youtube': {'player_client': ['android', 'mweb']}},
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            return jsonify({"error": f"Download failed: {str(e)}"}), 400

        audio_file = None
        for ext in ['wav', 'webm', 'mp4', 'm4a', 'opus', 'mp3']:
            candidate = out_path + '.' + ext
            if os.path.exists(candidate):
                audio_file = candidate
                break
        if not audio_file:
            files = os.listdir(tmpdir)
            audio_file = os.path.join(tmpdir, files[0]) if files else None
        if not audio_file:
            return jsonify({"error": "No audio file downloaded"}), 400

        try:
            y, sr = librosa.load(audio_file, sr=16000, mono=True)
        except Exception as e:
            return jsonify({"error": f"Could not load audio: {str(e)}"}), 400

    # ── Step 2: Split into 30s chunks → ASR → HateDetector (lexicon + model) ─
    chunk_samples = sr * 30
    chunks = []

    for start_sample in range(0, len(y), chunk_samples):
        end_sample  = min(start_sample + chunk_samples, len(y))
        chunk       = y[start_sample:end_sample]
        start_sec   = int(start_sample / sr)
        end_sec     = int(end_sample / sr)

        if len(chunk) < sr:   # skip chunks shorter than 1 second
            continue

        # Denoise chunk before ASR
        try:
            chunk_clean = nr.reduce_noise(y=chunk, sr=sr, stationary=False, prop_decrease=0.8)
            chunk_clean = librosa.util.normalize(chunk_clean)
        except Exception:
            chunk_clean = chunk

        # Sarvam AI → Google STT → Whisper fallback → Tamil text
        transcription, asr_used = transcribe_audio(chunk_clean, sr)
        print(f"[YouTube] {start_sec}s–{end_sec}s | ASR:{asr_used} | text: {transcription[:40]}", flush=True)

        # Silent SVM pass (audio_model_combined_tuned.pkl) — logged only, not
        # used in the response/verdict and never shown in the UI. Wrapped so a
        # failure here can never affect the real analysis.
        try:
            svm_features = extract_audio_features_fusion_v2(chunk_clean, sr)
            svm_hate_prob = float(fusion_audio_model.predict_proba(svm_features)[0][1])
            print(f"[YouTube][SVM audio] {start_sec}s–{end_sec}s | hate_prob: {round(svm_hate_prob, 3)}", flush=True)
        except Exception as e:
            print(f"[YouTube][SVM audio failed] {type(e).__name__}: {e}", flush=True)

        if not transcription or transcription == "[no speech detected]":
            transcription = ""

        # check_chunk(): HATE only if a Tier-1 slur fires, OR 1 sentence is
        # flagged very strongly (>=0.93) — otherwise SAFE.
        result = youtube_detector.check_chunk(transcription, use_lexicon=use_lexicon)

        chunk_entry = {
            "chunk":            len(chunks) + 1,
            "start_time":       f"{start_sec//60:02d}:{start_sec%60:02d}",
            "end_time":         f"{end_sec//60:02d}:{end_sec%60:02d}",
            "start_sec":        start_sec,
            "end_sec":          end_sec,
            "asr_model":        asr_used,
            "transcript":       transcription,
            "verdict":          result["verdict"],
            "n_sentences":      result["n_sentences"],
            "n_flagged":        result.get("n_flagged", 0),
            "flagged":          result.get("flagged", []),
            "worst_sentence":   result.get("worst_sentence", ""),
            "worst_confidence": result.get("worst_confidence", 0.0),
        }
        chunks.append(chunk_entry)
        _log_chunk_analysis(url, chunk_entry["chunk"], result)

    total            = len(chunks)
    hate_chunks      = sum(1 for c in chunks if c["verdict"] == "HATE")
    uncertain_chunks = sum(1 for c in chunks if c["verdict"] == "UNCERTAIN")
    ratio            = round(hate_chunks / total, 3) if total > 0 else 0.0

    def _instance(c):
        return {
            "timestamp":       f"{c['start_time']} - {c['end_time']}",
            "sentence":        c["worst_sentence"],
            "full_transcript": c["transcript"],
            "confidence":      c["worst_confidence"],
            "asr_model":       c["asr_model"],
        }

    hate_instances      = [_instance(c) for c in chunks if c["verdict"] == "HATE"]
    uncertain_instances = [_instance(c) for c in chunks if c["verdict"] == "UNCERTAIN"]

    verdict = "HATE SPEECH DETECTED" if ratio >= YOUTUBE_HATE_CHUNK_RATIO else "VIDEO IS SAFE"

    return jsonify({
        "url":                 url,
        "total_chunks":        total,
        "hate_chunks":         hate_chunks,
        "uncertain_chunks":    uncertain_chunks,
        "safe_chunks":         total - hate_chunks - uncertain_chunks,
        "ratio":               ratio,
        "verdict":             verdict,
        "hate_instances":      hate_instances,
        "uncertain_instances": uncertain_instances,
        "chunks":              chunks
    })


def fetch_youtube_video_info(url):
    """Fetch a YouTube video's title/thumbnail/duration/uploader/description via yt-dlp, no download."""
    import yt_dlp
    ydl_opts = {'skip_download': True, 'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title":       info.get("title"),
        "thumbnail":   info.get("thumbnail"),
        "duration":    info.get("duration"),
        "uploader":    info.get("uploader"),
        "description": info.get("description"),
    }


@app.route("/youtube_video_info", methods=["POST"])
def youtube_video_info():
    """Lightweight metadata-only lookup — used to show a preview card before/while analysis runs."""
    try:
        data = request.get_json(silent=True)
        if not data or "youtube_url" not in data:
            return jsonify({"error": "Please provide youtube_url"}), 400
        url = data["youtube_url"].strip()

        try:
            import yt_dlp
        except ImportError:
            return jsonify({"error": "yt-dlp not installed. Run: pip install yt-dlp"}), 500

        info = fetch_youtube_video_info(url)
        return jsonify(info)
    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": f"Could not fetch video info: {str(e)}"}), 400


def fetch_youtube_comments(url, max_comments=100):
    """Fetch top-level YouTube comments via yt-dlp (no API key needed)."""
    import yt_dlp
    ydl_opts = {
        'skip_download': True,
        'getcomments': True,
        'extractor_args': {'youtube': {'max_comments': [str(max_comments)]}},
        'quiet': True, 'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    raw_comments = info.get('comments') or []
    return [c['text'] for c in raw_comments if c.get('text', '').strip()]


# ── Comment-only rules (used ONLY by /analyze_youtube_comments) ────────
# Not used by /predict, /analyze_youtube, or any other route.
def is_other_language(text):
    """True if the comment has no Tamil script at all (skips English/other-language comments)."""
    return not re.search(r'[஀-௿]', str(text))

def has_no_content(text):
    stripped = re.sub(r'[^\w஀-௿]', '', str(text))
    return len(stripped) < 2

def check_comment(text, use_lexicon=True):
    """youtube_detector (v4) + lexicon. Non-Tamil and empty/no-content comments
    are skipped before this; every remaining comment is scored by the model,
    regardless of length."""
    raw = str(text).strip()

    if has_no_content(raw):
        return {"verdict": "SKIPPED", "reason": "no_content", "confidence": 0.0, "source": "skip", "text": text}

    if is_other_language(raw):
        return {"verdict": "SKIPPED", "reason": "non_tamil", "confidence": 0.0, "source": "skip", "text": text}

    cleaned = clean_text(raw)

    if use_lexicon:
        lex, reason = lexicon_classify(cleaned)
        if lex == 1:
            return {"verdict": "HATE", "confidence": 100.0, "source": "lexicon", "reason": reason, "text": text}

    p = youtube_detector._model_prob(cleaned)

    verdict = "HATE" if p >= 0.75 else "SAFE"
    return {"verdict": verdict, "confidence": round(p * 100, 2), "source": "model", "reason": "model_score", "text": text}


@app.route("/analyze_youtube_comments", methods=["POST"])
def analyze_youtube_comments():
    """Fetch a YouTube video's comments and classify each one with check_comment()
    (youtube_detector / vigilai_v4_model + slur lexicon + comment-only rules above).
    Completely separate from /predict's text_detector (v6)."""
    try:
        data = request.get_json(silent=True)
        if not data or "youtube_url" not in data:
            return jsonify({"error": "Please provide youtube_url"}), 400
        url          = data["youtube_url"].strip()
        max_comments = int(data.get("max_comments", 100))
        use_lexicon  = data.get("use_lexicon", True)

        try:
            import yt_dlp
        except ImportError:
            return jsonify({"error": "yt-dlp not installed. Run: pip install yt-dlp"}), 500

        try:
            comments = fetch_youtube_comments(url, max_comments=max_comments)
        except Exception as e:
            return jsonify({"error": f"Could not fetch comments: {str(e)}"}), 400

        if not comments:
            return jsonify({
                "url": url, "total": 0, "checked_total": 0, "hate_count": 0, "uncertain_count": 0,
                "safe_count": 0, "skipped_count": 0, "hate_pct": 0.0, "hate_comments": [], "results": []
            })

        results = [check_comment(c, use_lexicon=use_lexicon) for c in comments]

        hate_count      = sum(1 for r in results if r["verdict"] == "HATE")
        uncertain_count = sum(1 for r in results if r["verdict"] == "UNCERTAIN")
        safe_count      = sum(1 for r in results if r["verdict"] == "SAFE")
        skipped_count   = sum(1 for r in results if r["verdict"] == "SKIPPED")
        checked_total   = len(results) - skipped_count

        return jsonify({
            "url":             url,
            "total":           len(results),
            "checked_total":   checked_total,
            "hate_count":      hate_count,
            "uncertain_count": uncertain_count,
            "safe_count":      safe_count,
            "skipped_count":   skipped_count,
            "hate_pct":        round(hate_count / checked_total * 100, 1) if checked_total else 0.0,
            "hate_comments":   [r for r in results if r["verdict"] == "HATE"],
            "results":         results,
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/debug_sentence", methods=["POST"])
def debug_sentence():
    """Diagnose a single sentence through the YouTube-only HateDetector
    (lexicon + vigilai_xlmr_clean) without re-running a whole video."""
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please provide text field"}), 400
    use_lexicon = data.get("use_lexicon", True)
    return jsonify(youtube_detector.check(data["text"], use_lexicon=use_lexicon))


@app.route("/predict_mic", methods=["POST"])
def predict_mic():
    if "audio" not in request.files:
        return jsonify({"error": "Please upload audio"}), 400
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    try:
        features = extract_audio_features(tmp_path)
    finally:
        os.unlink(tmp_path)
    if features is None:
        return jsonify({"error": "Could not process audio"}), 400
    features_scaled = audio_scaler.transform(features)
    pred = audio_svm.predict(features_scaled)[0]
    prob = audio_svm.predict_proba(features_scaled)[0]
    return jsonify({
        "label":      "hate" if pred == 1 else "not_hate",
        "confidence": round(float(prob[pred]) * 100, 2),
        "modality":   "microphone"
    })


@app.route("/analyze_document", methods=["POST"])
def analyze_document():
    if "file" not in request.files:
        return jsonify({"error": "Please upload a file"}), 400
    file = request.files["file"]
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        try:
            text = file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            return jsonify({"error": f"Could not read file: {e}"}), 400
    elif filename.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return jsonify({"error": "PyPDF2 not installed. Run: pip install PyPDF2"}), 500
        except Exception as e:
            return jsonify({"error": f"PDF read error: {e}"}), 400
    else:
        return jsonify({"error": "Only .txt and .pdf files supported"}), 400

    # Split by double newline first; if that gives only 1 block, also split by single newline
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 15]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 15]
    if not paragraphs:
        return jsonify({"error": "No readable text found in document"}), 400

    results = []
    for para in paragraphs[:50]:
        r = predict_text(para)
        results.append({"paragraph": para[:600], "label": r["label"], "confidence": r["confidence"]})

    hate_count = sum(1 for r in results if r["label"] == "hate")
    return jsonify({
        "total_paragraphs": len(results),
        "hate_paragraphs":  hate_count,
        "safe_paragraphs":  len(results) - hate_count,
        "results":          results
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)
