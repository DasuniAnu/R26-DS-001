from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.util
import joblib
import whisper
import noisereduce as nr
import soundfile as sf
import re
import os
import tempfile

app = Flask(__name__)
CORS(app)  # allow Chrome extension content scripts to call the API

@app.after_request
def add_private_network_header(response):
    # Required for Chrome extensions to call localhost from YouTube pages
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Access-Control-Request-Private-Network'
    return response

# ── Paths ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text_models", "xlmr_saved")
AUDIO_SCALER    = os.path.join(BASE_DIR, "models", "audio_models", "audio_scaler.pkl")
AUDIO_SVM       = os.path.join(BASE_DIR, "models", "audio_models", "audio_svm.pkl")
AUDIO_DNN       = os.path.join(BASE_DIR, "models", "audio_models", "audio_dnn.pt")
FUSION_MODEL    = os.path.join(BASE_DIR, "models", "fusion_models", "fusion_final.pt")
FUSION_SCALER   = os.path.join(BASE_DIR, "models", "fusion_models", "fusion_scaler_final.pkl")

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

audio_dnn = AudioDNN()
audio_dnn.load_state_dict(torch.load(AUDIO_DNN, map_location="cpu", weights_only=True))
audio_dnn.eval()

fusion_model = FusionDNN()
fusion_model.load_state_dict(torch.load(FUSION_MODEL, map_location="cpu", weights_only=True))
fusion_model.eval()

fusion_scaler = joblib.load(FUSION_SCALER)

print("Loading Whisper medium...")
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
def predict_text(text):
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, padding="max_length", max_length=128
    )
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
        pred    = torch.argmax(probs, dim=1).item()
        conf    = probs[0][pred].item()
    return {
        "label":      "hate" if pred == 1 else "not_hate",
        "label_code": pred,
        "confidence": round(conf * 100, 2),
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
    result = predict_text(data["text"])
    return jsonify(result)

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Please provide texts field (list)"}), 400
    results    = [predict_text(t) for t in data["texts"]]
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
    """WAV → denoise → normalize → Whisper ASR → XLM-R + SVM → 30/70 weighted fusion."""
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

    # Save clean audio to temp file (for traceability; Whisper reads numpy directly)
    clean_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    clean_path = clean_tmp.name
    clean_tmp.close()
    sf.write(clean_path, y_clean, sr)

    # ── Step 3: Whisper ASR (medium) on clean audio ────
    asr_result    = whisper_model.transcribe(y_clean.astype(np.float32),
                                             language="ta", task="transcribe")
    transcription = asr_result["text"].strip()
    os.unlink(clean_path)
    if not transcription:
        transcription = "[no speech detected]"

    # ── Step 4: XLM-R text probability ────────────────
    cleaned = clean_text(transcription)
    inputs  = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, padding="max_length", max_length=128
    )
    with torch.no_grad():
        outputs    = text_model(**inputs)
        text_probs = torch.softmax(outputs.logits, dim=1)[0]
    text_hate_prob = text_probs[1].item()

    # ── Step 5: Librosa feature extraction + SVM ───────
    y = y_clean   # features on denoised audio
    f = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    for j in range(40):
        f.extend([np.mean(mfcc[j]), np.std(mfcc[j])])
    pv = librosa.piptrack(y=y, sr=sr)[0]
    pv = pv[pv > 0]
    f.extend([np.mean(pv) if len(pv) > 0 else 0,
              np.std(pv)  if len(pv) > 0 else 0,
              np.max(pv)  if len(pv) > 0 else 0])
    rms = librosa.feature.rms(y=y)
    f.extend([np.mean(rms), np.std(rms), np.max(rms)])
    c = librosa.feature.spectral_centroid(y=y, sr=sr)
    f.extend([np.mean(c), np.std(c)])
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    f.extend([np.mean(bw), np.std(bw)])
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    f.extend([np.mean(ro), np.std(ro)])
    zcr = librosa.feature.zero_crossing_rate(y)
    f.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f.extend([np.mean(chroma), np.std(chroma)])
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    f.extend([np.mean(mel), np.std(mel)])
    harmonic = librosa.effects.harmonic(y)
    tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
    f.extend([np.mean(tonnetz), np.std(tonnetz)])
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    f.append(_safe(np.asarray(tempo).flat[0]))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    f.extend([np.mean(contrast), np.std(contrast)])

    features        = np.array(f, dtype=np.float32).reshape(1, -1)
    features_scaled = audio_scaler.transform(features)
    audio_proba     = audio_svm.predict_proba(features_scaled)[0]
    audio_hate_prob = float(audio_proba[1])

    # ── Step 6: Weighted fusion 30% text + 70% audio ──
    fusion_hate_prob = 0.30 * text_hate_prob + 0.70 * audio_hate_prob
    label      = "hate" if fusion_hate_prob >= 0.5 else "not_hate"
    confidence = max(fusion_hate_prob, 1.0 - fusion_hate_prob) * 100

    return jsonify({
        "label":            label,
        "confidence":       round(confidence, 2),
        "transcription":    transcription,
        "text_hate_prob":   round(text_hate_prob * 100, 2),
        "audio_hate_prob":  round(audio_hate_prob * 100, 2),
        "fusion_hate_prob": round(fusion_hate_prob * 100, 2),
        "fusion_weights":   "30% text + 70% audio",
        "modality":         "fusion (denoise → ASR → text + audio)"
    })

@app.route("/predict_fusion_video", methods=["POST"])
def predict_fusion_video():
    """Chrome extension endpoint: WAV audio blob → denoise → Whisper tiny → XLM-R + SVM → 30/70 fusion."""
    if "audio" not in request.files:
        return jsonify({"error": "Please upload audio file"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_path = tmp.name
        audio_file.save(raw_path)

    try:
        y_raw, sr = librosa.load(raw_path, sr=16000, duration=60)
    finally:
        os.unlink(raw_path)

    if len(y_raw) < 1000:
        return jsonify({"error": "Audio too short to process"}), 400

    # Step 1: Denoise + normalize
    y_clean = nr.reduce_noise(y=y_raw, sr=sr, stationary=False, prop_decrease=0.8)
    y_clean = librosa.util.normalize(y_clean)

    # Step 2: Whisper tiny transcription
    asr         = whisper_model.transcribe(y_clean.astype(np.float32), language="ta", task="transcribe")
    transcription = asr["text"].strip() or "[no speech detected]"

    # Step 3: XLM-RoBERTa text probability
    cleaned_text = clean_text(transcription)
    inputs = tokenizer(cleaned_text, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        text_probs = torch.softmax(text_model(**inputs).logits, dim=1)[0]
    text_hate_prob = text_probs[1].item()
    text_pred = "hate" if text_hate_prob >= 0.5 else "not_hate"
    text_conf = text_hate_prob if text_pred == "hate" else (1.0 - text_hate_prob)

    # Step 4: Librosa 104 features + SVM audio probability
    y = y_clean
    f = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    for j in range(40):
        f.extend([np.mean(mfcc[j]), np.std(mfcc[j])])
    pv = librosa.piptrack(y=y, sr=sr)[0]
    pv = pv[pv > 0]
    f.extend([np.mean(pv) if len(pv) > 0 else 0,
              np.std(pv)  if len(pv) > 0 else 0,
              np.max(pv)  if len(pv) > 0 else 0])
    rms = librosa.feature.rms(y=y)
    f.extend([np.mean(rms), np.std(rms), np.max(rms)])
    c = librosa.feature.spectral_centroid(y=y, sr=sr)
    f.extend([np.mean(c), np.std(c)])
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    f.extend([np.mean(bw), np.std(bw)])
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    f.extend([np.mean(ro), np.std(ro)])
    zcr = librosa.feature.zero_crossing_rate(y)
    f.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f.extend([np.mean(chroma), np.std(chroma)])
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    f.extend([np.mean(mel), np.std(mel)])
    harmonic = librosa.effects.harmonic(y)
    tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
    f.extend([np.mean(tonnetz), np.std(tonnetz)])
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    f.append(_safe(np.asarray(tempo).flat[0]))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    f.extend([np.mean(contrast), np.std(contrast)])

    features        = np.array(f, dtype=np.float32).reshape(1, -1)
    features_scaled = audio_scaler.transform(features)
    audio_proba     = audio_svm.predict_proba(features_scaled)[0]
    audio_hate_prob = float(audio_proba[1])
    audio_pred = "hate" if audio_hate_prob >= 0.5 else "not_hate"
    audio_conf = audio_hate_prob if audio_pred == "hate" else (1.0 - audio_hate_prob)

    # Step 5: Weighted fusion 30% text + 70% audio
    fusion_hate_prob = 0.30 * text_hate_prob + 0.70 * audio_hate_prob
    fusion_pred = "hate" if fusion_hate_prob >= 0.5 else "not_hate"
    fusion_conf = max(fusion_hate_prob, 1.0 - fusion_hate_prob)

    return jsonify({
        "transcription":     transcription,
        "text_prediction":   text_pred,
        "text_confidence":   round(text_conf, 4),
        "audio_prediction":  audio_pred,
        "audio_confidence":  round(audio_conf, 4),
        "fusion_prediction": fusion_pred,
        "fusion_confidence": round(fusion_conf, 4),
        "text_weight":       0.30,
        "audio_weight":      0.70
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)
