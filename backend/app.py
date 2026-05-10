# =============================================================
# FLASK BACKEND — HATE SPEECH DETECTION API
#
# Endpoints:
#   POST /api/text/predict   — Text hate speech detection
#   POST /api/audio/predict  — Audio hate speech + deepfake
#   GET  /api/health         — Health check
# =============================================================

import os
import uuid
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

from model_manager import predict_text, predict_audio

app = Flask(__name__)
CORS(app)

# Temp directory for uploaded audio files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "hate_detect_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".webm"}


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Sinhala Hate Speech Detection API",
        "models": {
            "text_hate_speech": "XLM-RoBERTa (lazy loaded)",
            "audio_hate_speech": "Wav2Vec2 (lazy loaded)",
            "audio_deepfake": "Wav2Vec2 (lazy loaded)",
        }
    })


@app.route("/api/text/predict", methods=["POST"])
def text_predict():
    """
    Text hate speech detection.

    Expects JSON: {"text": "your sinhala text here"}
    Returns prediction with confidence scores.
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    try:
        result = predict_text(text)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/audio/predict", methods=["POST"])
def audio_predict():
    """
    Audio hate speech + deepfake detection.

    Expects: multipart/form-data with 'audio' file field.
    Returns hate speech + authenticity predictions.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use 'audio' field."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Check extension
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported audio format: {ext}. Supported: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        }), 400

    # Save to temp file
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        audio_file.save(temp_path)
        result = predict_audio(temp_path)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    print("=" * 60)
    print("  Sinhala Hate Speech Detection API")
    print("  Models: Lazy loaded on first request")
    print("  Device: CPU")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
