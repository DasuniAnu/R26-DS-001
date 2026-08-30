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
import platform
import glob

# Dynamically add WinGet FFmpeg path to environment variables if on Windows
if platform.system() == "Windows":
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        winget_packages = os.path.join(user_profile, "AppData", "Local", "Microsoft", "WinGet", "Packages")
        if os.path.exists(winget_packages):
            search_pattern = os.path.join(winget_packages, "*FFmpeg*", "**", "bin")
            bin_dirs = glob.glob(search_pattern, recursive=True)
            for bin_dir in bin_dirs:
                if os.path.isdir(bin_dir):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                    print(f"[FFmpeg Setup] Dynamically added FFmpeg to PATH: {bin_dir}")
                    break
from flask import Flask, request, jsonify
from flask_cors import CORS

from model_manager import predict_text, predict_text_batch, predict_audio, predict_audio_array
from youtube_utils import analyze_youtube_video, fetch_youtube_comments, get_video_metadata
from asr_manager import transcribe_sinhala_audio, MODEL_ID as SINHALA_ASR_MODEL

app = Flask(__name__)
CORS(app)

# Temp directory for uploaded audio files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "hate_detect_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".webm"}


def _fuse_youtube_hate_predictions(audio_prediction: dict, transcript_prediction: dict | None) -> dict:
    """Fuse audio and transcript hate predictions for a YouTube segment only."""
    audio_label = audio_prediction["hate_label"]
    text_label = transcript_prediction["prediction"] if transcript_prediction else "NOT_ASSESSED"

    if audio_label == "NO_SPEECH":
        label = "NOT_ASSESSED"
        is_offensive = False
        reason = "No speech was detected, so transcript hate analysis was not assessed."
    elif audio_label == "OFF" or text_label == "OFF":
        label = "OFF"
        is_offensive = True
        sources = []
        if audio_label == "OFF":
            sources.append("audio model")
        if text_label == "OFF":
            sources.append("Sinhala ASR transcript")
        reason = "Offensive content flagged by " + " and ".join(sources) + "."
    elif audio_label == "UNCERTAIN" and transcript_prediction is None:
        label = "UNCERTAIN"
        is_offensive = False
        reason = "Audio prediction is uncertain and no usable transcript was produced."
    else:
        label = "NOT"
        is_offensive = False
        reason = "Neither the audio model nor the transcript model flagged offensive content."

    return {
        "hate_speech": is_offensive,
        "hate_label": label,
        "audio_hate_label": audio_label,
        "transcript_hate_label": text_label,
        "fusion_reason": reason,
    }


def _is_usable_sinhala_transcript(text: str) -> bool:
    """Reject decoder hallucinations before they reach the text hate model."""
    meaningful = [char for char in text if not char.isspace() and char.isalnum()]
    if not meaningful:
        return False
    sinhala_chars = sum("\u0D80" <= char <= "\u0DFF" for char in meaningful)
    return sinhala_chars >= 3 and sinhala_chars / len(meaningful) >= 0.25


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
            "youtube_sinhala_asr": SINHALA_ASR_MODEL,
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



@app.route("/api/youtube/analyze", methods=["POST"])
def youtube_analyze():
    """
    YouTube video hate speech + deepfake analysis.

    Expects JSON: {"url": "https://youtube.com/..."}
    Returns per-segment predictions with timestamps + summary.
    """
    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field"}), 400

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400

    try:
        metadata, segments = analyze_youtube_video(url)

        results = []
        for seg in segments:
            audio = seg.pop("audio")   # remove numpy array before JSON
            asr_audio = seg.pop("asr_audio", audio)
            # Keep the established VAD-masked hate path; use original audio
            # only for the YouTube deepfake model.
            pred = predict_audio_array(audio, deepfake_audio=asr_audio)
            transcript = ""
            transcript_prediction = None
            asr_status = "NOT_ASSESSED"

            # VAD/no-speech decisions from the existing audio pipeline prevent
            # unnecessary ASR work and make the result explicitly not assessed.
            if pred["hate_label"] != "NO_SPEECH":
                candidate_transcript = transcribe_sinhala_audio(asr_audio)
                if _is_usable_sinhala_transcript(candidate_transcript):
                    transcript = candidate_transcript
                    transcript_prediction = predict_text(transcript)
                    asr_status = "TRANSCRIBED"
                else:
                    asr_status = "LOW_QUALITY_OUTPUT"

            fusion = _fuse_youtube_hate_predictions(pred, transcript_prediction)
            results.append({
                **seg,
                **pred,
                **fusion,
                "transcript": transcript,
                "transcript_analysis": transcript_prediction,
                "asr_status": asr_status,
            })
            del audio
            del asr_audio
            
        import gc
        gc.collect()

        total      = len(results)
        hate_segs  = [r for r in results if r["hate_speech"]]
        transcript_hate_segs = [r for r in results if r["transcript_hate_label"] == "OFF"]
        fake_segs  = [r for r in results if not r["audio_authentic"]]
        both_segs  = [r for r in results if r["hate_speech"] and not r["audio_authentic"]]

        summary = {
            "total_segments":  total,
            "hate_segments":   len(hate_segs),
            "fake_segments":   len(fake_segs),
            "both_segments":   len(both_segs),
            "clean_segments":  total - len(set(
                [r["start_time"] for r in hate_segs + fake_segs]
            )),
            "hate_percentage": round(len(hate_segs) / total * 100, 1) if total else 0,
            "fake_percentage": round(len(fake_segs) / total * 100, 1) if total else 0,
            "transcript_hate_segments": len(transcript_hate_segs),
        }

        return jsonify({
            "success":  True,
            "metadata": metadata,
            "segments": results,
            "summary":  summary,
        })

    except ValueError as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e) if str(e) else type(e).__name__
        return jsonify({"error": f"Analysis failed: {error_msg}"}), 500


@app.route("/api/youtube/metadata", methods=["POST"])
def youtube_metadata():
    """Fetch lightweight video metadata before starting full analysis."""
    data = request.get_json(silent=True)
    if not data or not data.get("url", "").strip():
        return jsonify({"error": "URL cannot be empty"}), 400

    try:
        return jsonify({"success": True, "metadata": get_video_metadata(data["url"].strip())})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Metadata fetch failed: {str(e)}"}), 500


@app.route("/api/youtube/comments", methods=["POST"])
def youtube_comments():
    """
    Fetch and analyze all public Unicode-Sinhala YouTube comments for hate speech.
    Expects JSON: {"url": "https://youtube.com/..."}
    """
    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field"}), 400

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400

    max_comments = data.get("max_comments")
    if max_comments is not None:
        try:
            max_comments = max(1, int(max_comments))
        except (TypeError, ValueError):
            return jsonify({"error": "'max_comments' must be a whole number"}), 400

    try:
        comments, source = fetch_youtube_comments(url, max_comments)
        # Only Unicode Sinhala comments are in scope. This automatically
        # excludes English, romanized Sinhala, and emoji-only comments.
        comments = [
            comment for comment in comments
            if any("\u0D80" <= char <= "\u0DFF" for char in comment["text"])
        ]
        
        predictions = predict_text_batch([comment["text"] for comment in comments])
        results = []
        for c, pred in zip(comments, predictions):
            results.append({
                "author": c["author"],
                "text": c["text"],
                "likes": c["likes"],
                "is_offensive": pred["is_offensive"],
                "confidence": pred["confidence"]
            })

        hate_comments = [r for r in results if r["is_offensive"]]
        
        summary = {
            "total_fetched": len(results),
            "hate_comments": len(hate_comments),
            "hate_percentage": round(len(hate_comments) / len(results) * 100, 1) if results else 0,
        }

        return jsonify({
            "success": True,
            "comments": results,
            "summary": summary,
            "source": source,
        })
    except ValueError as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Comment analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  Sinhala Hate Speech Detection API")
    print("  Models: Lazy loaded on first request")
    print("  Device: CPU")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
