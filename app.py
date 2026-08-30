# =========================================================
# backend/app.py
# DeepShield - Condition C v3 Video Deepfake Detection API
# =========================================================

from flask import Flask, request, jsonify
import json
import os
import sys
import tempfile
import traceback
import time

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
CONFIG_DIR = os.path.join(PROJECT_DIR, "config")

sys.path.insert(0, SCRIPTS_DIR)

from temporal_scoring import sliding_window_temporal_scorer

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "condition_c_v3_hardneg_best.keras"
)

CONFIG_PATH = os.path.join(
    CONFIG_DIR,
    "condition_c_v3_deployment_config.json"
)

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm",
    "m4v"
}

# Maximum uploaded file size: 500 MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# =========================================================
# HELPERS
# =========================================================

def allowed_file(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def load_config_for_health():
    if not os.path.exists(CONFIG_PATH):
        return None

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def error_response(message, status_code=400, details=None):
    response = {
        "verdict": "ERROR",
        "fake_percentage": 0.0,
        "fake_segments": [],
        "error": message
    }

    if details is not None:
        response["details"] = details

    return jsonify(response), status_code


def clean_result_for_api(result):
    """
    Remove very large internal per-window arrays from the normal frontend
    response while preserving the useful v3 output.
    """
    if not isinstance(result, dict):
        return {
            "verdict": "ERROR",
            "fake_percentage": 0.0,
            "fake_segments": [],
            "error": "Temporal scorer returned an invalid response."
        }

    cleaned_segments = []

    for segment in result.get("fake_segments", []):
        cleaned_segments.append({
            "start_time": float(segment.get("start_time", 0) or 0),
            "end_time": float(segment.get("end_time", 0) or 0),
            "duration": float(segment.get("duration", 0) or 0),
            "confidence": float(segment.get("confidence", 0) or 0),
            "peak_confidence": float(
                segment.get(
                    "peak_confidence",
                    segment.get("confidence", 0)
                ) or 0
            )
        })

    cleaned = {
        "verdict": result.get("verdict", "UNKNOWN"),
        "fake_percentage": float(result.get("fake_percentage", 0) or 0),
        "fake_segments": cleaned_segments,
        "segments_found": int(
            result.get("segments_found", len(cleaned_segments)) or 0
        ),
        "video_duration": float(result.get("video_duration", 0) or 0),
        "candidate_windows": int(result.get("candidate_windows", 0) or 0),
        "windows_scored": int(result.get("windows_scored", 0) or 0),
        "rejected_windows": int(result.get("rejected_windows", 0) or 0),
        "analysable_percentage": float(
            result.get("analysable_percentage", 0) or 0
        ),
        "raw_fake_windows": int(result.get("raw_fake_windows", 0) or 0),
        "fake_windows": int(result.get("fake_windows", 0) or 0),
        "mean_fake_probability": float(
            result.get("mean_fake_probability", 0) or 0
        ),
        "max_fake_probability": float(
            result.get("max_fake_probability", 0) or 0
        ),
        "window_threshold": result.get("window_threshold"),
        "minimum_consecutive_fake_windows": result.get(
            "minimum_consecutive_fake_windows"
        ),
        "full_fake_coverage_threshold": result.get(
            "full_fake_coverage_threshold"
        ),
        "model": result.get(
            "model",
            os.path.basename(MODEL_PATH)
        )
    }

    if "error" in result:
        cleaned["error"] = result["error"]

    return cleaned


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    config = load_config_for_health()

    ready = (
        os.path.exists(MODEL_PATH)
        and
        os.path.exists(CONFIG_PATH)
    )

    return jsonify({
        "status": "ok" if ready else "not_ready",
        "service": "DeepShield",
        "version": "Condition C v3",
        "model": os.path.basename(MODEL_PATH),
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "config_path": CONFIG_PATH,
        "config_exists": os.path.exists(CONFIG_PATH),
        "configuration": config
    }), (200 if ready else 503)


# =========================================================
# ANALYZE VIDEO
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():
    temp_path = None
    request_start = time.time()

    try:
        # -----------------------------------------------------
        # MODEL / CONFIG CHECK
        # -----------------------------------------------------

        if not os.path.exists(MODEL_PATH):
            return error_response(
                "Condition C v3 model file was not found.",
                500,
                {"model_path": MODEL_PATH}
            )

        if not os.path.exists(CONFIG_PATH):
            return error_response(
                "Condition C v3 deployment config was not found.",
                500,
                {"config_path": CONFIG_PATH}
            )

        # -----------------------------------------------------
        # FILE CHECK
        # -----------------------------------------------------

        if "video" not in request.files:
            return error_response(
                "No video file was provided.",
                400
            )

        uploaded_file = request.files["video"]

        if not uploaded_file.filename:
            return error_response(
                "The uploaded video has no filename.",
                400
            )

        if not allowed_file(uploaded_file.filename):
            return error_response(
                "Unsupported video format.",
                400,
                {
                    "allowed_formats":
                        sorted(list(ALLOWED_EXTENSIONS))
                }
            )

        # -----------------------------------------------------
        # SAVE TEMPORARY VIDEO
        # -----------------------------------------------------

        extension = os.path.splitext(
            uploaded_file.filename
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            suffix=extension
        ) as temporary_file:

            uploaded_file.save(temporary_file.name)
            temp_path = temporary_file.name

        print()
        print("=" * 70)
        print("DEEPSHIELD CONDITION C v3 ANALYSIS REQUEST")
        print("=" * 70)
        print("Filename:", uploaded_file.filename)
        print("Temporary file:", temp_path)
        print("Model:", MODEL_PATH)
        print("Config:", CONFIG_PATH)

        # -----------------------------------------------------
        # RUN FINAL v3 TEMPORAL SCORER
        # -----------------------------------------------------

        result = sliding_window_temporal_scorer(
            temp_path,
            MODEL_PATH,
            CONFIG_PATH
        )

        processing_time = time.time() - request_start

        api_result = clean_result_for_api(result)
        api_result["api_processing_time"] = round(processing_time, 2)
        api_result["source_filename"] = uploaded_file.filename

        print()
        print("-" * 70)
        print("Verdict:", api_result.get("verdict"))
        print(
            "Fake percentage:",
            api_result.get("fake_percentage"),
            "%"
        )
        print(
            "Analysable percentage:",
            api_result.get("analysable_percentage"),
            "%"
        )
        print(
            "Fake sections:",
            len(api_result.get("fake_segments", []))
        )
        print(
            "Processing time:",
            round(processing_time, 2),
            "seconds"
        )
        print("=" * 70)

        if api_result.get("verdict") == "ERROR":
            return jsonify(api_result), 500

        return jsonify(api_result)

    except Exception as exception:
        print()
        print("=" * 70)
        print("DEEPSHIELD API ERROR")
        print("=" * 70)
        traceback.print_exc()

        return error_response(
            str(exception),
            500
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                print(
                    "Temporary file cleanup warning:",
                    cleanup_error
                )


# =========================================================
# FILE TOO LARGE HANDLER
# =========================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return error_response(
        "The uploaded video is too large. "
        "Maximum allowed size is 500 MB.",
        413
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("DeepShield API — Condition C v3")
    print("=" * 70)
    print("Project:", PROJECT_DIR)
    print("Model:", MODEL_PATH)
    print("Model exists:", os.path.exists(MODEL_PATH))
    print("Config:", CONFIG_PATH)
    print("Config exists:", os.path.exists(CONFIG_PATH))
    print()
    print("Health:")
    print("http://127.0.0.1:5000/health")
    print()
    print("Analyze:")
    print("POST http://127.0.0.1:5000/analyze")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True
    )