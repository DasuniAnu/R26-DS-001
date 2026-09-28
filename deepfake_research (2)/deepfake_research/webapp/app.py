# =========================================================
# backend/app.py
# DeepShield - Condition C v3 Video Deepfake Detection API
# =========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys
import tempfile
import traceback
import time
import subprocess
from pathlib import Path

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

# Same real Deno location used by app.py's YouTube-URL path (fixed there
# after the original C:\Users\LENOVO\... path was found to be from a
# different machine). Kept as its own constant here rather than importing
# from app.py, since app.py starts a Streamlit server as a side effect of
# being imported — this backend must not trigger that.
DENO_PATH = r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe\deno.exe"

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# Registered BEFORE CORS(app) below on purpose: Flask runs after_request
# hooks in reverse registration order, so this one (registered first) runs
# LAST — after flask-cors's own hook, which by default sets this same
# header to "false". Without that ordering, both values end up present on
# the response (confirmed while testing this), which is exactly the kind
# of ambiguous duplicate header a browser's Private Network Access check
# should not be handed.
@app.after_request
def add_private_network_header(response):
    # Same Chrome Private Network Access requirement handled for the other
    # backends (see gateway/app.py) — needed for a browser extension to call
    # this from a chrome-extension:// origin.
    if "Access-Control-Allow-Private-Network" in response.headers:
        del response.headers["Access-Control-Allow-Private-Network"]
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


CORS(app)  # allow the unified web app / browser extension to call this API


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
# YOUTUBE URL ANALYSIS (Chrome extension)
# =========================================================
# Additive only: the direct-upload /analyze route above is completely
# unchanged. This exists so the Chrome extension can hand over a YouTube
# URL (same as it already does for Tamil/Sinhala) instead of a file, since
# a browser extension can't read/upload a page's video file directly.
# Mirrors app.py's own YouTube-download logic (yt-dlp + Deno) rather than
# importing it, since importing app.py would start a second Streamlit
# server as a side effect — this keeps the two completely independent.

def download_youtube_video_for_api(url, output_directory):
    if not os.path.exists(DENO_PATH):
        raise RuntimeError(f"Deno was not found at: {DENO_PATH}")

    os.makedirs(output_directory, exist_ok=True)
    output_template = os.path.join(output_directory, "youtube_video.%(ext)s")

    command = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--restrict-filenames",
        "--js-runtimes", f"deno:{DENO_PATH}",
        "--remote-components", "ejs:npm",
        "-f", "bv*[height<=720]+ba/b[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        url
    ]

    try:
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=300
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("YouTube download timed out after 5 minutes.")

    if process.returncode != 0:
        details = process.stderr or process.stdout or "Unknown yt-dlp error."
        raise RuntimeError("Could not download the YouTube video.\n\n" + details[-2500:])

    downloaded_files = [
        p for p in Path(output_directory).glob("youtube_video.*")
        if p.is_file() and p.suffix.lower() not in {".part", ".ytdl"}
    ]
    if not downloaded_files:
        raise RuntimeError("YouTube download completed, but the final video file could not be located.")

    mp4_files = [p for p in downloaded_files if p.suffix.lower() == ".mp4"]
    downloaded_file = max(mp4_files or downloaded_files, key=lambda p: p.stat().st_mtime)
    return str(downloaded_file)


@app.route("/analyze_youtube", methods=["POST"])
def analyze_youtube():
    temp_dir = None
    request_start = time.time()

    try:
        if not os.path.exists(MODEL_PATH):
            return error_response("Condition C v3 model file was not found.", 500, {"model_path": MODEL_PATH})
        if not os.path.exists(CONFIG_PATH):
            return error_response("Condition C v3 deployment config was not found.", 500, {"config_path": CONFIG_PATH})

        payload = request.get_json(silent=True) or {}
        url = (payload.get("url") or payload.get("youtube_url") or "").strip()
        if not url:
            return error_response("No YouTube URL was provided.", 400)

        temp_dir = tempfile.mkdtemp(prefix="deepshield_yt_")

        print()
        print("=" * 70)
        print("DEEPSHIELD YOUTUBE ANALYSIS REQUEST")
        print("=" * 70)
        print("URL:", url)

        try:
            video_path = download_youtube_video_for_api(url, temp_dir)
        except RuntimeError as download_error:
            return error_response(str(download_error), 400)

        print("Downloaded video:", video_path)

        result = sliding_window_temporal_scorer(video_path, MODEL_PATH, CONFIG_PATH)

        processing_time = time.time() - request_start
        api_result = clean_result_for_api(result)
        api_result["api_processing_time"] = round(processing_time, 2)
        api_result["source_url"] = url

        print("Verdict:", api_result.get("verdict"))
        print("Processing time:", round(processing_time, 2), "seconds")
        print("=" * 70)

        if api_result.get("verdict") == "ERROR":
            return jsonify(api_result), 500

        return jsonify(api_result)

    except Exception as exception:
        print()
        print("=" * 70)
        print("DEEPSHIELD YOUTUBE ANALYSIS ERROR")
        print("=" * 70)
        traceback.print_exc()
        return error_response(str(exception), 500)

    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                print("Temporary directory cleanup warning:", cleanup_error)


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
    # Defaults to the original port 5001, unchanged, so running this file
    # the normal standalone way (`python app.py`) is unaffected. Only the
    # integration launcher sets DEEPFAKE_BACKEND_PORT, since 5001 is
    # already used by the Sinhala backend in the combined setup.
    _port = int(os.environ.get("DEEPFAKE_BACKEND_PORT", 5001))

    print()
    print("Health:")
    print(f"http://127.0.0.1:{_port}/health")
    print()
    print("Analyze:")
    print(f"POST http://127.0.0.1:{_port}/analyze")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=_port,
        debug=False,
        threaded=True
    )