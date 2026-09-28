"""Flask interface for Sinhala YouTube false-content detection."""

from __future__ import annotations

import os
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from services.youtube_client import (
    YouTubeMetadata,
    audio_duration_seconds,
    audio_transcript_status,
    download_youtube_audio,
    extract_video_id,
    fetch_youtube_metadata,
    generate_whisper_transcript,
    merge_metadata,
    search_youtube_videos,
)
from utils.consistency import (
    CONSISTENT_LABEL,
    INCONSISTENT_LABEL,
    compute_consistency_features,
    consistency_similarity,
    normalize_consistency_label,
    normalize_for_consistency,
    normalize_mismatch_type,
)
from utils.data_utils import clean_text, is_meaningful_text, normalize_binary_label, normalize_label


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "instance" / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_DATASET_PATH = BASE_DIR / "data" / "real_sinhala_youtube_false_content.csv"
YOUTUBE_QUEUE_PATH = BASE_DIR / "data" / "youtube_candidate_queue.csv"
SYNTHETIC_LABELING_POOL_PATH = BASE_DIR / "data" / "synthetic_similarity_labeling_pool.csv"
ASR_MANIFEST_PATH = BASE_DIR / "data" / "sinhala_asr_manifest.csv"
REAL_DATASET_COLUMNS = [
    "video_id",
    "url",
    "title",
    "description",
    "transcript",
    "topic",
    "original_label",
    "binary_label",
    "label_source",
    "annotator_notes",
    "group_id",
    "claim_text",
    "evidence_url",
    "evidence_summary",
    "label_confidence",
    "queue_source",
    "caption_source",
    "consistency_label",
    "mismatch_type",
    "support_rationale",
    "generated_negative",
    "source_title_video_id",
    "source_transcript_video_id",
    "title_transcript_similarity",
    "description_transcript_similarity",
    "title_description_similarity",
]
YOUTUBE_QUEUE_COLUMNS = [
    "video_id",
    "url",
    "topic",
    "title",
    "description",
    "transcript",
    "caption_language",
    "channel_title",
    "published_at",
    "thumbnail_url",
    "collection_query",
    "risk_score",
    "claim_score",
    "model_prediction",
    "model_confidence",
    "suggested_group_id",
    "queue_status",
    "collected_at",
]
SYNTHETIC_LABELING_POOL_COLUMNS = [
    "synthetic_id",
    "video_id",
    "url",
    "title",
    "description",
    "transcript",
    "topic",
    "seed_binary_label",
    "seed_consistency_label",
    "seed_mismatch_type",
    "seed_support_rationale",
    "generated_negative",
    "source_title_video_id",
    "source_transcript_video_id",
    "title_transcript_similarity",
    "description_transcript_similarity",
    "title_description_similarity",
    "similarity_band",
    "suggested_group_id",
    "label_status",
    "reviewed_binary_label",
    "reviewed_consistency_label",
    "reviewed_mismatch_type",
    "reviewed_support_rationale",
    "reviewed_at",
]
QUEUE_STATUSES = {"new", "in_review", "saved", "skipped", "duplicate", "not_factual", "needs_evidence"}
ASR_MANIFEST_COLUMNS = ["audio_path", "transcript", "split", "source", "duration_seconds"]
ASR_SPLITS = ["train", "validation", "test"]
ASR_MIN_SINHALA_CHARS = 120
ASR_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
ASR_BRACKET_RE = re.compile(r"\[[^\]]+\]")
SINHALA_CHAR_RE = re.compile(r"[\u0d80-\u0dff]")
NON_SINHALA_SCRIPT_RE = re.compile(r"[\u0900-\u097f\u0980-\u09ff\u0b80-\u0bff\u0400-\u04ff]")
TOPICS = [
    "Agriculture",
    "Crime",
    "Education",
    "Energy",
    "Entertainment",
    "Finance",
    "Government",
    "Health",
    "Jobs",
    "Politics",
    "Religion",
    "Sports",
    "Technology",
    "Transport",
    "Weather",
]
TOPIC_SEARCH_QUERIES = {
    "Agriculture": "සිංහල කෘෂිකර්ම ගොවිතැන පුවත්",
    "Education": "සිංහල අධ්‍යාපන පුවත් පාසල් විශ්වවිද්‍යාල",
    "Entertainment": "සිංහල විනෝදාස්වාද පුවත්",
    "Finance": "සිංහල මුදල් ණය ආයෝජන පුවත්",
    "Government": "සිංහල රජය නිවේදන පුවත්",
    "Health": "සිංහල සෞඛ්‍ය වෛද්‍ය ප්‍රතිකාර පුවත්",
    "Politics": "සිංහල දේශපාලන මැතිවරණ පුවත්",
    "Religion": "සිංහල ආගමික පුවත්",
    "Technology": "සිංහල තාක්ෂණ පුවත්",
    "Weather": "සිංහල කාලගුණ වැසි සුළි කුණාටු පුවත්",
}
TOPIC_RISK_TERMS = {
    "Agriculture": ["fertilizer", "crop disease", "harvest loss", "farmer subsidy"],
    "Education": ["exam", "school closure", "university", "results"],
    "Entertainment": ["viral", "death", "rumour", "actor"],
    "Finance": ["loan", "investment", "crypto", "money", "bank"],
    "Government": ["gazette", "subsidy", "allowance", "official notice"],
    "Health": ["cure", "medicine", "doctor", "disease", "vaccine"],
    "Politics": ["election", "minister", "vote", "parliament"],
    "Religion": ["miracle", "temple", "religious claim", "prophecy"],
    "Technology": ["phone", "AI", "hack", "data leak", "app"],
    "Weather": ["storm", "rain", "flood", "cyclone", "warning"],
}
SEARCH_INTENTS = ["news", "claim", "viral", "official", "fact"]
SOURCE_TERMS = [
    "official",
    "source",
    "ministry",
    "government",
    "police",
    "hospital",
    "doctor",
    "research",
    "report",
    "study",
    "gazette",
    "statement",
    "නිල",
    "මූලාශ්‍ර",
    "වාර්තාව",
]
SENSATIONAL_TERMS = [
    "breaking",
    "viral",
    "shocking",
    "secret",
    "truth",
    "exposed",
    "100%",
    "guaranteed",
    "urgent",
    "danger",
    "leaked",
    "අනාවරණ",
    "රහස",
    "විශේෂ",
]
CLAIM_TERMS = [
    "claim",
    "says",
    "will",
    "can",
    "because",
    "warning",
    "announced",
    "confirmed",
    "rumour",
    "පවසයි",
    "කියයි",
    "නිවේදනය",
    "සත්‍ය",
    "අසත්‍ය",
]
# Clean search/labeling terms used by the collection UI. The older literals above are
# kept for compatibility, but these UTF-8 terms drive new candidate discovery.
TOPIC_SEARCH_QUERIES = {
    "Agriculture": "සිංහල කෘෂිකර්ම ගොවිතැන පුවත්",
    "Education": "සිංහල අධ්‍යාපන පුවත් පාසල් විශ්වවිද්‍යාල",
    "Entertainment": "සිංහල විනෝදාස්වාද පුවත්",
    "Finance": "සිංහල මුදල් ණය ආයෝජන පුවත්",
    "Government": "සිංහල රජය නිවේදන පුවත්",
    "Health": "සිංහල සෞඛ්‍ය වෛද්‍ය ප්‍රතිකාර පුවත්",
    "Politics": "සිංහල දේශපාලන මැතිවරණ පුවත්",
    "Religion": "සිංහල ආගමික පුවත්",
    "Technology": "සිංහල තාක්ෂණ පුවත්",
    "Weather": "සිංහල කාලගුණ වැසි සුළි කුණාටු පුවත්",
}
COLLECTION_MODES = {
    "balanced": "Balanced",
    "false_candidates": "Likely False/Misleading",
    "not_false_candidates": "Likely Not False/Reliable",
}
COLLECTION_MODE_TERMS = {
    "balanced": ["පුවත්", "විස්තර", "තහවුරු", "viral", "official"],
    "false_candidates": [
        "අසත්‍ය",
        "ව්‍යාජ",
        "කටකතා",
        "තහවුරු නොකළ",
        "බොරුවක්",
        "වැරදි තොරතුරු",
        "fake",
        "misleading",
        "fact check",
        "viral claim",
    ],
    "not_false_candidates": [
        "නිල",
        "නිල නිවේදනය",
        "පැහැදිලි කිරීම",
        "සම්පූර්ණ විස්තර",
        "දැනගන්න",
        "official",
        "explained",
        "ministry",
        "government",
        "education",
    ],
}
SOURCE_TERMS.extend(["නිල", "මූලාශ්‍ර", "වාර්තාව", "අමාත්‍යාංශය", "දෙපාර්තමේන්තුව"])
SENSATIONAL_TERMS.extend(["අනාවරණ", "රහස", "විශේෂ", "අදම", "අවසාන තීරණය"])
CLAIM_TERMS.extend(["පවසයි", "කියයි", "නිවේදනය", "සත්‍ය", "අසත්‍ය", "තහවුරු"])
BENCHMARK_TARGETS = {
    "rows": 300,
    "per_topic": 30,
    "false": 100,
    "not_false": 100,
    "groups": 80,
}


def load_env_file(path: Path = BASE_DIR / ".env") -> None:
    """Load KEY=value pairs from .env without overriding existing environment variables."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


def model_artifacts_available() -> dict[str, bool]:
    """Report saved model artifacts without importing the heavy model stack."""

    sklearn_text_artifact = BASE_DIR / "outputs" / "model" / "sklearn_text_classifier.joblib"
    return {
        "text": sklearn_text_artifact.exists() or (BASE_DIR / "outputs" / "model" / "text_config.json").exists(),
        "multimodal": (BASE_DIR / "outputs" / "model" / "multimodal_config.json").exists(),
    }


def default_model_type() -> str:
    available = model_artifacts_available()
    if available["text"]:
        return "text"
    if available["multimodal"]:
        return "multimodal"
    return "text"


def get_predictor(model_type: str):
    """Import predictor code only when a prediction is requested."""

    from services.predictor import get_predictor as load_predictor

    return load_predictor(model_type)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.after_request
    def add_extension_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            result=None,
            error=None,
            form_data={},
            models=model_artifacts_available(),
            default_model=default_model_type(),
        )

    @app.get("/youtube")
    def youtube_page():
        return render_template("youtube.html")

    @app.get("/test")
    def test_page():
        return render_template(
            "test.html",
            models=model_artifacts_available(),
            default_model=default_model_type(),
            youtube_api_key_loaded=bool(os.environ.get("YOUTUBE_API_KEY")),
        )

    @app.get("/label")
    def label_page():
        return render_template(
            "label.html",
            topics=TOPICS,
            topic_search_queries=TOPIC_SEARCH_QUERIES,
            collection_modes=COLLECTION_MODES,
            collection_mode_terms=COLLECTION_MODE_TERMS,
            youtube_api_key_loaded=bool(os.environ.get("YOUTUBE_API_KEY")),
        )

    @app.get("/synthetic-label")
    def synthetic_label_page():
        return render_template("synthetic_label.html", topics=TOPICS)

    @app.get("/asr")
    def asr_page():
        return render_template(
            "asr.html",
            youtube_api_key_loaded=bool(os.environ.get("YOUTUBE_API_KEY")),
            audio_transcript=audio_transcript_status(),
            splits=ASR_SPLITS,
            topics=TOPICS,
            topic_search_queries=TOPIC_SEARCH_QUERIES,
        )

    @app.get("/audio")
    def audio_page():
        return render_template(
            "audio.html",
            default_url=clean_text(request.args.get("url", "")) or "https://www.youtube.com/watch?v=3E7ilNQlCN4",
            audio_transcript=audio_transcript_status(),
        )

    @app.post("/predict")
    def predict_form():
        form_data = {
            "title": clean_text(request.form.get("title", "")),
            "description": clean_text(request.form.get("description", "")),
            "transcript": clean_text(request.form.get("transcript", "")),
            "thumbnail": clean_text(request.form.get("thumbnail", "")),
            "model_type": clean_text(request.form.get("model_type", default_model_type())),
        }

        try:
            validate_text_inputs(form_data)
            thumbnail = saved_upload_path() or form_data["thumbnail"]
            predictor = get_predictor(form_data["model_type"])
            result = predictor.predict(
                title=form_data["title"],
                description=form_data["description"],
                transcript=form_data["transcript"],
                thumbnail=thumbnail,
            )
            return render_template(
                "index.html",
                result=result.as_dict(),
                error=None,
                form_data=form_data,
                models=model_artifacts_available(),
                default_model=form_data["model_type"],
            )
        except Exception as exc:  # noqa: BLE001 - user-facing web error boundary.
            return render_template(
                "index.html",
                result=None,
                error=str(exc),
                form_data=form_data,
                models=model_artifacts_available(),
                default_model=form_data["model_type"],
            ), 400

    @app.post("/api/predict")
    def predict_api():
        payload = request.get_json(silent=True) or {}
        form_data = {
            "title": clean_text(payload.get("title", "")),
            "description": clean_text(payload.get("description", "")),
            "transcript": clean_text(payload.get("transcript", "")),
            "thumbnail": clean_text(payload.get("thumbnail", "")),
            "model_type": clean_text(payload.get("model_type", default_model_type())),
        }

        try:
            validate_text_inputs(form_data)
            predictor = get_predictor(form_data["model_type"])
            result = predictor.predict(
                title=form_data["title"],
                description=form_data["description"],
                transcript=form_data["transcript"],
                thumbnail=form_data["thumbnail"],
            )
            return jsonify(result.as_dict())
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/predict-youtube", methods=["POST", "OPTIONS"])
    def predict_youtube_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        video_id = extract_video_id(clean_text(payload.get("video_id") or payload.get("url") or ""))
        page_data = payload.get("page_data") or {}
        require_youtube_api = bool(payload.get("require_youtube_api"))
        use_audio_fallback = bool(payload.get("use_audio_fallback", not require_youtube_api))

        try:
            if require_youtube_api and not os.environ.get("YOUTUBE_API_KEY"):
                raise ValueError("YOUTUBE_API_KEY is required to load data through YouTube Data API v3.")
            api_metadata = (
                fetch_youtube_metadata(video_id, use_audio_fallback=use_audio_fallback)
                if video_id
                else YouTubeMetadata(video_id="", source="manual")
            )
            if require_youtube_api and api_metadata.source != "youtube_api":
                raise ValueError("Could not load this video through YouTube Data API v3.")
            metadata = merge_metadata(
                api_metadata,
                page_data,
                use_transcript=True,
                warn_missing_transcript=False,
            )
            if not any(is_meaningful_text(field) for field in (metadata.title, metadata.description, metadata.transcript)):
                raise ValueError(
                    "No YouTube title or description was available. Add YOUTUBE_API_KEY to .env, "
                    "restart Flask, or paste transcript text manually and try again."
                )
            validate_text_inputs(
                {
                    "title": metadata.title,
                    "description": metadata.description,
                    "transcript": metadata.transcript,
                }
            )
            predictor = get_predictor("text")
            prediction = predictor.predict(
                title=metadata.title,
                description=metadata.description,
                transcript=metadata.transcript,
                thumbnail=metadata.thumbnail_url,
            )
            result = prediction.as_dict()
            result["video_id"] = metadata.video_id
            result["input_source"] = metadata.source
            result["used_fields"] = {
                "title": is_meaningful_text(metadata.title),
                "description": is_meaningful_text(metadata.description),
                "transcript": is_meaningful_text(metadata.transcript),
            }
            result["input"] = {
                "title": metadata.title,
                "description": metadata.description,
                "thumbnail_url": metadata.thumbnail_url,
            }
            result["warnings"] = merge_warnings(metadata.warnings, result.get("warnings", []))
            return jsonify(result)
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/youtube-metadata", methods=["POST", "OPTIONS"])
    def youtube_metadata_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        video_id = extract_video_id(clean_text(payload.get("video_id") or payload.get("url") or ""))
        page_data = payload.get("page_data") or {}
        require_youtube_api = bool(payload.get("require_youtube_api"))
        use_audio_fallback = bool(payload.get("use_audio_fallback", not require_youtube_api))

        try:
            if require_youtube_api and not os.environ.get("YOUTUBE_API_KEY"):
                raise ValueError("YOUTUBE_API_KEY is required to load data through YouTube Data API v3.")
            api_metadata = fetch_youtube_metadata(video_id, use_audio_fallback=use_audio_fallback)
            if require_youtube_api and api_metadata.source != "youtube_api":
                raise ValueError("Could not load this video through YouTube Data API v3.")
            metadata = merge_metadata(api_metadata, page_data, use_transcript=True)
            return jsonify(
                {
                    "video_id": metadata.video_id,
                    "title": metadata.title,
                    "description": metadata.description,
                    "transcript": metadata.transcript,
                    "thumbnail_url": metadata.thumbnail_url,
                    "input_source": metadata.source,
                    "used_youtube_data_api_v3": metadata.source == "youtube_api",
                    "warnings": list(metadata.warnings),
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/youtube-transcript-whisper", methods=["POST", "OPTIONS"])
    def youtube_transcript_whisper_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        video_id = extract_video_id(clean_text(payload.get("video_id") or payload.get("url") or ""))
        force_refresh = bool(payload.get("force_refresh"))

        try:
            if not video_id:
                raise ValueError("Missing or invalid YouTube video id.")
            result = generate_whisper_transcript(video_id, force_refresh=force_refresh)
            if not result.transcript:
                return jsonify(result.as_dict()), 400
            return jsonify(result.as_dict())
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/youtube-audio-download", methods=["POST", "OPTIONS"])
    def youtube_audio_download_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        video_id = extract_video_id(clean_text(payload.get("video_id") or payload.get("url") or ""))

        try:
            if not video_id:
                raise ValueError("Missing or invalid YouTube video id.")
            audio_path, warnings = download_youtube_audio(
                video_id,
                max_duration_seconds=None,
                context="Audio download",
            )
            if not audio_path:
                raise ValueError(" ".join(warnings) or "Could not download audio.")

            return jsonify(
                {
                    "video_id": video_id,
                    "audio_path": workspace_relative_path(audio_path),
                    "duration_seconds": audio_duration_seconds(audio_path),
                    "download_url": f"/api/youtube-audio/{video_id}",
                    "filename": f"{video_id}.wav",
                    "warnings": list(warnings),
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/youtube-audio/<video_id>")
    def youtube_audio_file_api(video_id: str):
        clean_video_id = extract_video_id(video_id)
        if not clean_video_id:
            return jsonify({"error": "Missing or invalid YouTube video id."}), 400

        audio_path = (BASE_DIR / "instance" / "youtube_audio" / f"{clean_video_id}.wav").resolve()
        audio_root = (BASE_DIR / "instance" / "youtube_audio").resolve()
        try:
            audio_path.relative_to(audio_root)
        except ValueError:
            return jsonify({"error": "Invalid audio path."}), 400
        if not audio_path.exists():
            return jsonify({"error": "Audio has not been prepared yet."}), 404

        return send_file(
            audio_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"{clean_video_id}.wav",
        )

    @app.route("/api/labels", methods=["GET", "POST", "OPTIONS"])
    def labels_api():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method == "GET":
            return jsonify(dataset_summary())

        payload = request.get_json(silent=True) or {}
        try:
            row = build_real_dataset_row(payload)
            ensure_real_dataset()
            existing_video_ids = dataset_video_ids()
            duplicate = row["video_id"] in existing_video_ids
            with REAL_DATASET_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=REAL_DATASET_COLUMNS)
                writer.writerow(row)
            if row["queue_source"] == "dataset_queue":
                update_queue_status(row["video_id"], "saved")
            if row["queue_source"] == "synthetic_labeling_pool":
                update_synthetic_labeling_status(
                    row["video_id"],
                    "saved",
                    reviewed_binary_label=row["binary_label"],
                    reviewed_consistency_label=row["consistency_label"],
                    reviewed_mismatch_type=row["mismatch_type"],
                    reviewed_support_rationale=row["support_rationale"],
                )
            summary = dataset_summary()
            return jsonify(
                {
                    "saved": True,
                    "duplicate_video_id": duplicate,
                    "row": row,
                    "summary": summary,
                    "message": (
                        "Saved label, but this video_id already existed in the dataset."
                        if duplicate
                        else "Saved real Sinhala YouTube label."
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/asr-samples", methods=["GET", "POST", "OPTIONS"])
    def asr_samples_api():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method == "GET":
            return jsonify(asr_manifest_summary())

        payload = request.get_json(silent=True) or {}
        try:
            row = build_asr_manifest_row(payload)
            ensure_asr_manifest()
            duplicate = row["audio_path"].lower() in {
                clean_text(existing.get("audio_path", "")).lower() for existing in asr_manifest_rows()
            }
            with ASR_MANIFEST_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=ASR_MANIFEST_COLUMNS)
                writer.writerow(row)
            return jsonify(
                {
                    "saved": True,
                    "duplicate_audio_path": duplicate,
                    "row": row,
                    "summary": asr_manifest_summary(),
                    "message": (
                        "Saved ASR sample, but this audio path already existed in the manifest."
                        if duplicate
                        else "Saved ASR sample."
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/asr-youtube-candidates", methods=["POST", "OPTIONS"])
    def asr_youtube_candidates_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        topic = clean_text(payload.get("topic", ""))
        query = clean_text(payload.get("query", "")) or TOPIC_SEARCH_QUERIES.get(topic, "")
        max_results = int(payload.get("max_results", 10) or 10)
        page_token = clean_text(payload.get("page_token", ""))
        exclude_saved = bool(payload.get("exclude_saved", True))

        try:
            if topic and topic not in TOPICS:
                raise ValueError(f"Topic must be one of: {', '.join(TOPICS)}.")
            candidates, meta = search_youtube_videos(
                query=query,
                max_results=max_results,
                page_token=page_token,
            )
            saved_ids = asr_manifest_video_ids() if exclude_saved else set()
            filtered = [candidate for candidate in candidates if candidate.video_id not in saved_ids]
            return jsonify(
                {
                    "topic": topic,
                    "query": query,
                    "exclude_saved": exclude_saved,
                    "excluded_count": len(candidates) - len(filtered),
                    "candidates": [candidate.as_dict() for candidate in filtered],
                    "search_meta": meta,
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/youtube-search", methods=["POST", "OPTIONS"])
    def youtube_search_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        topic = clean_text(payload.get("topic", ""))
        query = clean_text(payload.get("query", "")) or TOPIC_SEARCH_QUERIES.get(topic, "")
        max_results = int(payload.get("max_results", 10) or 10)
        page_token = clean_text(payload.get("page_token", ""))
        exclude_labeled = bool(payload.get("exclude_labeled", True))

        try:
            if topic and topic not in TOPICS:
                raise ValueError(f"Topic must be one of: {', '.join(TOPICS)}.")
            candidates, meta = search_youtube_videos(
                query=query,
                max_results=max_results,
                page_token=page_token,
            )
            labeled_ids = dataset_video_ids() if exclude_labeled else set()
            filtered = [candidate for candidate in candidates if candidate.video_id not in labeled_ids]
            return jsonify(
                {
                    "topic": topic,
                    "query": query,
                    "exclude_labeled": exclude_labeled,
                    "excluded_count": len(candidates) - len(filtered),
                    "candidates": [candidate.as_dict() for candidate in filtered],
                    "search_meta": meta,
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/youtube-captioned-search", methods=["POST", "OPTIONS"])
    def youtube_captioned_search_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        topic = clean_text(payload.get("topic", ""))
        query = clean_text(payload.get("query", "")) or TOPIC_SEARCH_QUERIES.get(topic, "")
        max_results = int(payload.get("max_results", 10) or 10)
        page_token = clean_text(payload.get("page_token", ""))
        exclude_labeled = bool(payload.get("exclude_labeled", True))

        try:
            if topic and topic not in TOPICS:
                raise ValueError(f"Topic must be one of: {', '.join(TOPICS)}.")
            candidates, meta = search_youtube_videos(
                query=query,
                max_results=max_results,
                page_token=page_token,
                video_caption="closedCaption",
            )
            labeled_ids = dataset_video_ids() if exclude_labeled else set()
            captioned_candidates = []
            skipped_labeled = 0
            skipped_no_sinhala_caption = 0

            for candidate in candidates:
                if candidate.video_id in labeled_ids:
                    skipped_labeled += 1
                    continue
                metadata = fetch_youtube_metadata(
                    candidate.video_id,
                    use_audio_fallback=False,
                    transcript_languages=("si",),
                    allow_transcript_language_fallback=False,
                )
                if not is_meaningful_text(metadata.transcript):
                    skipped_no_sinhala_caption += 1
                    continue
                enriched = candidate.as_dict()
                enriched.update(
                    {
                        "title": metadata.title or candidate.title,
                        "description": metadata.description or candidate.description,
                        "thumbnail_url": metadata.thumbnail_url or candidate.thumbnail_url,
                        "transcript": metadata.transcript,
                        "transcript_source": "youtube_public_sinhala_caption",
                        "caption_language": "si",
                    }
                )
                captioned_candidates.append(enriched)

            return jsonify(
                {
                    "topic": topic,
                    "query": query,
                    "exclude_labeled": exclude_labeled,
                    "skipped_labeled": skipped_labeled,
                    "skipped_no_sinhala_caption": skipped_no_sinhala_caption,
                    "candidates": captioned_candidates,
                    "search_meta": meta,
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/dataset-queue/build", methods=["POST", "OPTIONS"])
    def dataset_queue_build_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        try:
            summary = build_dataset_queue(
                topics=parse_requested_topics(payload.get("topics") or payload.get("topic") or "all"),
                target_per_topic=max(1, min(int(payload.get("target_per_topic", 5) or 5), 50)),
                collection_mode=parse_collection_mode(payload.get("collection_mode", "balanced")),
                captioned_only=bool(payload.get("captioned_only", True)),
                exclude_labeled=bool(payload.get("exclude_labeled", True)),
                exclude_queued=bool(payload.get("exclude_queued", True)),
            )
            return jsonify(summary)
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/dataset-queue/next", methods=["GET", "OPTIONS"])
    def dataset_queue_next_api():
        if request.method == "OPTIONS":
            return ("", 204)

        topic = clean_text(request.args.get("topic", ""))
        try:
            candidate = next_queue_candidate(topic=topic)
            if not candidate:
                return jsonify({"candidate": None, "message": "No new queue candidates are available."})
            update_queue_status(candidate["video_id"], "in_review")
            candidate["queue_status"] = "in_review"
            return jsonify({"candidate": candidate, "quality": dataset_quality_summary()})
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/dataset-queue/status", methods=["POST", "OPTIONS"])
    def dataset_queue_status_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        try:
            video_id = extract_video_id(clean_text(payload.get("video_id") or payload.get("url") or ""))
            status = clean_text(payload.get("queue_status") or payload.get("status") or "")
            if not video_id:
                raise ValueError("Missing or invalid YouTube video id.")
            updated = update_queue_status(video_id, status)
            if not updated:
                raise ValueError("Video was not found in the dataset queue.")
            return jsonify({"updated": True, "video_id": video_id, "queue_status": status, "quality": dataset_quality_summary()})
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/synthetic-labeling/next", methods=["GET", "OPTIONS"])
    def synthetic_labeling_next_api():
        if request.method == "OPTIONS":
            return ("", 204)

        topic = clean_text(request.args.get("topic", ""))
        try:
            candidate = next_synthetic_labeling_candidate(topic=topic)
            if not candidate:
                return jsonify({"candidate": None, "message": "No new synthetic labeling candidates are available."})
            update_synthetic_labeling_status(candidate["video_id"], "in_review")
            candidate["label_status"] = "in_review"
            candidate["queue_status"] = "in_review"
            return jsonify(
                {
                    "candidate": public_synthetic_candidate(candidate),
                    "synthetic_pool_status_counts": synthetic_labeling_status_counts(),
                    "quality": dataset_quality_summary(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/synthetic-labeling/status", methods=["POST", "OPTIONS"])
    def synthetic_labeling_status_api():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        try:
            video_id = clean_text(payload.get("video_id") or payload.get("synthetic_id") or "")
            status = clean_text(payload.get("label_status") or payload.get("queue_status") or payload.get("status") or "")
            if not video_id:
                raise ValueError("Missing synthetic video id.")
            updated = update_synthetic_labeling_status(video_id, status)
            if not updated:
                raise ValueError("Synthetic candidate was not found in the labeling pool.")
            return jsonify(
                {
                    "updated": True,
                    "video_id": video_id,
                    "label_status": status,
                    "synthetic_pool_status_counts": synthetic_labeling_status_counts(),
                    "quality": dataset_quality_summary(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - API returns readable errors.
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/dataset-quality")
    def dataset_quality_api():
        return jsonify(dataset_quality_summary())

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "models": model_artifacts_available(),
                "default_model": default_model_type(),
                "youtube_api_key_loaded": bool(os.environ.get("YOUTUBE_API_KEY")),
                "audio_transcript": audio_transcript_status(),
            }
        )

    return app


def validate_text_inputs(form_data: dict[str, str]) -> None:
    if not any(is_meaningful_text(form_data.get(field, "")) for field in ("title", "description", "transcript")):
        raise ValueError(
            "Enter meaningful title, description, or transcript text before predicting. "
            "Placeholders such as '...' are not enough for classification."
        )


def merge_warnings(*warning_groups) -> list[str]:
    merged: list[str] = []
    for warnings in warning_groups:
        for warning in warnings or []:
            if warning and warning not in merged:
                merged.append(warning)
    return merged


def ensure_real_dataset() -> None:
    ensure_csv_columns(REAL_DATASET_PATH, REAL_DATASET_COLUMNS)


def ensure_youtube_queue() -> None:
    ensure_csv_columns(YOUTUBE_QUEUE_PATH, YOUTUBE_QUEUE_COLUMNS)


def ensure_synthetic_labeling_pool() -> None:
    ensure_csv_columns(SYNTHETIC_LABELING_POOL_PATH, SYNTHETIC_LABELING_POOL_COLUMNS)


def ensure_asr_manifest() -> None:
    ensure_csv_columns(ASR_MANIFEST_PATH, ASR_MANIFEST_COLUMNS)


def ensure_csv_columns(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
        return

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        existing_columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    merged_columns = existing_columns + [column for column in columns if column not in existing_columns]
    if merged_columns == existing_columns:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean_text(row.get(column, "")) for column in merged_columns})


def dataset_rows() -> list[dict[str, str]]:
    ensure_real_dataset()
    with REAL_DATASET_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def dataset_video_ids() -> set[str]:
    return {clean_text(row.get("video_id", "")) for row in dataset_rows() if clean_text(row.get("video_id", ""))}


def youtube_queue_rows() -> list[dict[str, str]]:
    ensure_youtube_queue()
    with YOUTUBE_QUEUE_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def queued_video_ids() -> set[str]:
    return {
        clean_text(row.get("video_id", ""))
        for row in youtube_queue_rows()
        if clean_text(row.get("video_id", ""))
    }


def synthetic_labeling_rows() -> list[dict[str, str]]:
    ensure_synthetic_labeling_pool()
    with SYNTHETIC_LABELING_POOL_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def synthetic_labeling_status_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in synthetic_labeling_rows():
        status = clean_text(row.get("label_status", "")) or "new"
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_synthetic_labeling_rows(rows: list[dict[str, str]]) -> None:
    ensure_synthetic_labeling_pool()
    with SYNTHETIC_LABELING_POOL_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SYNTHETIC_LABELING_POOL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean_text(row.get(column, "")) for column in SYNTHETIC_LABELING_POOL_COLUMNS})


def update_synthetic_labeling_status(
    video_id: str,
    status: str,
    reviewed_binary_label: str = "",
    reviewed_consistency_label: str = "",
    reviewed_mismatch_type: str = "",
    reviewed_support_rationale: str = "",
) -> bool:
    normalized_status = clean_text(status)
    if normalized_status not in QUEUE_STATUSES:
        raise ValueError(f"label_status must be one of: {', '.join(sorted(QUEUE_STATUSES))}.")
    rows = synthetic_labeling_rows()
    updated = False
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for row in rows:
        if clean_text(row.get("video_id", "")) == video_id:
            row["label_status"] = normalized_status
            if reviewed_binary_label:
                binary_label = normalize_binary_label(reviewed_binary_label)
                consistency_label = normalize_consistency_label(reviewed_consistency_label, binary_label)
                mismatch_type = normalize_mismatch_type(reviewed_mismatch_type, consistency_label)
                row["reviewed_binary_label"] = binary_label
                row["reviewed_consistency_label"] = consistency_label
                row["reviewed_mismatch_type"] = mismatch_type
                row["reviewed_support_rationale"] = clean_text(reviewed_support_rationale)
                row["reviewed_at"] = now
            updated = True
    if updated:
        write_synthetic_labeling_rows(rows)
    return updated


def write_queue_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    ensure_youtube_queue()
    with YOUTUBE_QUEUE_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=YOUTUBE_QUEUE_COLUMNS)
        for row in rows:
            writer.writerow({column: clean_text(row.get(column, "")) for column in YOUTUBE_QUEUE_COLUMNS})


def update_queue_status(video_id: str, status: str) -> bool:
    normalized_status = clean_text(status)
    if normalized_status not in QUEUE_STATUSES:
        raise ValueError(f"queue_status must be one of: {', '.join(sorted(QUEUE_STATUSES))}.")
    rows = youtube_queue_rows()
    updated = False
    for row in rows:
        if clean_text(row.get("video_id", "")) == video_id:
            row["queue_status"] = normalized_status
            updated = True
    if updated:
        with YOUTUBE_QUEUE_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=YOUTUBE_QUEUE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: clean_text(row.get(column, "")) for column in YOUTUBE_QUEUE_COLUMNS})
    return updated


def asr_manifest_rows() -> list[dict[str, str]]:
    ensure_asr_manifest()
    with ASR_MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def asr_manifest_video_ids() -> set[str]:
    video_ids = set()
    for row in asr_manifest_rows():
        audio_path = clean_text(row.get("audio_path", ""))
        if audio_path:
            video_id = Path(audio_path).stem
            if video_id:
                video_ids.add(video_id)
    return video_ids


def dataset_summary() -> dict:
    rows = dataset_rows()
    label_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    group_ids = set()
    for row in rows:
        label = clean_text(row.get("binary_label", ""))
        topic = clean_text(row.get("topic", ""))
        group_id = clean_text(row.get("group_id", ""))
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        if group_id:
            group_ids.add(group_id)
    return {
        "dataset_path": str(REAL_DATASET_PATH),
        "row_count": len(rows),
        "group_count": len(group_ids),
        "label_counts": label_counts,
        "topic_counts": topic_counts,
        "ready_for_grouped_split": len(group_ids) >= 20,
    }


def dataset_quality_summary() -> dict:
    rows = dataset_rows()
    queue_rows = youtube_queue_rows()
    synthetic_status_counts = synthetic_labeling_status_counts()
    summary = dataset_summary()
    duplicates = max(0, len([row for row in rows if clean_text(row.get("video_id", ""))]) - len(dataset_video_ids()))
    captioned_count = sum(1 for row in rows if is_meaningful_text(row.get("transcript", "")))
    generated_negative_count = sum(1 for row in rows if clean_text(row.get("generated_negative", "")).lower() == "true")
    human_reviewed_count = max(0, len(rows) - generated_negative_count)
    queue_status_counts: dict[str, int] = {}
    for row in queue_rows:
        status = clean_text(row.get("queue_status", "")) or "new"
        queue_status_counts[status] = queue_status_counts.get(status, 0) + 1

    label_counts = summary["label_counts"]
    topic_counts = summary["topic_counts"]
    readiness = {
        "rows": min(summary["row_count"] / BENCHMARK_TARGETS["rows"], 1),
        "false": min(label_counts.get("False", 0) / BENCHMARK_TARGETS["false"], 1),
        "not_false": min(label_counts.get("Not False", 0) / BENCHMARK_TARGETS["not_false"], 1),
        "groups": min(summary["group_count"] / BENCHMARK_TARGETS["groups"], 1),
        "topics_ready": sum(1 for topic in TOPICS if topic_counts.get(topic, 0) >= BENCHMARK_TARGETS["per_topic"]),
    }
    ready_for_first_benchmark = (
        summary["row_count"] >= BENCHMARK_TARGETS["rows"]
        and label_counts.get("False", 0) >= BENCHMARK_TARGETS["false"]
        and label_counts.get("Not False", 0) >= BENCHMARK_TARGETS["not_false"]
        and summary["group_count"] >= BENCHMARK_TARGETS["groups"]
        and readiness["topics_ready"] == len(TOPICS)
    )
    return {
        **summary,
        "duplicate_video_count": duplicates,
        "captioned_row_count": captioned_count,
        "no_caption_row_count": max(0, summary["row_count"] - captioned_count),
        "generated_negative_count": generated_negative_count,
        "human_reviewed_count": human_reviewed_count,
        "queue_path": str(YOUTUBE_QUEUE_PATH),
        "queue_count": len(queue_rows),
        "queue_status_counts": queue_status_counts,
        "synthetic_pool_path": str(SYNTHETIC_LABELING_POOL_PATH),
        "synthetic_pool_count": sum(synthetic_status_counts.values()),
        "synthetic_pool_status_counts": synthetic_status_counts,
        "benchmark_targets": BENCHMARK_TARGETS,
        "benchmark_progress": readiness,
        "ready_for_first_benchmark": ready_for_first_benchmark,
    }


def parse_requested_topics(value) -> list[str]:
    if isinstance(value, list):
        requested = [clean_text(topic) for topic in value if clean_text(topic)]
    else:
        text = clean_text(value)
        requested = list(TOPICS) if text.lower() == "all" else [clean_text(topic) for topic in text.split(",") if clean_text(topic)]
    if not requested:
        requested = list(TOPICS)
    invalid = [topic for topic in requested if topic not in TOPICS]
    if invalid:
        raise ValueError(f"Topic must be one of: {', '.join(TOPICS)}.")
    return requested


def parse_collection_mode(value: str) -> str:
    mode = clean_text(value) or "balanced"
    if mode not in COLLECTION_MODES:
        raise ValueError(f"collection_mode must be one of: {', '.join(COLLECTION_MODES)}.")
    return mode


def topic_queries(topic: str, collection_mode: str = "balanced") -> list[str]:
    base_query = TOPIC_SEARCH_QUERIES.get(topic, f"Sinhala {topic}")
    risk_terms = TOPIC_RISK_TERMS.get(topic, [])
    mode = parse_collection_mode(collection_mode)
    mode_terms = COLLECTION_MODE_TERMS.get(mode, COLLECTION_MODE_TERMS["balanced"])
    queries = [base_query]
    for term in mode_terms:
        queries.append(f"{base_query} {term}")
    for intent in SEARCH_INTENTS:
        queries.append(f"{base_query} {intent}")
    for term in risk_terms:
        queries.append(f"{base_query} {term}")
    unique_queries: list[str] = []
    for query in queries:
        cleaned = clean_text(query)
        if cleaned and cleaned not in unique_queries:
            unique_queries.append(cleaned)
    return unique_queries


def count_terms(text: str, terms: list[str]) -> int:
    lowered = clean_text(text).lower()
    return sum(1 for term in terms if term.lower() in lowered)


def normalized_token_set(text: str) -> set[str]:
    lowered = normalize_for_consistency(text)
    return {token for token in re.split(r"[^\w\u0D80-\u0DFF]+", lowered) if len(token) > 2}


def text_similarity(left: str, right: str) -> float:
    return consistency_similarity(left, right)


def duplicate_similarity(candidate: dict[str, str], existing_rows: list[dict[str, str]]) -> float:
    candidate_text = " ".join(
        [
            clean_text(candidate.get("title", "")),
            clean_text(candidate.get("description", "")),
            clean_text(candidate.get("transcript", ""))[:1200],
        ]
    )
    best = 0.0
    for row in existing_rows:
        existing_text = " ".join(
            [
                clean_text(row.get("title", "")),
                clean_text(row.get("description", "")),
                clean_text(row.get("transcript", ""))[:1200],
            ]
        )
        best = max(best, text_similarity(candidate_text, existing_text))
    return round(best, 3)


def model_prediction_for_candidate(title: str, description: str, transcript: str, thumbnail_url: str) -> tuple[str, str]:
    if not model_artifacts_available().get("text"):
        return "", ""
    try:
        prediction = get_predictor("text").predict(
            title=title,
            description=description,
            transcript=transcript,
            thumbnail=thumbnail_url,
        )
        return prediction.label, f"{prediction.confidence:.4f}"
    except Exception:
        return "", ""


def score_queue_candidate(candidate: dict[str, str], existing_rows: list[dict[str, str]]) -> tuple[float, float]:
    combined_text = " ".join(
        [
            clean_text(candidate.get("title", "")),
            clean_text(candidate.get("description", "")),
            clean_text(candidate.get("transcript", "")),
        ]
    )
    transcript_length = len(clean_text(candidate.get("transcript", "")).split())
    source_mentions = count_terms(combined_text, SOURCE_TERMS)
    sensational_hits = count_terms(combined_text, SENSATIONAL_TERMS)
    claim_hits = count_terms(combined_text, CLAIM_TERMS)
    high_risk_topic = 1 if clean_text(candidate.get("topic", "")) in {"Finance", "Government", "Health", "Politics", "Weather"} else 0
    duplicate_penalty = duplicate_similarity(candidate, existing_rows)

    claim_score = min(1.0, (claim_hits * 0.22) + (sensational_hits * 0.12) + min(transcript_length / 450, 0.25))
    risk_score = min(1.0, (sensational_hits * 0.18) + (high_risk_topic * 0.18) + ((1 if source_mentions == 0 else 0) * 0.18) + claim_score * 0.35)
    risk_score = max(0.0, risk_score - duplicate_penalty * 0.4)
    return round(risk_score, 3), round(claim_score, 3)


def build_queue_row(candidate, metadata: YouTubeMetadata, topic: str, query: str, existing_rows: list[dict[str, str]]) -> dict[str, str]:
    title = metadata.title or candidate.title
    description = metadata.description or candidate.description
    thumbnail_url = metadata.thumbnail_url or candidate.thumbnail_url
    row = {
        "video_id": metadata.video_id or candidate.video_id,
        "url": f"https://www.youtube.com/watch?v={metadata.video_id or candidate.video_id}",
        "topic": topic,
        "title": title,
        "description": description,
        "transcript": metadata.transcript,
        "caption_language": "si" if metadata.transcript else "",
        "channel_title": candidate.channel_title,
        "published_at": candidate.published_at,
        "thumbnail_url": thumbnail_url,
        "collection_query": query,
        "suggested_group_id": f"{topic.lower()}-{metadata.video_id or candidate.video_id}",
        "queue_status": "new",
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    risk_score, claim_score = score_queue_candidate(row, existing_rows)
    model_prediction, model_confidence = model_prediction_for_candidate(title, description, metadata.transcript, thumbnail_url)
    row["risk_score"] = f"{risk_score:.3f}"
    row["claim_score"] = f"{claim_score:.3f}"
    row["model_prediction"] = model_prediction
    row["model_confidence"] = model_confidence
    return row


def build_dataset_queue(
    topics: list[str],
    target_per_topic: int,
    collection_mode: str = "balanced",
    captioned_only: bool = True,
    exclude_labeled: bool = True,
    exclude_queued: bool = True,
) -> dict:
    mode = parse_collection_mode(collection_mode)
    ensure_youtube_queue()
    labeled_ids = dataset_video_ids() if exclude_labeled else set()
    queued_ids = queued_video_ids() if exclude_queued else set()
    existing_rows = dataset_rows() + youtube_queue_rows()
    seen_ids = set(labeled_ids) | set(queued_ids)
    summary = {
        "queue_path": str(YOUTUBE_QUEUE_PATH),
        "collection_mode": mode,
        "collection_mode_label": COLLECTION_MODES[mode],
        "topics": {},
        "saved_total": 0,
        "skipped_labeled": 0,
        "skipped_queued": 0,
        "skipped_no_sinhala_caption": 0,
        "errors": [],
    }

    for topic in topics:
        topic_saved = 0
        topic_checked = 0
        for query in topic_queries(topic, mode):
            if topic_saved >= target_per_topic:
                break
            try:
                candidates, _meta = search_youtube_videos(
                    query=query,
                    max_results=25,
                    video_caption="closedCaption" if captioned_only else "",
                )
            except Exception as exc:  # noqa: BLE001 - keep batch collection moving.
                summary["errors"].append(f"{topic}: search failed for {query}: {exc}")
                continue
            rows_to_write: list[dict[str, str]] = []
            for candidate in candidates:
                topic_checked += 1
                if candidate.video_id in labeled_ids:
                    summary["skipped_labeled"] += 1
                    continue
                if candidate.video_id in queued_ids or candidate.video_id in seen_ids:
                    summary["skipped_queued"] += 1
                    continue
                try:
                    metadata = fetch_youtube_metadata(
                        candidate.video_id,
                        use_audio_fallback=False,
                        transcript_languages=("si",),
                        allow_transcript_language_fallback=False,
                    )
                except Exception as exc:  # noqa: BLE001 - keep batch collection moving.
                    summary["errors"].append(f"{candidate.video_id}: metadata failed: {exc}")
                    continue
                if captioned_only and not is_meaningful_text(metadata.transcript):
                    summary["skipped_no_sinhala_caption"] += 1
                    continue
                queue_row = build_queue_row(candidate, metadata, topic, query, existing_rows + rows_to_write)
                rows_to_write.append(queue_row)
                seen_ids.add(candidate.video_id)
                topic_saved += 1
                if topic_saved >= target_per_topic:
                    break
            write_queue_rows(rows_to_write)
        summary["topics"][topic] = {"saved": topic_saved, "checked": topic_checked}
        summary["saved_total"] += topic_saved

    summary["quality"] = dataset_quality_summary()
    return summary


def queue_priority(row: dict[str, str], topic_counts: dict[str, int]) -> float:
    try:
        risk = float(row.get("risk_score", 0) or 0)
    except ValueError:
        risk = 0.0
    try:
        claim = float(row.get("claim_score", 0) or 0)
    except ValueError:
        claim = 0.0
    try:
        confidence = float(row.get("model_confidence", 0) or 0)
    except ValueError:
        confidence = 0.0
    uncertainty = 1 - abs(confidence - 0.5) * 2 if confidence else 0.25
    topic = clean_text(row.get("topic", ""))
    topic_boost = max(0, BENCHMARK_TARGETS["per_topic"] - topic_counts.get(topic, 0)) / BENCHMARK_TARGETS["per_topic"]
    return (risk * 0.45) + (claim * 0.3) + (uncertainty * 0.15) + (topic_boost * 0.1)


def next_queue_candidate(topic: str = "") -> dict[str, str] | None:
    if topic and topic not in TOPICS:
        raise ValueError(f"Topic must be one of: {', '.join(TOPICS)}.")
    rows = [
        row
        for row in youtube_queue_rows()
        if clean_text(row.get("queue_status", "")) in {"", "new"}
        and (not topic or clean_text(row.get("topic", "")) == topic)
    ]
    if not rows:
        return None
    topic_counts = dataset_summary()["topic_counts"]
    rows.sort(key=lambda row: queue_priority(row, topic_counts), reverse=True)
    return rows[0]


def next_synthetic_labeling_candidate(topic: str = "") -> dict[str, str] | None:
    if topic and topic not in TOPICS:
        raise ValueError(f"Topic must be one of: {', '.join(TOPICS)}.")
    labeled_ids = dataset_video_ids()
    rows = [
        row
        for row in synthetic_labeling_rows()
        if clean_text(row.get("label_status", "")) in {"", "new"}
        and clean_text(row.get("video_id", "")) not in labeled_ids
        and (not topic or clean_text(row.get("topic", "")) == topic)
    ]
    if not rows:
        return None
    topic_counts = dataset_summary()["topic_counts"]

    def priority(row: dict[str, str]) -> tuple[int, int, str]:
        row_topic = clean_text(row.get("topic", ""))
        label = clean_text(row.get("seed_binary_label", ""))
        topic_gap = max(0, BENCHMARK_TARGETS["per_topic"] - topic_counts.get(row_topic, 0))
        false_boost = 1 if label == "False" else 0
        return (topic_gap, false_boost, clean_text(row.get("video_id", "")))

    rows.sort(key=priority, reverse=True)
    candidate = dict(rows[0])
    candidate["queue_status"] = clean_text(candidate.get("label_status", "")) or "new"
    candidate["queue_source"] = "synthetic_labeling_pool"
    candidate["caption_source"] = "synthetic_transcript"
    candidate["thumbnail_url"] = ""
    return candidate


def public_synthetic_candidate(candidate: dict[str, str]) -> dict[str, str]:
    """Return only fields the human labeler should see."""

    hidden_prefixes = ("seed_", "reviewed_")
    return {
        key: value
        for key, value in candidate.items()
        if not key.startswith(hidden_prefixes)
    }


def asr_manifest_summary() -> dict:
    rows = asr_manifest_rows()
    split_counts = {split: 0 for split in ASR_SPLITS}
    source_counts: dict[str, int] = {}
    total_duration = 0.0
    for row in rows:
        split = clean_text(row.get("split", "")).lower()
        source = clean_text(row.get("source", ""))
        if split in split_counts:
            split_counts[split] += 1
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
        try:
            total_duration += float(row.get("duration_seconds", 0) or 0)
        except (TypeError, ValueError):
            pass
    return {
        "manifest_path": str(ASR_MANIFEST_PATH),
        "row_count": len(rows),
        "split_counts": split_counts,
        "source_counts": source_counts,
        "total_duration_seconds": round(total_duration, 2),
        "ready_for_first_eval": split_counts.get("test", 0) >= 10,
    }


def build_real_dataset_row(payload: dict) -> dict[str, str]:
    video_id = extract_video_id(clean_text(payload.get("video_id") or payload.get("url") or ""))
    url = clean_text(payload.get("url", ""))
    if video_id and not url:
        url = f"https://www.youtube.com/watch?v={video_id}"

    title = clean_text(payload.get("title", ""))
    description = clean_text(payload.get("description", ""))
    transcript = clean_text(payload.get("transcript", ""))
    topic = clean_text(payload.get("topic", ""))
    original_label = normalize_label(payload.get("original_label", ""))
    binary_label = normalize_binary_label(payload.get("binary_label", ""))
    label_source = clean_text(payload.get("label_source", "Human annotation"))
    annotator_notes = clean_text(payload.get("annotator_notes", ""))
    group_id = clean_text(payload.get("group_id", ""))
    claim_text = clean_text(payload.get("claim_text", ""))
    evidence_url = clean_text(payload.get("evidence_url", ""))
    evidence_summary = clean_text(payload.get("evidence_summary", ""))
    label_confidence = clean_text(payload.get("label_confidence", "medium")) or "medium"
    queue_source = clean_text(payload.get("queue_source", "manual")) or "manual"
    caption_source = clean_text(payload.get("caption_source", ""))
    consistency_label = normalize_consistency_label(payload.get("consistency_label", ""), binary_label)
    mismatch_type = normalize_mismatch_type(payload.get("mismatch_type", ""), consistency_label)
    support_rationale = clean_text(payload.get("support_rationale", "")) or evidence_summary
    generated_negative = "true" if clean_text(payload.get("generated_negative", "")).lower() == "true" else "false"
    source_title_video_id = clean_text(payload.get("source_title_video_id", "")) or video_id
    source_transcript_video_id = clean_text(payload.get("source_transcript_video_id", "")) or video_id
    consistency_features = compute_consistency_features(title, description, transcript).as_dict()

    if not video_id:
        raise ValueError("Missing or invalid YouTube video id.")
    if topic not in TOPICS:
        raise ValueError(f"Topic must be one of: {', '.join(TOPICS)}.")
    expected_binary_label = "False" if consistency_label == INCONSISTENT_LABEL else "Not False"
    if binary_label != expected_binary_label:
        binary_label = expected_binary_label
    if binary_label not in {"False", "Not False"}:
        raise ValueError("binary_label must be False or Not False.")
    if not original_label:
        original_label = binary_label
    if not group_id:
        raise ValueError("group_id is required so repeated claims do not leak across train/test splits.")
    if not annotator_notes:
        raise ValueError("annotator_notes is required. Add the evidence reason for this human label.")
    if binary_label == "False" and consistency_label != INCONSISTENT_LABEL:
        raise ValueError("False labels must use consistency_label=Inconsistent.")
    if binary_label == "False" and mismatch_type == "consistent":
        raise ValueError("False labels need a mismatch_type such as title_transcript_mismatch or clickbait_overclaim.")
    if binary_label == "False" and not is_meaningful_text(support_rationale):
        raise ValueError("support_rationale is required before saving a mismatch label.")
    if binary_label == "Not False" and not is_meaningful_text(annotator_notes):
        raise ValueError("A Not False label needs an annotator note explaining the reason.")
    if not any(is_meaningful_text(field) for field in (title, description, transcript)):
        raise ValueError("Save at least one meaningful text field: title, description, or transcript.")

    return {
        "video_id": video_id,
        "url": url or f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "description": description,
        "transcript": transcript,
        "topic": topic,
        "original_label": original_label,
        "binary_label": binary_label,
        "label_source": label_source or "Human annotation",
        "annotator_notes": annotator_notes,
        "group_id": group_id,
        "claim_text": claim_text,
        "evidence_url": evidence_url,
        "evidence_summary": evidence_summary,
        "label_confidence": label_confidence,
        "queue_source": queue_source,
        "caption_source": caption_source,
        "consistency_label": consistency_label,
        "mismatch_type": mismatch_type,
        "support_rationale": support_rationale,
        "generated_negative": generated_negative,
        "source_title_video_id": source_title_video_id,
        "source_transcript_video_id": source_transcript_video_id,
        "title_transcript_similarity": f"{consistency_features['title_transcript_similarity']:.3f}",
        "description_transcript_similarity": f"{consistency_features['description_transcript_similarity']:.3f}",
        "title_description_similarity": f"{consistency_features['title_description_similarity']:.3f}",
    }


def build_asr_manifest_row(payload: dict) -> dict[str, str]:
    audio_path = clean_text(payload.get("audio_path", ""))
    video_id = extract_video_id(clean_text(payload.get("video_id") or payload.get("url") or ""))
    transcript = clean_asr_reference_transcript(payload.get("transcript", ""))
    split = clean_text(payload.get("split", "test")).lower()
    source = clean_text(payload.get("source", "youtube_whisper")) or "youtube_whisper"
    duration_value = payload.get("duration_seconds", "")

    if not audio_path:
        if video_id:
            audio_path = str(BASE_DIR / "instance" / "youtube_audio" / f"{video_id}.wav")
        else:
            raise ValueError("audio_path is required and must point to an existing downloaded audio file.")
    resolved_audio_path = resolve_workspace_path(audio_path)
    if not resolved_audio_path.exists():
        if not video_id:
            raise ValueError("audio_path is required and must point to an existing downloaded audio file.")
        downloaded_audio_path, download_warnings = download_youtube_audio(video_id)
        if not downloaded_audio_path:
            warning_text = " ".join(download_warnings)
            raise ValueError(f"Could not prepare audio for ASR manifest: {warning_text}")
        resolved_audio_path = downloaded_audio_path.resolve()
    if not is_meaningful_text(transcript):
        raise ValueError("Human reference transcript is required before saving an ASR sample.")
    quality_errors = asr_reference_quality_errors(transcript)
    if quality_errors:
        raise ValueError("Clean Sinhala reference transcript required: " + " ".join(quality_errors))
    if split not in ASR_SPLITS:
        raise ValueError(f"split must be one of: {', '.join(ASR_SPLITS)}.")

    try:
        duration_seconds = float(duration_value)
    except (TypeError, ValueError):
        duration_seconds = 0.0
    if duration_seconds <= 0:
        duration_seconds = float(audio_duration_seconds(resolved_audio_path))
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0.")

    return {
        "audio_path": workspace_relative_path(resolved_audio_path),
        "transcript": transcript,
        "split": split,
        "source": source,
        "duration_seconds": f"{duration_seconds:.2f}".rstrip("0").rstrip("."),
    }


def clean_asr_reference_transcript(value: object) -> str:
    """Remove ASR draft noise before storing a human reference transcript."""

    text = clean_text(value)
    text = ASR_TIMESTAMP_RE.sub(" ", text)
    text = ASR_BRACKET_RE.sub(" ", text)
    text = text.replace("\ufeff", " ")
    return clean_text(text)


def asr_reference_quality_errors(transcript: str) -> list[str]:
    cleaned = clean_asr_reference_transcript(transcript)
    errors: list[str] = []
    sinhala_chars = len(SINHALA_CHAR_RE.findall(cleaned))
    foreign_script_chars = len(NON_SINHALA_SCRIPT_RE.findall(cleaned))
    latin_chars = len(re.findall(r"[A-Za-z]", cleaned))

    if sinhala_chars < ASR_MIN_SINHALA_CHARS:
        errors.append(f"Need at least {ASR_MIN_SINHALA_CHARS} Sinhala characters after cleanup.")
    if foreign_script_chars > max(20, sinhala_chars // 6):
        errors.append("Transcript contains too much non-Sinhala script text.")
    if latin_chars > max(80, sinhala_chars // 3):
        errors.append("Transcript contains too much Latin/English text for a Sinhala ASR reference.")
    return errors


def resolve_workspace_path(value: str) -> Path:
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def workspace_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(resolved)


def saved_upload_path() -> str:
    thumbnail_file = request.files.get("thumbnail_file")
    if not thumbnail_file or not thumbnail_file.filename:
        return ""

    original_name = secure_filename(thumbnail_file.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Thumbnail uploads must be JPG, PNG, WEBP, or BMP images.")

    filename = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / filename
    thumbnail_file.save(destination)
    return str(destination)


app = create_app()


if __name__ == "__main__":
    # Defaults to the original port 5000 and debug=True, unchanged, so running
    # this file the normal standalone way (`python app.py`) is unaffected.
    # Only the integration launcher sets these env vars, so it can run
    # alongside the Tamil backend, which is fixed on port 5000.
    _port = int(os.environ.get("FALSE_CONTENT_PORT", 5000))
    _debug = os.environ.get("FALSE_CONTENT_DEBUG", "true").lower() != "false"
    app.run(debug=_debug, port=_port)
