import unittest
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from training.prepare_data import add_same_topic_hard_negatives, prepare_text_dataframe, validate_schema
from utils.consistency import (
    compute_consistency_features,
    format_consistency_text_input,
    normalize_for_consistency,
)
from training.validate_asr_manifest import validate_asr_manifest
from utils.data_utils import is_meaningful_text, normalize_binary_label


def runtime_dependencies_available() -> bool:
    return (
        os.environ.get("RUN_MODEL_TESTS") == "1"
        and
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


class BinaryLabelTests(unittest.TestCase):
    def test_four_class_labels_map_to_binary_labels(self):
        self.assertEqual(normalize_binary_label("False"), "False")
        self.assertEqual(normalize_binary_label("Misleading"), "False")
        self.assertEqual(normalize_binary_label("Reliable"), "Not False")
        self.assertEqual(normalize_binary_label("Unverified"), "Not False")

    def test_placeholder_text_is_not_meaningful(self):
        self.assertFalse(is_meaningful_text("..."))
        self.assertFalse(is_meaningful_text("   ---   "))
        self.assertTrue(is_meaningful_text("Sinhala title"))

    def test_consistency_normalization_and_feature_block(self):
        normalized = normalize_for_consistency("official award ceremony update")
        self.assertIn("නිල", normalized)
        self.assertIn("සම්මාන උළෙල", normalized)

        text = format_consistency_text_input(
            "නිල සම්මාන උළෙල",
            "official award ceremony update",
            "සම්මාන උළෙල ගැන විස්තර",
        )
        self.assertIn("[CONSISTENCY_FEATURES]", text)
        self.assertIn("title_transcript_similarity=", text)

        features = compute_consistency_features("award ceremony", "", "සම්මාන උළෙල")
        self.assertGreater(features.title_transcript_similarity, 0)

    def test_legacy_probabilities_fold_to_binary(self):
        if not runtime_dependencies_available():
            self.skipTest("torch and transformers are not installed in this Python environment")
        import torch
        from services.predictor import choose_binary_label, fold_binary_probabilities

        probabilities = torch.tensor([0.30, 0.25, 0.35, 0.10])
        id_to_label = {0: "False", 1: "Misleading", 2: "Reliable", 3: "Unverified"}

        folded = fold_binary_probabilities(probabilities, id_to_label)
        label, confidence = choose_binary_label(folded)

        self.assertAlmostEqual(folded["False"], 0.55)
        self.assertAlmostEqual(folded["Not False"], 0.45)
        self.assertEqual(label, "False")
        self.assertAlmostEqual(confidence, 0.55)

    def test_synthetic_decision_allows_high_overlap_hard_negatives(self):
        from services.predictor import choose_synthetic_model_label

        hard_negative = choose_synthetic_model_label(
            {"False": 0.70, "Not False": 0.30},
            {"title_transcript_coverage": 0.55, "description_transcript_coverage": 0.70},
        )
        unsupported_but_uncertain = choose_synthetic_model_label(
            {"False": 0.70, "Not False": 0.30},
            {"title_transcript_coverage": 0.0, "description_transcript_coverage": 0.0},
        )
        unsupported_and_strong = choose_synthetic_model_label(
            {"False": 0.95, "Not False": 0.05},
            {"title_transcript_coverage": 0.0, "description_transcript_coverage": 0.0},
        )

        self.assertEqual(hard_negative, ("False", 0.70))
        self.assertEqual(unsupported_but_uncertain, ("Not False", 0.30))
        self.assertEqual(unsupported_and_strong, ("False", 0.95))


class EvidenceSignalTests(unittest.TestCase):
    def test_missing_transcript_and_sources_create_warnings(self):
        if not runtime_dependencies_available():
            self.skipTest("torch and transformers are not installed in this Python environment")
        from services.predictor import analyze_input_evidence, build_prediction_warnings

        signals = analyze_input_evidence(
            title="Breaking 100% cure",
            description="Short claim",
            transcript="",
        )
        warnings = build_prediction_warnings(
            confidence=0.54,
            evidence_signals=signals,
            id_to_label={0: "False", 1: "Not False"},
        )

        statuses = {(item["name"], item["status"]) for item in signals}
        self.assertIn(("transcript", "missing"), statuses)
        self.assertIn(("source_mentions", "missing"), statuses)
        self.assertTrue(any("confidence is low" in warning for warning in warnings))
        self.assertTrue(any("Transcript was missing" in warning for warning in warnings))

    def test_placeholder_transcript_is_missing(self):
        if not runtime_dependencies_available():
            self.skipTest("torch and transformers are not installed in this Python environment")
        from services.predictor import analyze_input_evidence

        signals = analyze_input_evidence(title="...", description="...", transcript="...")
        statuses = {(item["name"], item["status"]) for item in signals}

        self.assertIn(("transcript", "missing"), statuses)
        self.assertIn(("description", "missing"), statuses)


class YouTubeClientTests(unittest.TestCase):
    def test_default_caption_fetch_reuses_local_transcript_cache(self):
        from services import youtube_client

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(youtube_client, "AUDIO_CACHE_DIR", Path(temp_dir)):
                youtube_client.write_cached_transcript("abc123", "cached transcript")
                with patch.object(
                    youtube_client,
                    "fetch_transcript_with_package",
                    side_effect=AssertionError("network caption lookup should not run"),
                ):
                    transcript, warnings = youtube_client.fetch_youtube_transcript("abc123")

        self.assertEqual(transcript, "cached transcript")
        self.assertEqual(warnings, ())

    def test_strict_sinhala_caption_fetch_does_not_reuse_unknown_language_cache(self):
        from services import youtube_client

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(youtube_client, "AUDIO_CACHE_DIR", Path(temp_dir)):
                youtube_client.write_cached_transcript("abc123", "possibly English cache")
                with patch.object(youtube_client, "fetch_transcript_with_package", return_value="සිංහල පිටපත"):
                    transcript, _warnings = youtube_client.fetch_youtube_transcript(
                        "abc123",
                        languages=("si",),
                        allow_language_fallback=False,
                    )

        self.assertEqual(transcript, "සිංහල පිටපත")

    def test_faster_whisper_model_is_loaded_once_for_multiple_audio_files(self):
        from services import youtube_client

        calls = {"loads": 0, "transcribes": 0}

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs):
                calls["loads"] += 1

            def transcribe(self, _path, **_kwargs):
                calls["transcribes"] += 1
                return [types.SimpleNamespace(text="සිංහල පිටපත")], None

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        youtube_client._FASTER_WHISPER_MODEL_CACHE.clear()
        try:
            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                with patch.object(youtube_client, "configured_faster_whisper_device", return_value="cpu"):
                    with patch.object(youtube_client, "configured_faster_whisper_compute_type", return_value="int8"):
                        first = youtube_client.transcribe_audio_with_faster_whisper(Path("first.m4a"))
                        second = youtube_client.transcribe_audio_with_faster_whisper(Path("second.m4a"))
        finally:
            youtube_client._FASTER_WHISPER_MODEL_CACHE.clear()

        self.assertEqual(first, "සිංහල පිටපත")
        self.assertEqual(second, "සිංහල පිටපත")
        self.assertEqual(calls, {"loads": 1, "transcribes": 2})

    def test_transcription_download_can_skip_wav_conversion(self):
        from services import youtube_client

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "abc123.m4a"
            audio_path.write_bytes(b"cached audio")
            with patch.object(youtube_client, "AUDIO_CACHE_DIR", Path(temp_dir)):
                with patch.object(youtube_client, "audio_duration_seconds", return_value=10):
                    with patch.object(youtube_client, "convert_audio_to_wav") as converter:
                        selected, warnings = youtube_client.download_youtube_audio(
                            "abc123",
                            convert_to_wav=False,
                        )

        self.assertEqual(selected, audio_path)
        self.assertEqual(warnings, ())
        converter.assert_not_called()

    def test_ytdlp_duration_filter_rejects_before_download(self):
        from services.youtube_client import ytdlp_duration_filter

        duration_filter = ytdlp_duration_filter(900, "Audio transcript fallback")

        self.assertIsNone(duration_filter({"duration": 900}, incomplete=False))
        self.assertIsNone(duration_filter({"duration": 1200}, incomplete=True))
        self.assertIn("longer than 15 minutes", duration_filter({"duration": 901}, incomplete=False))

    def test_broken_openai_whisper_does_not_crash_status(self):
        import builtins
        from services.youtube_client import openai_whisper_status

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "whisper":
                raise OSError("Could not load libtorchaudio.pyd")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            available, error = openai_whisper_status()

        self.assertFalse(available)
        self.assertIn("libtorchaudio", error)

    def test_youtube_metadata_can_skip_audio_fallback(self):
        from services import youtube_client

        calls = {"audio": 0}

        def fake_audio(_video_id):
            calls["audio"] += 1
            return "", ("audio should not run",)

        def fake_transcript(_video_id, **_kwargs):
            return "", ("captions missing",)

        def fake_get(*_args, **_kwargs):
            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "items": [
                            {
                                "snippet": {
                                    "title": "Video title",
                                    "description": "Video description",
                                    "thumbnails": {"high": {"url": "https://example.test/thumb.jpg"}},
                                }
                            }
                        ]
                    }

            return Response()

        with patch.object(youtube_client, "fetch_audio_transcript", side_effect=fake_audio):
            with patch.object(youtube_client, "fetch_youtube_transcript", side_effect=fake_transcript):
                with patch.object(youtube_client.requests, "get", side_effect=fake_get):
                    metadata = youtube_client.fetch_youtube_metadata(
                        "abc123",
                        api_key="fake-key",
                        use_audio_fallback=False,
                    )

        self.assertEqual(calls["audio"], 0)
        self.assertEqual(metadata.source, "youtube_api")
        self.assertEqual(metadata.title, "Video title")

    def test_youtube_search_can_request_captioned_videos_only(self):
        from services import youtube_client

        captured_params = {}

        def fake_get(_url, params=None, **_kwargs):
            captured_params.update(params or {})

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"items": [], "pageInfo": {"totalResults": 0, "resultsPerPage": 0}}

            return Response()

        with patch.object(youtube_client.requests, "get", side_effect=fake_get):
            candidates, _meta = youtube_client.search_youtube_videos(
                query="health sinhala",
                api_key="fake-key",
                video_caption="closedCaption",
            )

        self.assertEqual(candidates, [])
        self.assertEqual(captured_params["type"], "video")
        self.assertEqual(captured_params["videoCaption"], "closedCaption")

    def test_whisper_endpoint_rejects_invalid_video_id(self):
        from app import app

        response = app.test_client().post(
            "/api/youtube-transcript-whisper",
            json={"url": "not a youtube url"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid YouTube video id", response.get_json()["error"])

    def test_whisper_endpoint_returns_generated_transcript(self):
        import app as app_module
        from services.youtube_client import WhisperTranscriptResult

        fake_result = WhisperTranscriptResult(
            video_id="abc123",
            transcript="සිංහල පිටපත",
            warnings=("Transcript was generated from downloaded audio with Whisper.",),
            audio_duration_seconds=10,
            model_name="base",
            runtime="faster_whisper",
        )

        with patch.object(app_module, "generate_whisper_transcript", return_value=fake_result):
            response = app_module.app.test_client().post(
                "/api/youtube-transcript-whisper",
                json={"video_id": "abc123"},
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["transcript"], "සිංහල පිටපත")
        self.assertEqual(payload["source"], "whisper")

    def test_test_page_shows_generate_transcript_choice(self):
        from app import app

        response = app.test_client().get("/test")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Generate Transcript", response.data)
        self.assertIn(b"Analyze Without Transcript", response.data)

    def test_label_page_is_available(self):
        from app import app

        response = app.test_client().get("/label")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Real Dataset Labeling", response.data)
        self.assertIn(b"Find Sinhala-Captioned", response.data)
        self.assertIn(b"Save Label", response.data)

    def test_synthetic_label_page_is_available(self):
        from app import app

        response = app.test_client().get("/synthetic-label")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Synthetic Consistency Review", response.data)
        self.assertIn(b"Next Synthetic", response.data)
        self.assertIn(b"Save Review", response.data)

    def test_labels_api_saves_real_dataset_row(self):
        import app as app_module

        payload = {
            "video_id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "title": "Sinhala title",
            "description": "",
            "transcript": "Transcript text",
            "topic": "Health",
            "original_label": "Misleading",
            "binary_label": "False",
            "label_source": "Human annotation",
            "annotator_notes": "Evidence checked",
            "group_id": "health-claim-001",
            "claim_text": "Claim checked",
            "evidence_summary": "Official source contradicts the claim.",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "real_dataset.csv"
            queue_path = Path(temp_dir) / "queue.csv"
            with patch.object(app_module, "REAL_DATASET_PATH", dataset_path):
                with patch.object(app_module, "YOUTUBE_QUEUE_PATH", queue_path):
                    response = app_module.app.test_client().post("/api/labels", json=payload)
                    summary_response = app_module.app.test_client().get("/api/labels")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["saved"])
        self.assertEqual(summary_response.get_json()["row_count"], 1)

    def test_mismatch_label_requires_support_rationale(self):
        import app as app_module

        payload = {
            "video_id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "title": "Sinhala title",
            "description": "",
            "transcript": "Transcript text",
            "topic": "Health",
            "original_label": "False",
            "binary_label": "False",
            "label_source": "Human annotation",
            "annotator_notes": "Evidence checked",
            "group_id": "health-claim-001",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "real_dataset.csv"
            with patch.object(app_module, "REAL_DATASET_PATH", dataset_path):
                response = app_module.app.test_client().post("/api/labels", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("support_rationale is required", response.get_json()["error"])

    def test_youtube_search_endpoint_returns_unlabeled_candidates(self):
        import app as app_module
        from services import youtube_client

        def fake_search(query, max_results=10, page_token="", **_kwargs):
            self.assertIn("health", query.lower())
            return (
                [
                    youtube_client.YouTubeSearchCandidate(
                        video_id="abc123",
                        url="https://www.youtube.com/watch?v=abc123",
                        title="Already labeled",
                    ),
                    youtube_client.YouTubeSearchCandidate(
                        video_id="def456",
                        url="https://www.youtube.com/watch?v=def456",
                        title="New candidate",
                    ),
                ],
                {"next_page_token": "NEXT", "total_results": 2, "results_per_page": max_results},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "real_dataset.csv"
            with patch.object(app_module, "REAL_DATASET_PATH", dataset_path):
                app_module.ensure_real_dataset()
                app_module.app.test_client().post(
                    "/api/labels",
                    json={
                        "video_id": "abc123",
                        "url": "https://www.youtube.com/watch?v=abc123",
                        "title": "Sinhala title",
                        "description": "",
                        "transcript": "Transcript text",
                        "topic": "Health",
                        "original_label": "False",
                        "binary_label": "False",
                        "label_source": "Human annotation",
                        "annotator_notes": "Evidence checked",
                        "group_id": "health-claim-001",
                        "claim_text": "Claim checked",
                        "evidence_summary": "Official source contradicts the claim.",
                    },
                )
                with patch.object(app_module, "search_youtube_videos", side_effect=fake_search):
                    response = app_module.app.test_client().post(
                        "/api/youtube-search",
                        json={"topic": "Health", "query": "health sinhala", "max_results": 10},
                    )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["excluded_count"], 1)
        self.assertEqual([candidate["video_id"] for candidate in payload["candidates"]], ["def456"])

    def test_youtube_captioned_search_only_returns_sinhala_captioned_candidates(self):
        import app as app_module
        from services import youtube_client

        def fake_search(query, max_results=10, page_token="", **_kwargs):
            self.assertIn("health", query.lower())
            self.assertEqual(_kwargs["video_caption"], "closedCaption")
            return (
                [
                    youtube_client.YouTubeSearchCandidate(
                        video_id="captioned1",
                        url="https://www.youtube.com/watch?v=captioned1",
                        title="Captioned candidate",
                        description="Search description",
                    ),
                    youtube_client.YouTubeSearchCandidate(
                        video_id="nocaption2",
                        url="https://www.youtube.com/watch?v=nocaption2",
                        title="No Sinhala caption",
                    ),
                ],
                {"next_page_token": "NEXT", "total_results": 2, "results_per_page": max_results},
            )

        def fake_metadata(video_id, **kwargs):
            self.assertEqual(kwargs["transcript_languages"], ("si",))
            self.assertFalse(kwargs["allow_transcript_language_fallback"])
            if video_id == "captioned1":
                return youtube_client.YouTubeMetadata(
                    video_id=video_id,
                    title="API title",
                    description="API description",
                    transcript="Sinhala caption transcript text",
                    thumbnail_url="https://example.test/thumb.jpg",
                )
            return youtube_client.YouTubeMetadata(video_id=video_id, title="No Sinhala caption", transcript="")

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "real_dataset.csv"
            with patch.object(app_module, "REAL_DATASET_PATH", dataset_path):
                with patch.object(app_module, "search_youtube_videos", side_effect=fake_search):
                    with patch.object(app_module, "fetch_youtube_metadata", side_effect=fake_metadata):
                        response = app_module.app.test_client().post(
                            "/api/youtube-captioned-search",
                            json={"topic": "Health", "query": "health sinhala", "max_results": 10},
                        )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["skipped_no_sinhala_caption"], 1)
        self.assertEqual([candidate["video_id"] for candidate in payload["candidates"]], ["captioned1"])
        self.assertEqual(payload["candidates"][0]["transcript"], "Sinhala caption transcript text")
        self.assertEqual(payload["candidates"][0]["caption_language"], "si")

    def test_dataset_queue_build_next_and_status_flow(self):
        import app as app_module
        from services import youtube_client

        def fake_search(query, max_results=25, page_token="", **_kwargs):
            self.assertEqual(_kwargs["video_caption"], "closedCaption")
            return (
                [
                    youtube_client.YouTubeSearchCandidate(
                        video_id="captioned1",
                        url="https://www.youtube.com/watch?v=captioned1",
                        title="Breaking health claim",
                        description="No official source shown",
                        channel_title="Channel A",
                    ),
                    youtube_client.YouTubeSearchCandidate(
                        video_id="nocaption2",
                        url="https://www.youtube.com/watch?v=nocaption2",
                        title="No Sinhala caption",
                    ),
                ],
                {"next_page_token": "", "total_results": 2, "results_per_page": max_results},
            )

        def fake_metadata(video_id, **kwargs):
            self.assertEqual(kwargs["transcript_languages"], ("si",))
            self.assertFalse(kwargs["allow_transcript_language_fallback"])
            if video_id == "captioned1":
                return youtube_client.YouTubeMetadata(
                    video_id=video_id,
                    title="Breaking health claim",
                    description="No official source shown",
                    transcript="This viral claim says a medicine can cure disease without evidence.",
                    thumbnail_url="https://example.test/thumb.jpg",
                )
            return youtube_client.YouTubeMetadata(video_id=video_id, title="No Sinhala caption", transcript="")

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "real_dataset.csv"
            queue_path = Path(temp_dir) / "queue.csv"
            with patch.object(app_module, "REAL_DATASET_PATH", dataset_path):
                with patch.object(app_module, "YOUTUBE_QUEUE_PATH", queue_path):
                    with patch.object(app_module, "search_youtube_videos", side_effect=fake_search):
                        with patch.object(app_module, "fetch_youtube_metadata", side_effect=fake_metadata):
                            build_response = app_module.app.test_client().post(
                                "/api/dataset-queue/build",
                                json={"topics": ["Health"], "target_per_topic": 1},
                            )
                    next_response = app_module.app.test_client().get("/api/dataset-queue/next?topic=Health")
                    status_response = app_module.app.test_client().post(
                        "/api/dataset-queue/status",
                        json={"video_id": "captioned1", "queue_status": "skipped"},
                    )
                    label_summary = app_module.app.test_client().get("/api/labels")

        build_payload = build_response.get_json()
        next_payload = next_response.get_json()
        self.assertEqual(build_response.status_code, 200)
        self.assertEqual(build_payload["saved_total"], 1)
        self.assertEqual(next_response.status_code, 200)
        self.assertEqual(next_payload["candidate"]["video_id"], "captioned1")
        self.assertEqual(next_payload["candidate"]["queue_status"], "in_review")
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(label_summary.get_json()["row_count"], 0)

    def test_dataset_quality_reports_benchmark_balance(self):
        import app as app_module

        payloads = [
            {
                "video_id": "false123",
                "url": "https://www.youtube.com/watch?v=false123",
                "title": "Sinhala false title",
                "description": "",
                "transcript": "Transcript text",
                "topic": "Health",
                "original_label": "False",
                "binary_label": "False",
                "label_source": "Human annotation",
                "annotator_notes": "Evidence checked",
                "group_id": "health-claim-001",
                "claim_text": "Claim checked",
                "evidence_summary": "Official source contradicts the claim.",
            },
            {
                "video_id": "notfalse123",
                "url": "https://www.youtube.com/watch?v=notfalse123",
                "title": "Sinhala reliable title",
                "description": "",
                "transcript": "Transcript text",
                "topic": "Health",
                "original_label": "Reliable",
                "binary_label": "Not False",
                "label_source": "Human annotation",
                "annotator_notes": "Reliable official update.",
                "group_id": "health-claim-002",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "real_dataset.csv"
            queue_path = Path(temp_dir) / "queue.csv"
            with patch.object(app_module, "REAL_DATASET_PATH", dataset_path):
                with patch.object(app_module, "YOUTUBE_QUEUE_PATH", queue_path):
                    for payload in payloads:
                        app_module.app.test_client().post("/api/labels", json=payload)
                    response = app_module.app.test_client().get("/api/dataset-quality")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["label_counts"]["False"], 1)
        self.assertEqual(payload["label_counts"]["Not False"], 1)
        self.assertEqual(payload["group_count"], 2)
        self.assertFalse(payload["ready_for_first_benchmark"])


class PrepareDataTests(unittest.TestCase):
    def test_real_training_collapses_majority_duplicates_and_excludes_ties(self):
        from training.train_tfidf_real_calibrated import collapse_repeated_video_ids

        rows = [
            {"video_id": "majority", "label": "False", "title": "a", "transcript": "short"},
            {"video_id": "majority", "label": "False", "title": "a", "transcript": "longer evidence"},
            {"video_id": "majority", "label": "Not False", "title": "a", "transcript": "conflict"},
            {"video_id": "tie", "label": "False", "title": "b", "transcript": "one"},
            {"video_id": "tie", "label": "Not False", "title": "b", "transcript": "two"},
            {"video_id": "unique", "label": "Not False", "title": "c", "transcript": "aligned"},
        ]

        collapsed = collapse_repeated_video_ids(pd.DataFrame(rows))

        self.assertEqual(set(collapsed["video_id"]), {"majority", "unique"})
        self.assertEqual(collapsed.set_index("video_id").loc["majority", "label"], "False")
        self.assertEqual(collapsed.set_index("video_id").loc["majority", "transcript"], "longer evidence")

    def test_content_match_generator_covers_controlled_same_category_hard_negatives(self):
        from training.build_content_match_dataset import (
            HARD_NEGATIVE_CASES,
            HARD_NEGATIVE_CHANGE_TYPES,
            SCENARIO_FIELDS,
            build_transcript,
            generate_dataset,
            make_description,
            make_title,
            validate_hard_negative_cases,
        )

        validate_hard_negative_cases()
        rows, problems = generate_dataset(row_count=240, seed=17)
        generated = pd.DataFrame(rows)
        hard_rows = [row for row in rows if row["hard_negative_type"] != "none"]

        self.assertFalse(problems)
        self.assertEqual(sum(row["label"] == "0" for row in rows), 120)
        self.assertEqual(sum(row["label"] == "1" for row in rows), 120)
        self.assertEqual({row["hard_negative_type"] for row in hard_rows}, HARD_NEGATIVE_CHANGE_TYPES)
        self.assertTrue(all(row["label"] == "1" for row in hard_rows))
        self.assertTrue(all(case.metadata[0] == case.transcript[0] for case in HARD_NEGATIVE_CASES))
        for index, case in enumerate(HARD_NEGATIVE_CASES, start=1):
            metadata = " ".join(
                [
                    make_title(case.metadata, index, 1, "hard"),
                    make_description(case.metadata, index, 1, "hard"),
                ]
            )
            transcript = build_transcript(case.transcript, index, "match", 80)
            for field in case.changed_fields:
                field_index = SCENARIO_FIELDS.index(field)
                self.assertIn(case.metadata[field_index], metadata)
                self.assertIn(case.transcript[field_index], transcript)
        self.assertEqual(generated["group_id"].nunique(), 120)
        self.assertTrue(all(set(group["label"]) == {"0", "1"} for _, group in generated.groupby("group_id")))
        self.assertEqual(
            generated[generated["label"] == "0"]["topic"].value_counts().to_dict(),
            generated[generated["label"] == "1"]["topic"].value_counts().to_dict(),
        )

        main_topic_rows = [row for row in hard_rows if row["hard_negative_type"] == "main_topic"]
        self.assertTrue(any("mobile app" in row["title"] and "variables" in row["transcript"] for row in main_topic_rows))

    def test_real_schema_prepares_binary_training_label(self):
        df = pd.DataFrame(
            [
                {
                    "video_id": "abc123",
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "title": "Sinhala title",
                    "description": "",
                    "transcript": "Transcript text",
                    "topic": "Health",
                    "original_label": "Misleading",
                    "binary_label": "False",
                    "label_source": "Human annotation",
                    "annotator_notes": "Evidence reviewed",
                    "group_id": "claim-1",
                },
                {
                    "video_id": "def456",
                    "url": "https://www.youtube.com/watch?v=def456",
                    "title": "Another Sinhala title",
                    "description": "",
                    "transcript": "",
                    "topic": "Education",
                    "original_label": "Unverified",
                    "binary_label": "Not False",
                    "label_source": "Human annotation",
                    "annotator_notes": "Insufficient evidence",
                    "group_id": "claim-2",
                },
            ]
        )

        self.assertEqual(validate_schema(df), "real")
        prepared = prepare_text_dataframe(df, "real")

        self.assertEqual(prepared["label"].tolist(), ["False", "Not False"])
        self.assertEqual(prepared["original_label"].tolist(), ["Misleading", "Unverified"])
        self.assertIn("[TITLE]", prepared.loc[0, "text"])
        self.assertIn("[CONSISTENCY_FEATURES]", prepared.loc[0, "text"])
        self.assertEqual(prepared.loc[0, "consistency_label"], "Inconsistent")
        self.assertEqual(prepared.loc[1, "consistency_label"], "Consistent")

    def test_content_match_schema_maps_zero_one_labels(self):
        df = pd.DataFrame(
            [
                {
                    "video_id": "SYN000001",
                    "title": "Title matches",
                    "description": "Description matches",
                    "transcript": "Transcript supports the metadata",
                    "label": 0,
                },
                {
                    "video_id": "SYN000002",
                    "title": "Title about phones",
                    "description": "Phone review",
                    "transcript": "Transcript discusses cricket news",
                    "label": 1,
                },
            ]
        )

        self.assertEqual(validate_schema(df), "content_match")
        prepared = prepare_text_dataframe(df, "content_match")

        self.assertEqual(prepared["label"].tolist(), ["Not False", "False"])
        self.assertEqual(prepared["binary_label"].tolist(), ["Not False", "False"])
        self.assertEqual(prepared["mismatch_type"].tolist(), ["consistent", "title_transcript_mismatch"])
        self.assertTrue((prepared["group_id"] == ["content-match-SYN000001", "content-match-SYN000002"]).all())

    def test_synthetic_consistent_false_is_not_false_for_consistency_task(self):
        df = pd.DataFrame(
            [
                {
                    "video_id": "syn1",
                    "group_id": "g1",
                    "title": "Title",
                    "description": "Description",
                    "transcript": "Transcript",
                    "thumbnail_text": "",
                    "thumbnail_prompt": "",
                    "label": "False",
                    "topic": "Health",
                    "consistency_pattern": "CONSISTENT_FALSE",
                },
                {
                    "video_id": "syn2",
                    "group_id": "g2",
                    "title": "Breaking claim",
                    "description": "Description",
                    "transcript": "Different transcript",
                    "thumbnail_text": "",
                    "thumbnail_prompt": "",
                    "label": "Misleading",
                    "topic": "Health",
                    "consistency_pattern": "TITLE_CLICKBAIT",
                },
            ]
        )

        prepared = prepare_text_dataframe(df, "synthetic")

        self.assertEqual(prepared["label"].tolist(), ["Not False", "False"])
        self.assertEqual(prepared["mismatch_type"].tolist(), ["consistent", "title_transcript_mismatch"])

    def test_same_topic_hard_negatives_keep_topic_and_source_ids(self):
        df = pd.DataFrame(
            [
                {
                    "video_id": "a1",
                    "url": "https://www.youtube.com/watch?v=a1",
                    "group_id": "g1",
                    "title": "Health title one",
                    "description": "Health description one",
                    "transcript": "Health transcript one",
                    "label": "Not False",
                    "binary_label": "Not False",
                    "original_label": "Reliable",
                    "topic": "Health",
                    "label_source": "Human annotation",
                    "annotator_notes": "Aligned",
                    "consistency_label": "Consistent",
                    "mismatch_type": "consistent",
                    "support_rationale": "Aligned",
                    "generated_negative": "false",
                    "source_title_video_id": "a1",
                    "source_transcript_video_id": "a1",
                },
                {
                    "video_id": "a2",
                    "url": "https://www.youtube.com/watch?v=a2",
                    "group_id": "g2",
                    "title": "Health title two",
                    "description": "Health description two",
                    "transcript": "Health transcript two",
                    "label": "Not False",
                    "binary_label": "Not False",
                    "original_label": "Reliable",
                    "topic": "Health",
                    "label_source": "Human annotation",
                    "annotator_notes": "Aligned",
                    "consistency_label": "Consistent",
                    "mismatch_type": "consistent",
                    "support_rationale": "Aligned",
                    "generated_negative": "false",
                    "source_title_video_id": "a2",
                    "source_transcript_video_id": "a2",
                },
            ]
        )

        generated = add_same_topic_hard_negatives(df)
        negatives = generated[generated["generated_negative"] == "true"]

        self.assertFalse(negatives.empty)
        self.assertTrue((negatives["topic"] == "Health").all())
        self.assertTrue((negatives["label"] == "False").all())
        self.assertTrue((negatives["source_title_video_id"] != negatives["source_transcript_video_id"]).all())
        self.assertTrue((negatives["mismatch_type"] == "same_topic_swap").all())


class SyntheticConsistencyLabelingPoolTests(unittest.TestCase):
    def test_human_friendly_similarity_pool_is_balanced_and_review_ready(self):
        from training.build_human_friendly_similarity_pool import OUTPUT_COLUMNS, generate_rows

        rows = generate_rows(row_count=180, seed=21)
        df = pd.DataFrame(rows)

        self.assertEqual(len(rows), 180)
        self.assertTrue(set(OUTPUT_COLUMNS).issubset(df.columns))
        self.assertEqual(df["seed_binary_label"].value_counts().to_dict(), {"False": 90, "Not False": 90})
        self.assertEqual(set(df["label_status"]), {"new"})
        self.assertEqual(df["similarity_band"].nunique(), 6)
        self.assertTrue((df[df["seed_binary_label"] == "False"]["generated_negative"] == "true").all())
        self.assertTrue((df[df["seed_binary_label"] == "Not False"]["generated_negative"] == "false").all())
        self.assertEqual(len(df.drop_duplicates(subset=["title", "description", "transcript"])), len(df))
        for column in [
            "title_transcript_similarity",
            "description_transcript_similarity",
            "title_description_similarity",
        ]:
            values = df[column].astype(float)
            self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_synthetic_labeling_pool_has_balanced_review_cases(self):
        from training.build_synthetic_consistency_labeling_pool import OUTPUT_COLUMNS, generate_rows

        rows = generate_rows(row_count=100, seed=7)
        df = pd.DataFrame(rows)

        self.assertEqual(len(rows), 100)
        self.assertTrue(set(OUTPUT_COLUMNS).issubset(df.columns))
        self.assertEqual(df["seed_binary_label"].value_counts().to_dict(), {"Not False": 50, "False": 50})
        for topic, topic_df in df.groupby("topic"):
            self.assertIn("False", set(topic_df["seed_binary_label"]), topic)
            self.assertIn("Not False", set(topic_df["seed_binary_label"]), topic)
        self.assertTrue((df[df["seed_binary_label"] == "Not False"]["generated_negative"] == "false").all())
        self.assertTrue((df[df["seed_binary_label"] == "False"]["generated_negative"] == "true").all())

    def test_synthetic_same_topic_swaps_keep_source_ids_separate(self):
        from training.build_synthetic_consistency_labeling_pool import generate_rows

        df = pd.DataFrame(generate_rows(row_count=120, seed=11))
        swaps = df[df["seed_mismatch_type"] == "same_topic_swap"]

        self.assertFalse(swaps.empty)
        self.assertTrue((swaps["source_title_video_id"] != swaps["source_transcript_video_id"]).all())
        self.assertTrue(swaps["seed_support_rationale"].map(is_meaningful_text).all())
        for column in [
            "title_transcript_similarity",
            "description_transcript_similarity",
            "title_description_similarity",
        ]:
            values = swaps[column].astype(float)
            self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_synthetic_labeling_endpoint_returns_next_candidate_and_marks_saved(self):
        import app as app_module
        from training.build_synthetic_consistency_labeling_pool import generate_rows, write_rows

        with tempfile.TemporaryDirectory() as temp_dir:
            synthetic_path = Path(temp_dir) / "synthetic_pool.csv"
            dataset_path = Path(temp_dir) / "real_dataset.csv"
            queue_path = Path(temp_dir) / "queue.csv"
            write_rows(generate_rows(row_count=40, seed=13), synthetic_path)
            with patch.object(app_module, "SYNTHETIC_LABELING_POOL_PATH", synthetic_path):
                with patch.object(app_module, "REAL_DATASET_PATH", dataset_path):
                    with patch.object(app_module, "YOUTUBE_QUEUE_PATH", queue_path):
                        client = app_module.app.test_client()
                        response = client.get("/api/synthetic-labeling/next?topic=Health")
                        self.assertEqual(response.status_code, 200)
                        candidate = response.get_json()["candidate"]
                        self.assertEqual(candidate["topic"], "Health")
                        self.assertEqual(candidate["queue_source"], "synthetic_labeling_pool")
                        self.assertFalse(any(key.startswith("seed_") for key in candidate))

                        save_response = client.post(
                            "/api/labels",
                            json={
                                "video_id": candidate["video_id"],
                                "url": candidate["url"],
                                "title": candidate["title"],
                                "description": candidate["description"],
                                "transcript": candidate["transcript"],
                                "topic": candidate["topic"],
                                "original_label": "Reliable",
                                "binary_label": "Not False",
                                "consistency_label": "Consistent",
                                "mismatch_type": "consistent",
                                "support_rationale": "Human review says metadata is aligned enough.",
                                "label_source": "Human-reviewed synthetic consistency annotation",
                                "annotator_notes": "Aligned enough",
                                "group_id": candidate["suggested_group_id"],
                                "queue_source": "synthetic_labeling_pool",
                                "caption_source": "synthetic_transcript",
                                "source_title_video_id": candidate["source_title_video_id"],
                                "source_transcript_video_id": candidate["source_transcript_video_id"],
                            },
                        )

                        self.assertEqual(save_response.status_code, 200)
                        rows = app_module.synthetic_labeling_rows()
                        saved = [row for row in rows if row["video_id"] == candidate["video_id"]][0]
                        self.assertEqual(saved["label_status"], "saved")
                        self.assertEqual(saved["reviewed_binary_label"], "Not False")


class SinhalaAsrManifestTests(unittest.TestCase):
    def test_manifest_validation_catches_bad_rows(self):
        dataframe = pd.DataFrame(
            [
                {
                    "audio_path": "missing.wav",
                    "transcript": "",
                    "split": "bad",
                    "source": "manual",
                    "duration_seconds": 99,
                },
                {
                    "audio_path": "missing.wav",
                    "transcript": "valid text",
                    "split": "train",
                    "source": "manual",
                    "duration_seconds": "not-a-number",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            errors = validate_asr_manifest(
                dataframe,
                base_dir=Path(temp_dir),
                max_duration_seconds=30,
            )

        joined = " ".join(errors)
        self.assertIn("audio file does not exist", joined)
        self.assertIn("transcript is empty", joined)
        self.assertIn("split must be one of", joined)
        self.assertIn("exceeds max", joined)
        self.assertIn("duration_seconds must be numeric", joined)
        self.assertIn("Duplicate audio file", joined)


class AsrReferenceQualityTests(unittest.TestCase):
    def test_asr_manifest_row_cleans_timestamps_and_bracket_noise(self):
        import app as app_module

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"fake")
            transcript = "0:00 [Music] " + " ".join(["සිංහල"] * 30) + " 1:23 [සංගීතය]"

            row = app_module.build_asr_manifest_row(
                {
                    "audio_path": str(audio_path),
                    "transcript": transcript,
                    "split": "test",
                    "source": "test",
                    "duration_seconds": "12",
                }
            )

        self.assertNotIn("0:00", row["transcript"])
        self.assertNotIn("[Music]", row["transcript"])
        self.assertNotIn("[සංගීතය]", row["transcript"])
        self.assertIn("සිංහල", row["transcript"])

    def test_asr_manifest_row_rejects_non_sinhala_reference(self):
        import app as app_module

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"fake")
            with self.assertRaises(ValueError) as context:
                app_module.build_asr_manifest_row(
                    {
                        "audio_path": str(audio_path),
                        "transcript": "0:00 foreign music thank you this is not a sinhala reference",
                        "split": "test",
                        "source": "test",
                        "duration_seconds": "12",
                    }
                )

        self.assertIn("Clean Sinhala reference transcript required", str(context.exception))


class SinhalaCaptionCollectorTests(unittest.TestCase):
    def test_strict_sinhala_caption_selection_rejects_other_languages(self):
        from services.youtube_client import preferred_caption_track

        selected = preferred_caption_track(
            [{"languageCode": "en", "baseUrl": "https://example.test/caption"}],
            languages=("si",),
            allow_language_fallback=False,
        )

        self.assertEqual(selected, {})

    def test_strict_sinhala_caption_selection_accepts_regional_and_named_tracks(self):
        from services.youtube_client import preferred_caption_track

        regional = preferred_caption_track(
            [{"languageCode": "si-LK", "baseUrl": "https://example.test/caption"}],
            languages=("si",),
            allow_language_fallback=False,
        )
        named = preferred_caption_track(
            [
                {
                    "languageCode": "und",
                    "name": {"simpleText": "Sinhala auto-generated"},
                    "baseUrl": "https://example.test/caption",
                }
            ],
            languages=("si",),
            allow_language_fallback=False,
        )

        self.assertEqual(regional["languageCode"], "si-LK")
        self.assertEqual(named["languageCode"], "und")

    def test_collector_writes_only_videos_with_sinhala_captions(self):
        import collection.collect_sinhala_captioned_youtube as collector
        from services.youtube_client import YouTubeMetadata, YouTubeSearchCandidate

        candidates = [
            YouTubeSearchCandidate(
                video_id="captioned123",
                url="https://www.youtube.com/watch?v=captioned123",
                title="Captioned video",
            ),
            YouTubeSearchCandidate(
                video_id="nocaption456",
                url="https://www.youtube.com/watch?v=nocaption456",
                title="No caption video",
            ),
        ]
        metadata = [
            YouTubeMetadata(
                video_id="captioned123",
                title="Captioned video",
                description="Description",
                transcript="Sinhala transcript text",
                source="youtube_api",
            ),
            YouTubeMetadata(
                video_id="nocaption456",
                title="No caption video",
                description="Description",
                transcript="",
                source="youtube_api",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "captioned.csv"
            with patch.object(collector, "TOPIC_SEARCH_QUERIES", {"Health": "health query"}):
                with patch.object(collector, "search_youtube_videos", return_value=(candidates, {"next_page_token": ""})):
                    with patch.object(collector, "fetch_youtube_metadata", side_effect=metadata):
                        summary = collector.collect_captioned_candidates(
                            topics=["Health"],
                            output_path=output_path,
                            target_per_topic=2,
                            search_page_size=10,
                            max_pages_per_topic=1,
                            overwrite=True,
                            api_key="fake",
                        )
            rows = pd.read_csv(output_path)

        self.assertEqual(summary["saved_total"], 1)
        self.assertEqual(summary["skipped_no_sinhala_caption"], 1)
        self.assertEqual(rows["video_id"].tolist(), ["captioned123"])
        self.assertEqual(rows["caption_language"].tolist(), ["si"])


if __name__ == "__main__":
    unittest.main()
