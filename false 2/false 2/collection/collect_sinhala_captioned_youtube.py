"""Collect Sinhala YouTube videos that already have public Sinhala captions.

This creates a transcript-rich candidate CSV for human false-content labeling.
It does not assign False / Not False labels automatically.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app import TOPIC_SEARCH_QUERIES, TOPICS  # noqa: E402
from services.youtube_client import fetch_youtube_metadata, search_youtube_videos  # noqa: E402
from utils.data_utils import clean_text, is_meaningful_text  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "captioned_sinhala_youtube_candidates.csv"
OUTPUT_COLUMNS = [
    "video_id",
    "url",
    "title",
    "description",
    "transcript",
    "topic",
    "caption_language",
    "caption_source",
    "channel_title",
    "published_at",
    "thumbnail_url",
    "collection_query",
    "group_id",
    "original_label",
    "binary_label",
    "label_source",
    "annotator_notes",
]


def ensure_output(path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()


def existing_video_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            clean_text(row.get("video_id", ""))
            for row in csv.DictReader(handle)
            if clean_text(row.get("video_id", ""))
        }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writerows(rows)


def parse_topics(value: str) -> list[str]:
    if clean_text(value).lower() == "all":
        return list(TOPICS)
    topics = [clean_text(topic) for topic in value.split(",") if clean_text(topic)]
    invalid = [topic for topic in topics if topic not in TOPICS]
    if invalid:
        raise ValueError(f"Unknown topics: {', '.join(invalid)}. Expected: {', '.join(TOPICS)}")
    return topics


def candidate_row(candidate, metadata, topic: str, query: str) -> dict[str, str]:
    return {
        "video_id": metadata.video_id,
        "url": f"https://www.youtube.com/watch?v={metadata.video_id}",
        "title": metadata.title or candidate.title,
        "description": metadata.description or candidate.description,
        "transcript": metadata.transcript,
        "topic": topic,
        "caption_language": "si",
        "caption_source": "youtube_public_caption",
        "channel_title": candidate.channel_title,
        "published_at": candidate.published_at,
        "thumbnail_url": metadata.thumbnail_url or candidate.thumbnail_url,
        "collection_query": query,
        "group_id": f"{topic.lower()}-{metadata.video_id}",
        "original_label": "",
        "binary_label": "",
        "label_source": "",
        "annotator_notes": "",
    }


def collect_captioned_candidates(
    topics: list[str],
    output_path: Path,
    target_per_topic: int,
    search_page_size: int,
    max_pages_per_topic: int,
    overwrite: bool = False,
    api_key: str | None = None,
) -> dict:
    ensure_output(output_path, overwrite=overwrite)
    seen_video_ids = existing_video_ids(output_path)
    summary = {
        "output": str(output_path),
        "topics": {},
        "saved_total": 0,
        "skipped_existing": 0,
        "skipped_no_sinhala_caption": 0,
        "errors": [],
    }

    for topic in topics:
        query = TOPIC_SEARCH_QUERIES.get(topic, f"Sinhala {topic}")
        topic_saved = 0
        page_token = ""
        pages_checked = 0
        while topic_saved < target_per_topic and pages_checked < max_pages_per_topic:
            pages_checked += 1
            try:
                candidates, meta = search_youtube_videos(
                    query=query,
                    api_key=api_key,
                    max_results=search_page_size,
                    page_token=page_token,
                )
            except (RuntimeError, requests.RequestException) as exc:
                summary["errors"].append(f"{topic}: search failed: {exc}")
                break

            rows_to_write: list[dict[str, str]] = []
            for candidate in candidates:
                if candidate.video_id in seen_video_ids:
                    summary["skipped_existing"] += 1
                    continue
                try:
                    metadata = fetch_youtube_metadata(
                        candidate.video_id,
                        api_key=api_key,
                        use_audio_fallback=False,
                        transcript_languages=("si",),
                        allow_transcript_language_fallback=False,
                    )
                except requests.RequestException as exc:
                    summary["errors"].append(f"{candidate.video_id}: metadata failed: {exc}")
                    continue

                if not is_meaningful_text(metadata.transcript):
                    summary["skipped_no_sinhala_caption"] += 1
                    continue

                rows_to_write.append(candidate_row(candidate, metadata, topic, query))
                seen_video_ids.add(candidate.video_id)
                topic_saved += 1
                if topic_saved >= target_per_topic:
                    break

            write_rows(output_path, rows_to_write)
            page_token = clean_text((meta or {}).get("next_page_token", ""))
            if not page_token:
                break

        summary["topics"][topic] = {"saved": topic_saved, "pages_checked": pages_checked}
        summary["saved_total"] += topic_saved

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Sinhala-captioned YouTube candidates.")
    parser.add_argument("--topics", default="all", help="all or comma-separated topic names.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-per-topic", type=int, default=20)
    parser.add_argument("--search-page-size", type=int, default=25)
    parser.add_argument("--max-pages-per-topic", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    summary = collect_captioned_candidates(
        topics=parse_topics(args.topics),
        output_path=Path(args.output),
        target_per_topic=max(1, args.target_per_topic),
        search_page_size=max(1, min(args.search_page_size, 50)),
        max_pages_per_topic=max(1, args.max_pages_per_topic),
        overwrite=args.overwrite,
        api_key=args.api_key or None,
    )

    print("Captioned Sinhala YouTube collection summary")
    print(f"Output: {summary['output']}")
    print(f"Saved total: {summary['saved_total']}")
    print(f"Skipped existing: {summary['skipped_existing']}")
    print(f"Skipped no Sinhala caption: {summary['skipped_no_sinhala_caption']}")
    for topic, topic_summary in summary["topics"].items():
        print(f"- {topic}: saved {topic_summary['saved']} from {topic_summary['pages_checked']} search page(s)")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"][:20]:
            print(f"- {error}")
    return 0 if summary["saved_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
