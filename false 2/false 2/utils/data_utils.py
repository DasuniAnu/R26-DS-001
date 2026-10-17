"""Dataset loading, cleaning, and image helper functions."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image, ImageFile

from config.default_config import (
    BINARY_FALSE_LABEL,
    BINARY_LABEL_ORDER,
    BINARY_NOT_FALSE_LABEL,
    PREFERRED_LABEL_ORDER,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger(__name__)


COLUMN_ALIASES: dict[str, list[str]] = {
    "title": ["title", "video_title", "title_si", "video_title_si"],
    "description": [
        "description",
        "video_description",
        "description_si",
        "video_description_si",
    ],
    "transcript": [
        "transcript",
        "transcript_excerpt",
        "audio_transcript",
        "transcript_si",
        "transcript_excerpt_si",
        "audio_transcript_si",
        "asr_noisy_transcript_si",
    ],
    "thumbnail": [
        "thumbnail_path",
        "thumbnail_url",
        "thumbnail",
        "thumbnail_path_si",
        "thumbnail_url_si",
    ],
    "label": [
        "label",
        "final_label",
        "video_label",
        "classification",
        "claim_label",
    ],
}


LABEL_ALIASES: dict[str, str] = {
    "false": "False",
    "fake": "False",
    "misinformation": "False",
    "not false": "Not False",
    "not_false": "Not False",
    "not-false": "Not False",
    "non false": "Not False",
    "non_false": "Not False",
    "non-false": "Not False",
    "misleading": "Misleading",
    "partly false": "Misleading",
    "partially false": "Misleading",
    "reliable": "Reliable",
    "true": "Reliable",
    "credible": "Reliable",
    "verified": "Reliable",
    "unverified": "Unverified",
    "unknown": "Unverified",
    "uncertain": "Unverified",
    "unverifiable": "Unverified",
}

BINARY_LABEL_MAP: dict[str, str] = {
    "False": BINARY_FALSE_LABEL,
    "Misleading": BINARY_FALSE_LABEL,
    "Reliable": BINARY_NOT_FALSE_LABEL,
    "Unverified": BINARY_NOT_FALSE_LABEL,
    BINARY_FALSE_LABEL: BINARY_FALSE_LABEL,
    BINARY_NOT_FALSE_LABEL: BINARY_NOT_FALSE_LABEL,
}

MEANINGFUL_TEXT_PATTERN = re.compile(r"[A-Za-z0-9\u0D80-\u0DFF]")


def normalize_column_name(name: str) -> str:
    """Normalize column names so aliases like title-si and title_si match."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return normalized.strip("_")


def resolve_input_path(input_path: str | Path) -> Path:
    """Resolve an Excel path and handle common downloaded-file name variants."""

    path = Path(input_path)
    if path.exists():
        return path

    candidate = Path(str(path).replace("(1)", ""))
    if candidate.exists():
        LOGGER.warning("Input %s was not found; using %s instead.", path, candidate)
        return candidate

    raise FileNotFoundError(f"Could not find dataset file: {input_path}")


def detect_column(columns: Iterable[str], canonical_name: str, required: bool = True) -> str | None:
    """Find a dataset column using aliases and light normalization."""

    normalized_to_original = {normalize_column_name(col): col for col in columns}
    aliases = COLUMN_ALIASES[canonical_name]

    for alias in aliases:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized_to_original:
            return normalized_to_original[normalized_alias]

    # Helpful fallback for Sinhala-specific columns such as title_si.
    for normalized, original in normalized_to_original.items():
        without_suffix = re.sub(r"(_si|_sin)$", "", normalized)
        if without_suffix in {normalize_column_name(alias) for alias in aliases}:
            return original

    if required:
        raise ValueError(
            f"Could not detect required '{canonical_name}' column. "
            f"Available columns: {list(columns)}"
        )
    return None


def clean_text(value: object) -> str:
    """Convert missing values to empty strings while preserving Sinhala Unicode."""

    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text.strip()


def is_meaningful_text(value: object) -> bool:
    """Return true when text contains real Sinhala, Latin, or numeric content."""

    text = clean_text(value)
    if not text:
        return False
    placeholders = {"...", "…", ".", "-", "--", "---", "n/a", "na", "none", "null"}
    if text.lower() in placeholders:
        return False
    return MEANINGFUL_TEXT_PATTERN.search(text) is not None


def normalize_label(value: object) -> str:
    """Normalize label spellings while keeping unknown labels readable."""

    text = clean_text(value)
    key = text.lower().replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key).strip()
    return LABEL_ALIASES.get(key, text)


def normalize_binary_label(value: object) -> str:
    """Map supported Sinhala false-content labels into False / Not False."""

    label = normalize_label(value)
    if label in BINARY_LABEL_MAP:
        return BINARY_LABEL_MAP[label]
    raise ValueError(
        f"Unsupported label for binary false-content detection: {value!r}. "
        f"Expected one of: {sorted(BINARY_LABEL_MAP)}"
    )


def build_label_mapping(labels: Iterable[str]) -> dict[str, int]:
    """Build a stable label-to-id mapping, prioritizing the required class order."""

    unique_labels = {normalize_label(label) for label in labels if clean_text(label)}
    ordered = [label for label in PREFERRED_LABEL_ORDER if label in unique_labels]
    ordered.extend(sorted(unique_labels.difference(ordered)))
    return {label: idx for idx, label in enumerate(ordered)}


def build_binary_label_mapping(labels: Iterable[str]) -> dict[str, int]:
    """Build the stable two-label mapping used by real-world detection."""

    unique_labels = {normalize_binary_label(label) for label in labels if clean_text(label)}
    ordered = [label for label in BINARY_LABEL_ORDER if label in unique_labels]
    ordered.extend(sorted(unique_labels.difference(ordered)))
    return {label: idx for idx, label in enumerate(ordered)}


def combine_text(title: object, description: object, transcript: object) -> str:
    """Join the three textual inputs into one encoder input."""

    parts = [clean_text(title), clean_text(description), clean_text(transcript)]
    return " [SEP] ".join(part for part in parts if part)


def format_text_input(title: object, description: object, transcript: object) -> str:
    """Build the exact text-only model input without changing Sinhala Unicode."""

    return (
        f"[TITLE]\n{clean_text(title)}\n\n"
        f"[DESCRIPTION]\n{clean_text(description)}\n\n"
        f"[TRANSCRIPT]\n{clean_text(transcript)}"
    ).strip()


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def blank_image(image_size: int = 224) -> Image.Image:
    """Return a neutral RGB fallback image."""

    return Image.new("RGB", (image_size, image_size), color=(255, 255, 255))


def load_thumbnail(
    thumbnail: object,
    image_size: int = 224,
    base_dir: Path | None = None,
    timeout: int = 10,
) -> Image.Image:
    """Load a thumbnail from path or URL; fall back to a blank image on failure."""

    value = clean_text(thumbnail)
    if not value:
        return blank_image(image_size)

    try:
        if is_url(value):
            response = requests.get(value, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        path = Path(value)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if path.exists():
            return Image.open(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - fallback is intentional for bad thumbnails.
        LOGGER.warning("Failed to load thumbnail '%s': %s", value, exc)

    return blank_image(image_size)
