"""Shared consistency preprocessing and feature helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.data_utils import clean_text


CONSISTENT_LABEL = "Consistent"
INCONSISTENT_LABEL = "Inconsistent"
MISMATCH_TYPES = {
    "consistent",
    "title_transcript_mismatch",
    "description_transcript_mismatch",
    "clickbait_overclaim",
    "same_topic_swap",
    "weak_support",
}
CLICKBAIT_PATTERNS = {
    "TITLE_CLICKBAIT",
    "THUMBNAIL_CLICKBAIT",
    "TITLE_THUMBNAIL_CLICKBAIT",
}

ENGLISH_TO_SINHALA_LEXICON = {
    "award ceremony": "සම්මාන උළෙල",
    "breaking news": "හදිසි පුවත්",
    "claim": "හිමිකම් කියන කරුණ",
    "confirmed": "තහවුරු කළ",
    "crypto": "ක්‍රිප්ටෝ",
    "cure": "සුව කිරීම",
    "doctor": "වෛද්‍යවරයා",
    "education": "අධ්‍යාපනය",
    "election": "මැතිවරණය",
    "fact check": "කරුණු පරීක්ෂාව",
    "fake": "ව්‍යාජ",
    "finance": "මුදල්",
    "government": "රජය",
    "health": "සෞඛ්‍ය",
    "investment": "ආයෝජනය",
    "loan": "ණය",
    "medical": "වෛද්‍ය",
    "medicine": "ඖෂධ",
    "minister": "අමාත්‍යවරයා",
    "ministry": "අමාත්‍යාංශය",
    "misleading": "වැරදි අවබෝධයක් ඇති කරන",
    "news": "පුවත්",
    "official": "නිල",
    "politics": "දේශපාලනය",
    "rumour": "කටකතාව",
    "rumor": "කටකතාව",
    "source": "මූලාශ්‍රය",
    "technology": "තාක්ෂණය",
    "transcript": "පිටපත",
    "update": "යාවත්කාලීන කිරීම",
    "viral": "වයිරල්",
    "warning": "අනතුරු ඇඟවීම",
    "weather": "කාලගුණය",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9\u0D80-\u0DFF]+", flags=re.UNICODE)


@dataclass(frozen=True)
class ConsistencyFeatures:
    """Pairwise support signals between metadata and transcript."""

    title_transcript_similarity: float
    description_transcript_similarity: float
    title_description_similarity: float

    def as_dict(self) -> dict[str, float]:
        return {
            "title_transcript_similarity": self.title_transcript_similarity,
            "description_transcript_similarity": self.description_transcript_similarity,
            "title_description_similarity": self.title_description_similarity,
        }


def normalize_for_consistency(value: object) -> str:
    """Clean text and translate common English words/phrases to Sinhala."""

    text = clean_text(value).lower()
    if not text:
        return ""
    for english in sorted(ENGLISH_TO_SINHALA_LEXICON, key=len, reverse=True):
        sinhala = ENGLISH_TO_SINHALA_LEXICON[english]
        text = re.sub(rf"\b{re.escape(english)}\b", f" {sinhala} ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s\u0D80-\u0DFF.:%/-]+", " ", text, flags=re.UNICODE)
    return clean_text(text)


def consistency_token_set(value: object) -> set[str]:
    text = normalize_for_consistency(value)
    return {token for token in TOKEN_RE.findall(text) if len(token) > 1}


def consistency_similarity(left: object, right: object) -> float:
    left_tokens = consistency_token_set(left)
    right_tokens = consistency_token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 3)


def compute_consistency_features(title: object, description: object, transcript: object) -> ConsistencyFeatures:
    return ConsistencyFeatures(
        title_transcript_similarity=consistency_similarity(title, transcript),
        description_transcript_similarity=consistency_similarity(description, transcript),
        title_description_similarity=consistency_similarity(title, description),
    )


def format_consistency_text_input(title: object, description: object, transcript: object) -> str:
    """Build the model input with normalized text and explicit consistency features."""

    normalized_title = normalize_for_consistency(title)
    normalized_description = normalize_for_consistency(description)
    normalized_transcript = normalize_for_consistency(transcript)
    features = compute_consistency_features(normalized_title, normalized_description, normalized_transcript)
    return (
        f"[TITLE]\n{normalized_title}\n\n"
        f"[DESCRIPTION]\n{normalized_description}\n\n"
        f"[TRANSCRIPT]\n{normalized_transcript}\n\n"
        "[CONSISTENCY_FEATURES]\n"
        f"title_transcript_similarity={features.title_transcript_similarity:.3f}\n"
        f"description_transcript_similarity={features.description_transcript_similarity:.3f}\n"
        f"title_description_similarity={features.title_description_similarity:.3f}"
    ).strip()


def normalize_consistency_label(value: object, binary_label: object = "") -> str:
    text = clean_text(value).lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"consistent", "supported", "aligned", "not false"}:
        return CONSISTENT_LABEL
    if text in {"inconsistent", "mismatch", "unsupported", "false", "misleading"}:
        return INCONSISTENT_LABEL
    return INCONSISTENT_LABEL if clean_text(binary_label) == "False" else CONSISTENT_LABEL


def normalize_mismatch_type(value: object, consistency_label: str = CONSISTENT_LABEL) -> str:
    mismatch_type = clean_text(value).lower().replace("-", "_").replace(" ", "_")
    if mismatch_type in MISMATCH_TYPES:
        return mismatch_type
    return "consistent" if consistency_label == CONSISTENT_LABEL else "weak_support"


def consistency_label_from_synthetic_pattern(pattern: object) -> str:
    normalized = clean_text(pattern).upper()
    return INCONSISTENT_LABEL if normalized in CLICKBAIT_PATTERNS else CONSISTENT_LABEL


def mismatch_type_from_synthetic_pattern(pattern: object) -> str:
    normalized = clean_text(pattern).upper()
    if normalized == "TITLE_CLICKBAIT":
        return "title_transcript_mismatch"
    if normalized in {"THUMBNAIL_CLICKBAIT", "TITLE_THUMBNAIL_CLICKBAIT"}:
        return "clickbait_overclaim"
    if normalized == "UNVERIFIED":
        return "weak_support"
    return "consistent"
