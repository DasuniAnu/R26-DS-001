"""Reusable prediction service for the Flask app."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import joblib
import torch

from config.default_config import (
    BINARY_FALSE_LABEL,
    BINARY_LABEL_ORDER,
    BINARY_NOT_FALSE_LABEL,
    FALSE_DETECTION_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    SYNTHETIC_MODEL_FALSE_THRESHOLD,
    SYNTHETIC_MODEL_HARD_NEGATIVE_THRESHOLD,
    DataConfig,
    PROJECT_ROOT,
)
from models.text_classifier import load_text_model_for_inference
from utils.consistency import (
    compute_consistency_features,
    consistency_token_set,
    format_consistency_text_input,
)
from utils.data_utils import (
    clean_text,
    is_meaningful_text,
    load_thumbnail,
    normalize_binary_label,
)


SKLEARN_TEXT_ARTIFACT = PROJECT_ROOT / "outputs" / "model" / "sklearn_text_classifier.joblib"
METADATA_COVERAGE_STOPWORDS = {
    "අද",
    "අලුත්",
    "අලුත්ම",
    "අතර",
    "අනුව",
    "අපි",
    "එක",
    "එකක්",
    "කරන",
    "කරනවා",
    "කිරීමක්",
    "කෙටි",
    "ගැන",
    "ගමු",
    "දැනගන්න",
    "දැන්",
    "නව",
    "නම්",
    "නිසා",
    "පැහැදිලි",
    "බලමු",
    "මෙහි",
    "මෙතැන",
    "මේ",
    "මේක",
    "ලෙස",
    "සහ",
    "සඳහා",
    "සම්බන්ධ",
    "සරල",
    "සම්පූර්ණ",
    "විදිහට",
    "විස්තර",
    "විශේෂ",
    "වෙනවා",
    "වෙනස්",
    "වෙලාවට",
    "headline",
    "latest",
    "live",
    "news",
    "official",
    "online",
    "point",
    "practical",
    "public",
    "ref",
    "result",
    "review",
    "short",
    "sinhala",
    "source",
    "summary",
    "title",
    "update",
    "video",
}


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction response."""

    label: str
    confidence: float
    model_type: str
    class_probabilities: dict[str, float]
    warnings: tuple[str, ...] = ()
    evidence_signals: tuple[dict[str, str], ...] = ()
    consistency_features: dict[str, float] | None = None
    review_recommendation: str = "review_optional"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "predicted_label": self.label,
            "confidence": self.confidence,
            "model_type": self.model_type,
            "class_probabilities": self.class_probabilities,
            "warnings": list(self.warnings),
            "evidence_signals": list(self.evidence_signals),
            "consistency_features": self.consistency_features or {},
            "review_recommendation": self.review_recommendation,
        }


class BasePredictor:
    """Common predictor interface."""

    model_type: str

    def predict(
        self,
        title: str,
        description: str = "",
        transcript: str = "",
        thumbnail: str = "",
    ) -> PredictionResult:
        raise NotImplementedError


class TextOnlyPredictor(BasePredictor):
    """Predict with the saved text-only model."""

    model_type = "text"

    def __init__(
        self,
        model_dir: str | Path = PROJECT_ROOT / "outputs" / "model",
        max_length: int = 256,
        local_files_only: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, self.id_to_label = load_text_model_for_inference(
            self.model_dir,
            self.device,
            local_files_only=local_files_only,
        )

    def predict(
        self,
        title: str,
        description: str = "",
        transcript: str = "",
        thumbnail: str = "",
    ) -> PredictionResult:
        text = format_consistency_text_input(title, description, transcript)
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

        with torch.no_grad():
            logits = self.model(**inputs)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
        class_probabilities = fold_binary_probabilities(probabilities, self.id_to_label)
        label, confidence = choose_binary_label(class_probabilities)
        evidence_signals = analyze_input_evidence(title, description, transcript, thumbnail)
        consistency_features = compute_consistency_features(title, description, transcript).as_dict()
        warnings = build_prediction_warnings(confidence, evidence_signals, self.id_to_label)

        return PredictionResult(
            label=label,
            confidence=confidence,
            model_type=self.model_type,
            class_probabilities=class_probabilities,
            warnings=warnings,
            evidence_signals=tuple(evidence_signals),
            consistency_features=consistency_features,
            review_recommendation=review_recommendation(label, confidence, evidence_signals),
        )


class SklearnTextPredictor(BasePredictor):
    """Predict with the fast TF-IDF text classifier."""

    model_type = "text"

    def __init__(self, artifact_path: str | Path = SKLEARN_TEXT_ARTIFACT) -> None:
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Sklearn text model artifact was not found: {self.artifact_path}")
        artifact = joblib.load(self.artifact_path)
        self.model = artifact["model"]
        self.id_to_label = {int(idx): label for idx, label in artifact["id_to_label"].items()}

    def predict(
        self,
        title: str,
        description: str = "",
        transcript: str = "",
        thumbnail: str = "",
    ) -> PredictionResult:
        text = format_consistency_text_input(title, description, transcript)
        probabilities = self.model.predict_proba([text])[0]
        class_probabilities = {label: 0.0 for label in BINARY_LABEL_ORDER}
        for class_id, probability in zip(self.model.classes_, probabilities):
            label = normalize_binary_label(self.id_to_label[int(class_id)])
            class_probabilities[label] += float(probability)
        total = sum(class_probabilities.values())
        if total > 0:
            class_probabilities = {
                label: probability / total
                for label, probability in class_probabilities.items()
            }

        evidence_signals = analyze_input_evidence(title, description, transcript, thumbnail)
        consistency_features = compute_consistency_features(title, description, transcript).as_dict()
        coverage_features = compute_metadata_transcript_coverage(title, description, transcript)
        consistency_features.update(coverage_features)
        label, confidence = choose_synthetic_model_label(class_probabilities, coverage_features)
        warnings = build_prediction_warnings(confidence, evidence_signals, self.id_to_label)

        return PredictionResult(
            label=label,
            confidence=confidence,
            model_type="text_sklearn",
            class_probabilities=class_probabilities,
            warnings=warnings,
            evidence_signals=tuple(evidence_signals),
            consistency_features=consistency_features,
            review_recommendation=review_recommendation(label, confidence, evidence_signals),
        )


class MultimodalPredictor(BasePredictor):
    """Predict with the saved multimodal text-image model."""

    model_type = "multimodal"

    def __init__(
        self,
        model_dir: str | Path = PROJECT_ROOT / "outputs" / "model",
        max_length: int = 256,
        local_files_only: bool | None = None,
    ) -> None:
        from models.multimodal_classifier import load_model_for_inference

        self.model_dir = Path(model_dir)
        self.data_config = DataConfig(max_length=max_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (
            self.model,
            self.tokenizer,
            self.image_processor,
            self.id_to_label,
            _,
        ) = load_model_for_inference(
            self.model_dir,
            self.device,
            local_files_only=local_files_only,
        )

    def predict(
        self,
        title: str,
        description: str = "",
        transcript: str = "",
        thumbnail: str = "",
    ) -> PredictionResult:
        text = format_consistency_text_input(title, description, transcript)
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.data_config.max_length,
            return_tensors="pt",
        )
        image = load_thumbnail(
            thumbnail,
            image_size=self.data_config.image_size,
            base_dir=PROJECT_ROOT,
        )
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]

        inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "pixel_values": pixel_values.to(self.device),
        }

        with torch.no_grad():
            logits = self.model(**inputs)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
        class_probabilities = fold_binary_probabilities(probabilities, self.id_to_label)
        label, confidence = choose_binary_label(class_probabilities)
        evidence_signals = analyze_input_evidence(title, description, transcript, thumbnail)
        consistency_features = compute_consistency_features(title, description, transcript).as_dict()
        warnings = build_prediction_warnings(confidence, evidence_signals, self.id_to_label)

        return PredictionResult(
            label=label,
            confidence=confidence,
            model_type=self.model_type,
            class_probabilities=class_probabilities,
            warnings=warnings,
            evidence_signals=tuple(evidence_signals),
            consistency_features=consistency_features,
            review_recommendation=review_recommendation(label, confidence, evidence_signals),
        )


def fold_binary_probabilities(
    probabilities: torch.Tensor,
    id_to_label: dict[int, str],
) -> dict[str, float]:
    """Convert legacy four-class or new two-class probabilities into binary output."""

    binary = {label: 0.0 for label in BINARY_LABEL_ORDER}
    for idx in sorted(id_to_label):
        label = normalize_binary_label(id_to_label[idx])
        binary[label] += float(probabilities[idx].item())

    total = sum(binary.values())
    if total > 0:
        binary = {label: value / total for label, value in binary.items()}
    return binary


def choose_binary_label(class_probabilities: dict[str, float]) -> tuple[str, float]:
    """Bias toward detecting false content while still returning binary confidence."""

    return choose_binary_label_with_threshold(class_probabilities, FALSE_DETECTION_THRESHOLD)


def choose_binary_label_with_threshold(
    class_probabilities: dict[str, float],
    false_threshold: float,
) -> tuple[str, float]:
    """Return False only after the supplied mismatch-confidence threshold is met."""

    false_probability = class_probabilities.get(BINARY_FALSE_LABEL, 0.0)
    not_false_probability = class_probabilities.get(BINARY_NOT_FALSE_LABEL, 0.0)
    if false_probability >= false_threshold:
        return BINARY_FALSE_LABEL, false_probability
    return BINARY_NOT_FALSE_LABEL, not_false_probability


def choose_synthetic_model_label(
    class_probabilities: dict[str, float],
    coverage_features: dict[str, float],
) -> tuple[str, float]:
    """Keep the synthetic bootstrap model conservative on real-world text."""

    false_probability = class_probabilities.get(BINARY_FALSE_LABEL, 0.0)
    not_false_probability = class_probabilities.get(BINARY_NOT_FALSE_LABEL, 0.0)
    title_coverage = coverage_features.get("title_transcript_coverage", 0.0)
    description_coverage = coverage_features.get("description_transcript_coverage", 0.0)
    clearly_unsupported = title_coverage == 0.0 and description_coverage == 0.0
    if false_probability >= SYNTHETIC_MODEL_HARD_NEGATIVE_THRESHOLD and not clearly_unsupported:
        return BINARY_FALSE_LABEL, false_probability
    if false_probability >= SYNTHETIC_MODEL_FALSE_THRESHOLD and clearly_unsupported:
        return BINARY_FALSE_LABEL, false_probability
    return BINARY_NOT_FALSE_LABEL, not_false_probability


def token_coverage(left: object, right: object) -> float:
    """Measure how much of the short metadata text appears in the transcript."""

    left_tokens = meaningful_metadata_tokens(left)
    right_tokens = meaningful_metadata_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(left_tokens), 3)


def meaningful_metadata_tokens(value: object) -> set[str]:
    return {
        token
        for token in consistency_token_set(value)
        if len(token) > 2 and token not in METADATA_COVERAGE_STOPWORDS and not token.isdigit()
    }


def compute_metadata_transcript_coverage(
    title: object,
    description: object,
    transcript: object,
) -> dict[str, float]:
    title_coverage = token_coverage(title, transcript)
    description_coverage = token_coverage(description, transcript)
    return {
        "title_transcript_coverage": title_coverage,
        "description_transcript_coverage": description_coverage,
        "metadata_transcript_coverage": max(title_coverage, description_coverage),
    }


def analyze_input_evidence(
    title: str,
    description: str = "",
    transcript: str = "",
    thumbnail: str = "",
) -> list[dict[str, str]]:
    """Create simple input-quality and consistency-support signals."""

    title_text = clean_text(title) if is_meaningful_text(title) else ""
    description_text = clean_text(description) if is_meaningful_text(description) else ""
    transcript_text = clean_text(transcript) if is_meaningful_text(transcript) else ""
    thumbnail_text = clean_text(thumbnail) if is_meaningful_text(thumbnail) else ""
    combined = " ".join([title_text, description_text, transcript_text]).lower()
    signals: list[dict[str, str]] = []

    if transcript_text:
        signals.append(signal("transcript", "present", "Transcript text was available."))
    else:
        signals.append(signal("transcript", "missing", "Transcript was missing; consistency checking used weaker context."))

    if len(description_text) >= 80:
        signals.append(signal("description", "present", "Description has enough text for context."))
    elif description_text:
        signals.append(signal("description", "weak", "Description is short; source context may be limited."))
    else:
        signals.append(signal("description", "missing", "Description was missing."))

    if has_source_mention(combined):
        signals.append(signal("source_mentions", "present", "Text mentions sources or official evidence."))
    else:
        signals.append(signal("source_mentions", "missing", "No clear source or official context mention was found."))

    if has_high_risk_topic(combined):
        signals.append(signal("high_risk_topic", "present", "Content appears related to a high-risk public topic."))

    if has_clickbait_claim(title_text) and not has_source_mention(combined):
        signals.append(
            signal(
                "claim_evidence_mismatch",
                "present",
                "Title uses strong claim language that may need transcript support review.",
            )
        )

    if thumbnail_text:
        signals.append(signal("thumbnail", "present", "Thumbnail URL or path was available for review."))
    return signals


def signal(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status, "message": message}


def has_source_mention(text: str) -> bool:
    negative_patterns = (
        r"\bno\s+(clear\s+)?(official\s+)?source",
        r"\bwithout\s+(an?\s+)?(official\s+)?source",
        r"\bno\s+(official\s+)?evidence",
        "මූලාශ්‍රයක් නැහැ",
        "නිල මූලාශ්‍රයක් නැහැ",
        "සාක්ෂි නැහැ",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in negative_patterns):
        return False

    patterns = (
        r"https?://",
        "source",
        "official",
        "research",
        "study",
        "who",
        "ministry",
        "මූලාශ්",
        "නිල",
        "පර්යේෂ",
        "අධ්‍යයන",
        "වාර්තා",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def has_high_risk_topic(text: str) -> bool:
    terms = (
        "health",
        "medicine",
        "election",
        "government",
        "finance",
        "weather",
        "tsunami",
        "covid",
        "vaccine",
        "සෞඛ්‍ය",
        "ඖෂධ",
        "ඡන්ද",
        "රජය",
        "මුදල්",
        "කාලගුණ",
        "සුනාමි",
        "එන්නත්",
    )
    return any(term in text for term in terms)


def has_clickbait_claim(title: str) -> bool:
    title = title.lower()
    terms = (
        "100%",
        "secret",
        "breaking",
        "shocking",
        "exposed",
        "රහස",
        "හෙළි",
        "දැන්ම",
        "අනිවාර්ය",
        "කවුරුත් නොදන්න",
    )
    return any(term in title for term in terms)


def build_prediction_warnings(
    confidence: float,
    evidence_signals: list[dict[str, str]],
    id_to_label: dict[int, str],
) -> tuple[str, ...]:
    warnings: list[str] = []
    statuses = {(item["name"], item["status"]) for item in evidence_signals}

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        warnings.append("Prediction confidence is low; send this video for human consistency review before action.")
    if ("transcript", "missing") in statuses:
        warnings.append("Transcript was missing, so the model may miss title/transcript mismatch context.")
    if ("source_mentions", "missing") in statuses:
        warnings.append("No clear source or official context mention was detected in the available text.")
    if any(normalize_binary_label(label) != label for label in id_to_label.values()):
        warnings.append("A legacy four-label model was folded into binary False / Not False output.")

    return tuple(dict.fromkeys(warnings))


def review_recommendation(
    label: str,
    confidence: float,
    evidence_signals: list[dict[str, str]],
) -> str:
    statuses = {(item["name"], item["status"]) for item in evidence_signals}
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "human_review_required"
    if label == BINARY_FALSE_LABEL and (
        ("transcript", "missing") in statuses or ("claim_evidence_mismatch", "present") in statuses
    ):
        return "prioritize_consistency_review"
    if label == BINARY_FALSE_LABEL:
        return "likely_mismatch_review_recommended"
    return "review_optional"


def model_artifacts_available() -> dict[str, bool]:
    """Report which saved model artifacts are present."""

    return {
        "text": SKLEARN_TEXT_ARTIFACT.exists() or (PROJECT_ROOT / "outputs" / "model" / "text_config.json").exists(),
        "multimodal": (PROJECT_ROOT / "outputs" / "model" / "multimodal_config.json").exists(),
    }


def default_model_type() -> str:
    """Prefer the self-contained text model when it exists."""

    available = model_artifacts_available()
    if available["text"]:
        return "text"
    if available["multimodal"]:
        return "multimodal"
    return "text"


@lru_cache(maxsize=2)
def get_predictor(model_type: str) -> BasePredictor:
    """Load each model at most once per process."""

    normalized = (model_type or default_model_type()).lower()
    if normalized == "text":
        if SKLEARN_TEXT_ARTIFACT.exists():
            return SklearnTextPredictor()
        return TextOnlyPredictor()
    if normalized == "multimodal":
        return MultimodalPredictor()
    raise ValueError(f"Unsupported model type: {model_type}")
