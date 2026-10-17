"""Default configuration for the Sinhala false-content classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    source_csv: Path = data_dir / "sinhala_youtube_false_content_synthetic_1000.csv"
    real_source_csv: Path = data_dir / "real_sinhala_youtube_false_content.csv"
    train_csv: Path = PROJECT_ROOT / "outputs" / "train.csv"
    val_csv: Path = PROJECT_ROOT / "outputs" / "validation.csv"
    test_csv: Path = PROJECT_ROOT / "outputs" / "test.csv"
    text_column: str = "text"
    thumbnail_column: str = "thumbnail"
    label_column: str = "label"
    max_length: int = 256
    image_size: int = 224


@dataclass(frozen=True)
class ModelConfig:
    text_model_name: str = "xlm-roberta-base"
    image_model_name: str = "openai/clip-vit-base-patch32"
    fusion_hidden_size: int = 512
    dropout: float = 0.2
    use_clip_vision: bool | None = None
    freeze_encoders: bool = True
    local_files_only: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path = PROJECT_ROOT / "outputs"
    model_dir: Path = output_dir / "model"
    metrics_path: Path = output_dir / "metrics.json"
    confusion_matrix_path: Path = output_dir / "confusion_matrix.png"
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_workers: int = 0
    seed: int = 42
    patience: int = 2


BINARY_FALSE_LABEL = "False"
BINARY_NOT_FALSE_LABEL = "Not False"
BINARY_LABEL_ORDER = [BINARY_FALSE_LABEL, BINARY_NOT_FALSE_LABEL]
PREFERRED_LABEL_ORDER = BINARY_LABEL_ORDER
FALSE_DETECTION_THRESHOLD = 0.45
SYNTHETIC_MODEL_FALSE_THRESHOLD = 0.90
SYNTHETIC_MODEL_HARD_NEGATIVE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.65
