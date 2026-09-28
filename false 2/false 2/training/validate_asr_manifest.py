"""Validate Sinhala ASR manifests before Whisper fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_COLUMNS = {"audio_path", "transcript", "split", "source", "duration_seconds"}
ALLOWED_SPLITS = {"train", "validation", "test"}
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sinhala_asr_manifest_template.csv"
DEFAULT_MAX_DURATION_SECONDS = 30.0


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a CSV manifest with string-safe defaults."""

    return pd.read_csv(path, keep_default_na=False)


def resolve_audio_path(audio_path: str, base_dir: str | Path = PROJECT_ROOT) -> Path:
    """Resolve a manifest audio path relative to the project root."""

    path = Path(str(audio_path).strip())
    return path if path.is_absolute() else Path(base_dir) / path


def validate_asr_manifest(
    dataframe: pd.DataFrame,
    base_dir: str | Path = PROJECT_ROOT,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
) -> list[str]:
    """Return human-readable validation errors for a Sinhala ASR manifest."""

    errors: list[str] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}.")
        return errors

    normalized_paths: list[str] = []
    for index, row in dataframe.iterrows():
        row_number = index + 2
        audio_path = str(row.get("audio_path", "")).strip()
        transcript = str(row.get("transcript", "")).strip()
        split = str(row.get("split", "")).strip().lower()
        duration_value = row.get("duration_seconds", "")

        if not audio_path:
            errors.append(f"Row {row_number}: audio_path is empty.")
        else:
            resolved_path = resolve_audio_path(audio_path, base_dir)
            normalized_paths.append(str(resolved_path).lower())
            if not resolved_path.exists():
                errors.append(f"Row {row_number}: audio file does not exist: {audio_path}.")

        if not transcript:
            errors.append(f"Row {row_number}: transcript is empty.")

        if split not in ALLOWED_SPLITS:
            errors.append(
                f"Row {row_number}: split must be one of {', '.join(sorted(ALLOWED_SPLITS))}."
            )

        try:
            duration_seconds = float(duration_value)
        except (TypeError, ValueError):
            errors.append(f"Row {row_number}: duration_seconds must be numeric.")
            continue

        if duration_seconds <= 0:
            errors.append(f"Row {row_number}: duration_seconds must be greater than 0.")
        elif duration_seconds > max_duration_seconds:
            errors.append(
                f"Row {row_number}: duration_seconds {duration_seconds:g} exceeds "
                f"max {max_duration_seconds:g}."
            )

    duplicate_paths = sorted({path for path in normalized_paths if normalized_paths.count(path) > 1})
    for path in duplicate_paths:
        errors.append(f"Duplicate audio file in manifest: {path}.")

    return errors


def split_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    if "split" not in dataframe.columns:
        return {}
    return {
        split: int(count)
        for split, count in dataframe["split"].astype(str).str.lower().value_counts().sort_index().items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Sinhala Whisper ASR manifest CSV.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to ASR manifest CSV.")
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=DEFAULT_MAX_DURATION_SECONDS,
        help="Maximum allowed clip duration for fine-tuning examples.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    dataframe = load_manifest(manifest_path)
    errors = validate_asr_manifest(
        dataframe,
        base_dir=PROJECT_ROOT,
        max_duration_seconds=args.max_duration_seconds,
    )
    payload = {
        "manifest": str(manifest_path),
        "row_count": int(len(dataframe)),
        "split_counts": split_counts(dataframe),
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
