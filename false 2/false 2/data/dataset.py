"""PyTorch dataset for Sinhala multimodal false-content detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from config.default_config import DataConfig
from utils.data_utils import load_thumbnail


class SinhalaYouTubeDataset(Dataset):
    """Load text, thumbnail, and label tensors from a CSV split."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: Any,
        image_processor: Any,
        label_to_id: dict[str, int],
        data_config: DataConfig | None = None,
        has_labels: bool = True,
        max_image_cache_size: int = 128,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_config = data_config or DataConfig()
        self.data = pd.read_csv(self.csv_path, encoding="utf-8-sig").fillna("")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.label_to_id = label_to_id
        self.has_labels = has_labels
        self.max_image_cache_size = max_image_cache_size
        self.image_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.data)

    def _get_pixel_values(self, thumbnail: str) -> torch.Tensor:
        cache_key = thumbnail or "__blank__"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]

        image = load_thumbnail(
            thumbnail,
            image_size=self.data_config.image_size,
            base_dir=self.csv_path.parent,
        )
        processed = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        if len(self.image_cache) >= self.max_image_cache_size:
            self.image_cache.pop(next(iter(self.image_cache)))
        self.image_cache[cache_key] = pixel_values
        return pixel_values

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        text = str(row.get(self.data_config.text_column, ""))
        thumbnail = str(row.get(self.data_config.thumbnail_column, ""))

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.data_config.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": self._get_pixel_values(thumbnail),
        }

        if self.has_labels:
            label = str(row[self.data_config.label_column])
            item["labels"] = torch.tensor(self.label_to_id[label], dtype=torch.long)

        return item
