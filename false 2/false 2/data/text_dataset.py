"""Text-only PyTorch dataset for Sinhala false-content detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from config.default_config import DataConfig
from utils.data_utils import normalize_binary_label, normalize_label


class SinhalaTextDataset(Dataset):
    """Load combined Sinhala text and labels from a prepared CSV split."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: Any,
        label_to_id: dict[str, int],
        data_config: DataConfig | None = None,
        has_labels: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_config = data_config or DataConfig()
        self.data = pd.read_csv(self.csv_path, encoding="utf-8-sig").fillna("")
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        text = str(row.get(self.data_config.text_column, ""))
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
        }

        if self.has_labels:
            raw_label = str(row[self.data_config.label_column])
            if "Not False" in self.label_to_id:
                label = normalize_binary_label(raw_label)
            else:
                label = normalize_label(raw_label)
            item["labels"] = torch.tensor(self.label_to_id[label], dtype=torch.long)

        return item
