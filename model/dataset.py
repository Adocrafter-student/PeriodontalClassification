"""PyTorch dataset for professor-labeled 2017 periodontal staging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(
    image_size: int,
    aug_cfg: dict | None,
    is_train: bool,
) -> transforms.Compose:
    """Build transform pipeline for train or eval."""
    if is_train and aug_cfg:
        t = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.5)),
            transforms.RandomRotation(degrees=aug_cfg.get("rotation_degrees", 8)),
            transforms.ColorJitter(
                brightness=aug_cfg.get("brightness", 0.12),
                contrast=aug_cfg.get("contrast", 0.12),
            ),
        ]
        if aug_cfg.get("gaussian_blur_prob", 0) > 0:
            t.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=7,
                            sigma=aug_cfg.get("gaussian_blur_sigma", [0.5, 1.5]),
                        )
                    ],
                    p=aug_cfg["gaussian_blur_prob"],
                )
            )
        t.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        if aug_cfg.get("random_erasing_prob", 0) > 0:
            t.append(transforms.RandomErasing(p=aug_cfg["random_erasing_prob"]))
        return transforms.Compose(t)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class BRARDataset(Dataset):
    """Dataset returning dictionaries for 2017 periodontal staging.

    Labels are zero-indexed: Stage I -> 0, Stage II -> 1, Stage III -> 2,
    Stage IV -> 3. Age/gender metadata is still returned so metadata ablations
    can be enabled without changing the data loader.
    """

    AGE_MIN = 18.0
    AGE_MAX = 100.0

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        transform: transforms.Compose | None = None,
        use_metadata: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.use_metadata = use_metadata

        if "label" not in self.df.columns and "stage_2017" not in self.df.columns:
            raise ValueError(
                f"{csv_path} must contain either 'label' or 'stage_2017'. "
                "Run scripts/01_prepare_data.py to regenerate 2017 splits."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        img_path = self.data_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if "label" in row:
            label = int(row["label"])
            stage_2017 = label + 1
        else:
            stage_2017 = int(row["stage_2017"])
            label = stage_2017 - 1

        age_norm = (float(row["Age"]) - self.AGE_MIN) / (self.AGE_MAX - self.AGE_MIN)
        meta = torch.tensor([age_norm, float(row["Gender"])], dtype=torch.float32)

        return {
            "image": image,
            "metadata": meta,
            "label": torch.tensor(label, dtype=torch.long),
            "stage_2017": torch.tensor(stage_2017, dtype=torch.long),
            "filename": row["File name"],
            "image_id": int(row["image_id"]) if "image_id" in row else -1,
            "brar_level": int(row["brar_level"]) if "brar_level" in row else -1,
            "brar": float(row["Bone resorption Age"]),
        }


def compute_class_weights(csv_path: str | Path, num_classes: int = 4) -> torch.Tensor:
    """Compute inverse-frequency class weights for zero-indexed labels."""
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        labels = df["label"].astype(int)
    elif "stage_2017" in df.columns:
        labels = df["stage_2017"].astype(int) - 1
    else:
        raise ValueError(
            f"{csv_path} must contain either 'label' or 'stage_2017' to compute weights."
        )

    counts = labels.value_counts().sort_index()
    total = len(labels)
    weights = []
    missing = []

    for class_idx in range(num_classes):
        count = int(counts.get(class_idx, 0))
        if count == 0:
            missing.append(class_idx + 1)
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    if missing:
        raise ValueError(f"Training split is missing 2017 stages: {missing}")

    return torch.tensor(weights, dtype=torch.float32)
