"""Smoke test the 2017 staging data loader and model forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.classifier import BRARClassifier
from model.dataset import BRARDataset, build_transforms


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test 2017 staging pipeline")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = (ROOT / cfg["dataset"]["root"]).resolve()
    splits_dir = (ROOT / cfg["dataset"]["splits_dir"]).resolve()
    split_csv = splits_dir / f"{args.split}.csv"
    if not split_csv.exists():
        raise SystemExit(
            f"{split_csv} does not exist. Run scripts/01_prepare_data.py first."
        )

    image_size = int(cfg["training"]["image_size"])
    num_classes = int(cfg["model"]["num_classes"])
    use_metadata = bool(cfg["model"].get("use_metadata", False))

    ds = BRARDataset(
        split_csv,
        data_root,
        build_transforms(image_size, None, is_train=False),
        use_metadata=use_metadata,
    )
    loader = DataLoader(ds, batch_size=min(2, len(ds)), shuffle=False)
    batch = next(iter(loader))

    labels = batch["label"]
    assert labels.min().item() >= 0
    assert labels.max().item() < num_classes

    model = BRARClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        num_classes=num_classes,
        dropout=float(cfg["model"]["dropout"]),
        use_metadata=use_metadata,
        metadata_dim=int(cfg["model"]["metadata_features"]),
    )
    metadata = batch["metadata"] if use_metadata else None
    logits = model(batch["image"], metadata)
    assert tuple(logits.shape) == (len(labels), num_classes)

    print(f"Loaded batch image tensor: {tuple(batch['image'].shape)}")
    print(f"Labels min/max: {labels.min().item()}..{labels.max().item()}")
    print(f"Model logits shape: {tuple(logits.shape)}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
