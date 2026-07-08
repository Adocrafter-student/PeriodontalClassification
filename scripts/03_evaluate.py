"""Evaluate the trained 2017 periodontal staging classifier.

Produces accuracy, balanced accuracy, macro/weighted F1, per-class recall,
ordinal stage MAE, a confusion matrix, and per-sample predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.classifier import BRARClassifier
from model.dataset import BRARDataset, build_transforms


def class_names_from_config(cfg: dict) -> list[str]:
    num_classes = int(cfg["model"]["num_classes"])
    classes = cfg.get("classes", {})
    return [
        classes.get(i) or classes.get(str(i)) or f"Stage {i}"
        for i in range(1, num_classes + 1)
    ]


def load_torch_state(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def ordinal_mae(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return 0.0
    return float(np.mean(np.abs(np.asarray(labels) - np.asarray(preds))))


@torch.no_grad()
def collect_predictions(
    model: BRARClassifier,
    loader: DataLoader,
    device: torch.device,
    use_metadata: bool,
) -> tuple[list[int], list[int], list[list[float]], list[str], list[int], list[int], list[float]]:
    model.eval()
    labels, preds, probs = [], [], []
    filenames, image_ids, brar_levels, brar_values = [], [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        meta = batch["metadata"].to(device) if use_metadata else None

        logits = model(images, meta)
        batch_probs = torch.softmax(logits, dim=1)
        batch_preds = logits.argmax(dim=1)

        labels.extend(batch["label"].cpu().tolist())
        preds.extend(batch_preds.cpu().tolist())
        probs.extend(batch_probs.cpu().tolist())
        filenames.extend(batch["filename"])
        image_ids.extend(batch["image_id"].cpu().tolist())
        brar_levels.extend(batch["brar_level"].cpu().tolist())
        brar_values.extend(batch["brar"].cpu().tolist())

    return labels, preds, probs, filenames, image_ids, brar_levels, brar_values


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_path: Path,
) -> list[list[int]]:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    ) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (counts)")

    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {out_path}")
    return cm.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 2017 staging classifier")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument("--metadata", action="store_true", help="Enable age/gender ablation")
    parser.add_argument("--no-metadata", action="store_true", help="Compatibility flag")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    class_names = class_names_from_config(cfg)
    num_classes = int(mcfg["num_classes"])

    use_metadata = bool(mcfg.get("use_metadata", False))
    if args.metadata:
        use_metadata = True
    if args.no_metadata:
        use_metadata = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Metadata fusion: {use_metadata}")

    data_root = (ROOT / cfg["dataset"]["root"]).resolve()
    splits_dir = (ROOT / cfg["dataset"]["splits_dir"]).resolve()
    split_csv = splits_dir / f"{args.split}.csv"

    tf = build_transforms(int(tcfg["image_size"]), None, is_train=False)
    ds = BRARDataset(split_csv, data_root, tf, use_metadata)
    loader = DataLoader(
        ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    print(f"Evaluating on {args.split}: {len(ds)} samples")

    model = BRARClassifier(
        backbone=mcfg["backbone"],
        pretrained=False,
        num_classes=num_classes,
        dropout=float(mcfg["dropout"]),
        use_metadata=use_metadata,
        metadata_dim=int(mcfg["metadata_features"]),
    ).to(device)

    out_dir = ROOT / cfg["output"]["project"] / cfg["output"]["name"]
    weights_path = Path(args.weights) if args.weights else out_dir / "weights" / "best.pt"
    state = load_torch_state(weights_path, device)
    model.load_state_dict(state)
    print(f"Loaded weights from {weights_path}")

    labels, preds, probs, filenames, image_ids, brar_levels, brar_values = collect_predictions(
        model, loader, device, use_metadata
    )

    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    ord_mae = ordinal_mae(labels, preds)

    print(f"\n{'=' * 50}")
    print(f"  {args.split.upper()} RESULTS")
    print(f"{'=' * 50}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Balanced accuracy:  {balanced_acc:.4f}")
    print(f"  F1 (macro):         {f1_macro:.4f}")
    print(f"  F1 (weighted):      {f1_w:.4f}")
    print(f"  Ordinal stage MAE:  {ord_mae:.4f}")
    print(f"{'=' * 50}\n")

    report = classification_report(
        labels,
        preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    print(report)

    eval_dir = out_dir / f"eval_{args.split}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cm = plot_confusion_matrix(labels, preds, class_names, eval_dir / "confusion_matrix.png")

    pred_data = {
        "image_id": image_ids,
        "filename": filenames,
        "true_stage": [l + 1 for l in labels],
        "pred_stage": [p + 1 for p in preds],
        "true_stage_name": [class_names[l] for l in labels],
        "pred_stage_name": [class_names[p] for p in preds],
        "correct": [l == p for l, p in zip(labels, preds)],
        "ordinal_error": [abs(l - p) for l, p in zip(labels, preds)],
        "brar_level": brar_levels,
        "brar_age_ratio": brar_values,
    }
    for idx, name in enumerate(class_names, start=1):
        column = f"prob_stage_{idx}"
        pred_data[column] = [float(row[idx - 1]) for row in probs]

    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(eval_dir / "predictions.csv", index=False)
    print(f"Per-sample predictions saved to {eval_dir / 'predictions.csv'}")

    report_dict = classification_report(
        labels,
        preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    per_class_recall = {
        name: float(report_dict[name]["recall"])
        for name in class_names
        if name in report_dict
    }
    metrics = {
        "split": args.split,
        "num_samples": len(labels),
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_w),
        "ordinal_stage_mae": float(ord_mae),
        "per_class_recall": per_class_recall,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {eval_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
