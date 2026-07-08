"""Train the 2017 periodontal staging classifier.

Usage:
    python scripts/02_train.py
    python scripts/02_train.py --epochs 1 --device cpu
    python scripts/02_train.py --metadata
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.classifier import BRARClassifier
from model.dataset import BRARDataset, build_transforms, compute_class_weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg_device: str | int) -> torch.device:
    """Integer 0 selects cuda:0 when CUDA is available; otherwise CPU."""
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(cfg_device, str) and cfg_device.isdigit():
        cfg_device = int(cfg_device)
    if isinstance(cfg_device, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cfg_device}")
        return torch.device("cpu")
    return torch.device(str(cfg_device))


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


def load_compatible_weights(
    model: nn.Module,
    init_from: str | None,
    device: torch.device,
) -> None:
    if not init_from:
        return

    init_path = Path(init_from)
    if not init_path.is_absolute():
        init_path = ROOT / init_path
    init_path = init_path.resolve()

    if not init_path.exists():
        print(f"WARNING: init checkpoint not found: {init_path}")
        return

    source_state = load_torch_state(init_path, device)
    target_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in source_state.items():
        clean_key = key.removeprefix("module.")
        if clean_key in target_state and target_state[clean_key].shape == value.shape:
            compatible[clean_key] = value
        else:
            skipped.append(clean_key)

    model.load_state_dict(compatible, strict=False)
    print(
        "Initialized from old BRAR checkpoint: "
        f"{len(compatible)} tensors loaded, {len(skipped)} skipped"
    )
    if skipped:
        print("Skipped incompatible tensors include:", ", ".join(skipped[:6]))


def cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total: int,
    warmup: int,
    base_lr: float,
) -> float:
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, total - warmup)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def ordinal_mae(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return 0.0
    return float(np.mean(np.abs(np.asarray(labels) - np.asarray(preds))))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_metadata: bool,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in tqdm(loader, desc="  train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        meta = batch["metadata"].to(device) if use_metadata else None

        logits = model(images, meta)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return running_loss / len(all_labels), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_metadata: bool,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        meta = batch["metadata"].to(device) if use_metadata else None

        logits = model(images, meta)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return {
        "loss": running_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "ordinal_mae": ordinal_mae(all_labels, all_preds),
    }


def is_improved(metric_name: str, current: float, best: float) -> bool:
    if metric_name == "val_loss":
        return current < best
    return current > best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2017 staging classifier")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metadata", action="store_true", help="Enable age/gender ablation")
    parser.add_argument("--no-metadata", action="store_true", help="Compatibility flag")
    parser.add_argument("--init-from", default=None, help="Override initialization checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    mcfg = cfg["model"]
    acfg = cfg["augmentation"]

    seed = int(cfg["dataset"]["random_seed"])
    set_seed(seed)

    epochs = args.epochs or int(tcfg["epochs"])
    batch_size = args.batch_size or int(tcfg["batch_size"])
    lr = args.lr or float(tcfg["lr"])
    use_metadata = bool(mcfg.get("use_metadata", False))
    if args.metadata:
        use_metadata = True
    if args.no_metadata:
        use_metadata = False

    image_size = int(tcfg["image_size"])
    patience = int(tcfg["patience"])
    warmup = int(tcfg.get("warmup_epochs", 3))
    label_smoothing = float(tcfg.get("label_smoothing", 0.0))
    monitor = tcfg.get("monitor", "macro_f1")
    valid_monitors = {"macro_f1", "balanced_accuracy", "accuracy", "val_loss"}
    if monitor not in valid_monitors:
        raise ValueError(f"training.monitor must be one of {sorted(valid_monitors)}")

    cfg_device = args.device if args.device is not None else tcfg["device"]
    device = get_device(cfg_device)
    print(f"Device: {device}")
    print(f"Metadata fusion: {use_metadata}")
    print(f"Early stopping monitor: {monitor}")

    data_root = (ROOT / cfg["dataset"]["root"]).resolve()
    splits_dir = (ROOT / cfg["dataset"]["splits_dir"]).resolve()

    train_tf = build_transforms(image_size, acfg, is_train=True)
    val_tf = build_transforms(image_size, None, is_train=False)

    train_ds = BRARDataset(splits_dir / "train.csv", data_root, train_tf, use_metadata)
    val_ds = BRARDataset(splits_dir / "val.csv", data_root, val_tf, use_metadata)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=pin_memory,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = BRARClassifier(
        backbone=mcfg["backbone"],
        pretrained=bool(mcfg["pretrained"]),
        num_classes=int(mcfg["num_classes"]),
        dropout=float(mcfg["dropout"]),
        use_metadata=use_metadata,
        metadata_dim=int(mcfg["metadata_features"]),
    ).to(device)

    init_from = args.init_from if args.init_from is not None else tcfg.get("init_from")
    load_compatible_weights(model, init_from, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    if bool(tcfg["use_class_weights"]):
        weights = compute_class_weights(
            splits_dir / "train.csv",
            num_classes=int(mcfg["num_classes"]),
        ).to(device)
        print(f"Class weights: {weights.tolist()}")
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(tcfg["weight_decay"]),
    )

    out_dir = ROOT / cfg["output"]["project"] / cfg["output"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    best_metric = math.inf if monitor == "val_loss" else -math.inf
    best_epoch = 0
    no_improve = 0
    history = []

    print(f"\n{'=' * 72}")
    print(f"Training for {epochs} epochs | patience={patience}")
    print(f"{'=' * 72}\n")

    t0 = time.time()

    for epoch in range(epochs):
        current_lr = cosine_lr(optimizer, epoch, epochs, warmup, lr)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_metadata
        )
        val_metrics = validate(model, val_loader, criterion, device, use_metadata)

        current_metric = (
            val_metrics["loss"] if monitor == "val_loss" else val_metrics[monitor]
        )
        improved = is_improved(monitor, current_metric, best_metric)
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), weights_dir / "best.pt")
        else:
            no_improve += 1

        marker = " *" if improved else ""
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_macro_f1={val_metrics['macro_f1']:.3f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f} "
            f"val_ord_mae={val_metrics['ordinal_mae']:.3f}{marker}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_ordinal_mae": val_metrics["ordinal_mae"],
                "monitor": monitor,
                "monitor_value": current_metric,
            }
        )

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    elapsed = time.time() - t0
    torch.save(model.state_dict(), weights_dir / "last.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"\n{'=' * 72}")
    print(f"Training complete in {elapsed / 60:.1f} min")
    print(f"Best {monitor}: {best_metric:.4f} at epoch {best_epoch + 1}")
    print(f"Weights saved to {weights_dir}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
