"""Run 2017 periodontal stage prediction on panoramic radiographs.

Usage:
    python scripts/04_inference.py --input path/to/opg.jpg
    python scripts/04_inference.py --input path/to/opg_dir --output results/
    python scripts/04_inference.py --input opg.jpg --metadata --age 45 --gender 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.classifier import BRARClassifier


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


def load_model(
    cfg: dict,
    weights_path: Path,
    device: torch.device,
    use_metadata: bool,
) -> BRARClassifier:
    mcfg = cfg["model"]
    model = BRARClassifier(
        backbone=mcfg["backbone"],
        pretrained=False,
        num_classes=int(mcfg["num_classes"]),
        dropout=0.0,
        use_metadata=use_metadata,
        metadata_dim=int(mcfg["metadata_features"]),
    ).to(device)

    state = load_torch_state(weights_path, device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_inference_transform(image_size: int) -> transforms.Compose:
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


@torch.no_grad()
def predict_single(
    model: BRARClassifier,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    class_names: list[str],
    age: float | None = None,
    gender: int | None = None,
    use_metadata: bool = False,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    meta = None
    if use_metadata:
        if age is None or gender is None:
            raise ValueError("--age and --gender are required when --metadata is used")
        age_norm = (age - 18.0) / (100.0 - 18.0)
        meta = torch.tensor([[age_norm, float(gender)]], dtype=torch.float32).to(device)

    logits = model(img_tensor, meta)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(logits.argmax(dim=1).item())

    return {
        "filename": image_path.name,
        "predicted_stage": pred_idx + 1,
        "predicted_stage_name": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            f"stage_{idx + 1}": float(prob)
            for idx, prob in enumerate(probs.cpu().tolist())
        },
        "metadata_used": use_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="2017 periodontal stage inference")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default=None, help="Output directory for results")
    parser.add_argument("--weights", default=None, help="Model weights path")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--age", type=float, default=None, help="Patient age")
    parser.add_argument("--gender", type=int, default=None, choices=[0, 1], help="0=female, 1=male")
    parser.add_argument("--metadata", action="store_true", help="Enable age/gender ablation")
    parser.add_argument("--no-metadata", action="store_true", help="Compatibility flag")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_metadata = bool(cfg["model"].get("use_metadata", False))
    if args.metadata:
        use_metadata = True
    if args.no_metadata:
        use_metadata = False
    if use_metadata and (args.age is None or args.gender is None):
        parser.error("--age and --gender are required when metadata fusion is enabled")

    class_names = class_names_from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / cfg["output"]["project"] / cfg["output"]["name"]
    weights_path = Path(args.weights) if args.weights else out_dir / "weights" / "best.pt"

    model = load_model(cfg, weights_path, device, use_metadata)
    transform = build_inference_transform(int(cfg["training"]["image_size"]))

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p
            for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        )
    else:
        print(f"ERROR: {input_path} is not a valid file or directory")
        sys.exit(1)

    print(f"Running inference on {len(image_paths)} image(s)...")
    results = []

    for img_path in image_paths:
        result = predict_single(
            model,
            img_path,
            transform,
            device,
            class_names,
            age=args.age,
            gender=args.gender,
            use_metadata=use_metadata,
        )
        results.append(result)

        probs = result["probabilities"]
        prob_text = "  ".join(
            f"{class_names[idx]}: {probs[f'stage_{idx + 1}']:.3f}"
            for idx in range(len(class_names))
        )
        print(f"\n  {result['filename']}")
        print(
            f"    -> {result['predicted_stage_name']} "
            f"(confidence: {result['confidence']:.3f})"
        )
        print(f"    -> {prob_text}")

    if args.output:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        results_file = out_path / "inference_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")

    if len(results) > 1:
        from collections import Counter

        dist = Counter(r["predicted_stage"] for r in results)
        print(f"\n{'=' * 40}")
        print(f"Summary ({len(results)} images):")
        for stage in sorted(dist):
            pct = dist[stage] / len(results) * 100
            print(f"  {class_names[stage - 1]}: {dist[stage]} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
