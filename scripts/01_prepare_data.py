"""Prepare professor-labeled 2017 periodontal staging splits.

The original BRAR dataset folders (level_1/2/3) are used only to locate image
files. The supervised label is stage_2017 from 2017_classification.csv.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGE_ID_RE = re.compile(r"patient_image_(\d+)_")
STAGE_NAMES = {
    1: "Stage I",
    2: "Stage II",
    3: "Stage III",
    4: "Stage IV",
}


def extract_image_id(value: str) -> int:
    match = IMAGE_ID_RE.search(str(value))
    if not match:
        raise ValueError(f"Could not extract image id from {value!r}")
    return int(match.group(1))


def read_professor_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["image_id", "stage_2017"],
        dtype={"image_id": int, "stage_2017": int},
    )

    if labels.empty:
        raise SystemExit(f"No professor labels found in {path}")

    invalid = labels[~labels["stage_2017"].isin(STAGE_NAMES.keys())]
    if not invalid.empty:
        raise SystemExit(
            "Invalid 2017 stages found: "
            + ", ".join(map(str, sorted(invalid["stage_2017"].unique())))
        )

    return labels


def build_image_index(data_root: Path) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, object]] = []
    seen: dict[int, list[str]] = {}

    for image_path in sorted(data_root.glob("level_*/*.jpg")):
        image_id = extract_image_id(image_path.name)
        rel_path = image_path.relative_to(data_root).as_posix()
        seen.setdefault(image_id, []).append(rel_path)
        records.append(
            {
                "image_id": image_id,
                "disk_file_name": image_path.name,
                "image_path": rel_path,
                "old_level_dir": image_path.parent.name,
            }
        )

    duplicate_image_ids = sum(1 for paths in seen.values() if len(paths) > 1)
    return pd.DataFrame(records), duplicate_image_ids


def stratified_three_way_split(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    test_parts = []

    for stage, group in df.groupby("stage_2017", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed + int(stage))
        n = len(shuffled)
        val_count = max(1, round(n * val_ratio))
        test_count = max(1, round(n * test_ratio))
        train_count = n - val_count - test_count

        if train_count < 1:
            raise SystemExit(
                f"Stage {stage} has only {n} samples; cannot preserve all splits."
            )

        train_parts.append(shuffled.iloc[:train_count])
        val_parts.append(shuffled.iloc[train_count : train_count + val_count])
        test_parts.append(shuffled.iloc[train_count + val_count :])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_stage_distribution(name: str, df: pd.DataFrame) -> dict[str, int]:
    counts = df["stage_2017"].value_counts().sort_index()
    print(f"\n{name}: {len(df)} samples")
    for stage, count in counts.items():
        pct = count / len(df) * 100
        print(f"  Stage {stage}: {count:4d} ({pct:5.1f}%)")
    return {str(int(stage)): int(count) for stage, count in counts.items()}


def main() -> None:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_root = (ROOT / cfg["dataset"]["root"]).resolve()
    metadata_csv = (ROOT / cfg["dataset"]["metadata_csv"]).resolve()
    labels_csv = (ROOT / cfg["dataset"]["classification_csv"]).resolve()
    splits_dir = (ROOT / cfg["dataset"]["splits_dir"]).resolve()
    splits_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["dataset"]["random_seed"])
    val_ratio = float(cfg["dataset"]["val_ratio"])
    test_ratio = float(cfg["dataset"]["test_ratio"])

    labels = read_professor_labels(labels_csv)
    metadata = pd.read_csv(metadata_csv)
    metadata["image_id"] = metadata["File name"].apply(extract_image_id)
    image_index, duplicate_image_ids = build_image_index(data_root)

    duplicate_label_ids = int(labels["image_id"].duplicated().sum())
    duplicate_metadata_ids = int(metadata["image_id"].duplicated().sum())

    merged = labels.merge(metadata, on="image_id", how="left", validate="one_to_one")
    merged = merged.merge(image_index, on="image_id", how="left", validate="one_to_one")

    missing_metadata = int(merged["File name"].isna().sum())
    missing_images = int(merged["image_path"].isna().sum())
    unlabeled_images = int((~image_index["image_id"].isin(labels["image_id"])).sum())

    print("2017 label audit")
    print(f"  labels loaded:              {len(labels)}")
    print(f"  duplicate label ids:        {duplicate_label_ids}")
    print(f"  duplicate metadata ids:     {duplicate_metadata_ids}")
    print(f"  duplicate image ids:        {duplicate_image_ids}")
    print(f"  missing labeled images:     {missing_images}")
    print(f"  missing metadata rows:      {missing_metadata}")
    print(f"  images without 2017 label:  {unlabeled_images}")

    print("\n2017 stage counts")
    stage_counts = print_stage_distribution("all labeled", merged)

    print("\n2017 stage vs old BRAR folder")
    stage_vs_folder = (
        merged.groupby(["stage_2017", "old_level_dir"])
        .size()
        .reset_index(name="count")
        .sort_values(["stage_2017", "old_level_dir"])
    )
    for row in stage_vs_folder.itertuples(index=False):
        print(f"  Stage {row.stage_2017}, {row.old_level_dir}: {row.count}")

    if duplicate_label_ids or duplicate_metadata_ids or duplicate_image_ids:
        raise SystemExit("Audit failed because duplicate ids were found.")
    if missing_images or missing_metadata:
        raise SystemExit("Audit failed because labeled rows could not be resolved.")

    merged = merged.rename(columns={"Level": "brar_level"})
    merged["label"] = merged["stage_2017"] - 1
    merged["stage_name"] = merged["stage_2017"].map(STAGE_NAMES)
    merged["label_source"] = "professor_2017"

    columns = [
        "image_id",
        "File name",
        "image_path",
        "stage_2017",
        "label",
        "stage_name",
        "brar_level",
        "old_level_dir",
        "Age",
        "Gender",
        "Bone resorption",
        "Bone resorption Age",
        "Number of missing teeth",
        "Implant",
        "Residual root",
        "Functional tooth logarithm",
        "label_source",
    ]
    split_source = merged[columns].sort_values("image_id").reset_index(drop=True)

    train_df, val_df, test_df = stratified_three_way_split(
        split_source,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_counts: dict[str, dict[str, int]] = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = splits_dir / f"{name}.csv"
        split_df = split_df.copy()
        split_df["split"] = name
        split_df.to_csv(out, index=False)
        split_counts[name] = print_stage_distribution(name, split_df)

    audit = {
        "labels_loaded": int(len(labels)),
        "stage_counts": stage_counts,
        "duplicate_label_ids": duplicate_label_ids,
        "duplicate_metadata_ids": duplicate_metadata_ids,
        "duplicate_image_ids": duplicate_image_ids,
        "missing_labeled_images": missing_images,
        "missing_metadata_rows": missing_metadata,
        "images_without_2017_label": unlabeled_images,
        "split_counts": split_counts,
        "unlabeled_policy": cfg["dataset"].get("unlabeled_policy", "exclude"),
    }

    with open(splits_dir / "audit_summary.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\nDone - split CSVs and audit saved to {splits_dir}")


if __name__ == "__main__":
    main()
