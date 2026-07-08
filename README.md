# 2017 Periodontal Staging From BRAR Panoramic Images

This project revives the old BRAR image classifier as a four-class 2017
periodontal staging classifier.

The supervised target is no longer the original BRAR `Level` column. Ground
truth now comes only from:

```text
../brar_anotation_dataset/brar/2017_classification.csv
```

That file is expected to be semicolon-separated with no header:

```text
image_id;stage_2017
```

Example: `47;3` means image id `000047` is professor-labeled as Stage III.

## Dataset Status

The current audit found:

| Item | Count |
| --- | ---: |
| Professor-labeled images | 252 |
| Missing labeled images | 0 |
| Duplicate label ids | 0 |
| BRAR images without 2017 labels | 736 |

2017 label distribution:

| Stage | Count |
| --- | ---: |
| Stage I | 10 |
| Stage II | 53 |
| Stage III | 140 |
| Stage IV | 49 |

Important: all 252 professor-labeled images currently live under the original
`level_3` BRAR folder. The old folder/BRAR level is retained for audit context
only and must not be used as the target.

## Quick Start

Create or activate a Python environment first. This shell did not expose
`python` or `py` on PATH during inspection, so use the interpreter available in
your IDE or install/activate one before running these commands.

```bash
cd brar_classification
pip install -r requirements.txt

# Build audited 2017 train/val/test CSVs.
python scripts/01_prepare_data.py

# Check one batch and one model forward pass.
python scripts/00_smoke_test.py

# Train the image-only four-stage classifier.
python scripts/02_train.py

# Evaluate the best checkpoint.
python scripts/03_evaluate.py

# Run inference on one image or a directory.
python scripts/04_inference.py --input path/to/opg.jpg
```

For the optional age/gender ablation:

```bash
python scripts/02_train.py --metadata
python scripts/03_evaluate.py --metadata
python scripts/04_inference.py --input path/to/opg.jpg --metadata --age 45 --gender 1
```

## Pipeline

1. `scripts/01_prepare_data.py`
   - Parses professor labels from `2017_classification.csv`.
   - Joins labels to `meta_data.csv` by numeric image id.
   - Resolves actual image paths by scanning `level_1`, `level_2`, and `level_3`.
   - Writes `data/splits/train.csv`, `val.csv`, `test.csv`, and
     `audit_summary.json`.

2. `model/dataset.py`
   - Loads `stage_2017` / zero-indexed `label`.
   - Keeps `brar_level` only as historical metadata.
   - Returns age/gender metadata for explicit ablation runs.

3. `scripts/02_train.py`
   - Trains a four-class EfficientNet classifier.
   - Defaults to image-only classification.
   - Uses class-weighted cross entropy and early stopping on macro F1.
   - Optionally initializes compatible encoder tensors from the old BRAR
     checkpoint while skipping incompatible three-class head tensors.

4. `scripts/03_evaluate.py`
   - Reports accuracy, balanced accuracy, macro F1, weighted F1, per-class
     recall, and ordinal stage MAE.
   - Saves a confusion matrix, `metrics.json`, and per-sample predictions.

## Outputs

New training runs are written to:

```text
runs/2017_classification/efficientnet_b4
```

The old BRAR checkpoint remains in:

```text
runs/brar_classification/efficientnet_b4/weights/best.pt
```

It is used only as a compatible initialization source, not as 2017-classified
evidence.
