# How The 2017 Staging Pipeline Works

## Clinical Target

The model predicts patient-level 2017 periodontal stage from a panoramic
radiograph:

| Class index | Clinical label |
| ---: | --- |
| 0 | Stage I |
| 1 | Stage II |
| 2 | Stage III |
| 3 | Stage IV |

The labels were assigned by the professor in
`brar_anotation_dataset/brar/2017_classification.csv`. The original BRAR
`Level` is not used as the target because it represents a different grading
system.

## Data Flow

1. The professor CSV provides `image_id;stage_2017`.
2. `meta_data.csv` provides filename, age, gender, and original BRAR metadata.
3. The image id is extracted from filenames such as
   `patient_image_000047_689da8a4.jpg`.
4. The split generator scans the actual image folders to resolve the real
   image path.
5. Train/validation/test CSVs store both `stage_2017` and zero-indexed `label`.

The remaining BRAR images without professor labels are excluded from supervised
training. They may be useful later for self-supervised or semi-supervised
experiments, but they are not part of the main thesis classifier.

## Model

The core model is EfficientNet with a four-class classification head. The
default configuration is image-only because the thesis task is image
recognition according to professor-provided 2017 labels.

Age and gender metadata are still available through `--metadata` for ablation
experiments. This keeps the comparison explicit instead of silently letting
metadata drive a label that should be judged from the image pipeline.

## Small-Data Safeguards

The labeled dataset has 252 images and a very small Stage I class. The training
script therefore uses:

- stratified train/validation/test splits with every stage present in every
  split
- class-weighted cross entropy
- lighter label smoothing
- early stopping on macro F1 by default
- balanced accuracy and ordinal stage MAE in evaluation

The old three-class BRAR checkpoint can initialize compatible tensors, mainly
the image encoder. The incompatible three-class head is skipped automatically.

## What To Trust

For thesis reporting, trust the files under:

```text
runs/2017_classification/efficientnet_b4
```

Do not mix metrics from the old:

```text
runs/brar_classification/efficientnet_b4
```

Those were trained for a different three-class BRAR severity task.

//TODO
Complexity
Add number of missing tooths, if its greater than 5
Third Molar excluded

Methodology next. Pipeline
2 levels

1st will be yes or no
2nd the current

test with another dataset in the end
