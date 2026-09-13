# BRAR small-data image comparison

This isolated experiment compares ImageNet-initialized EfficientNet-B4 with a
frozen DINOv2 ViT-S/14 encoder and class-weighted logistic regression. Both use
the same professor labels and the same five outer evaluation folds. No original
project files, splits, or trained weights are modified.

## Current scope

Three prespecified image-only experiments:

| Method | Image representation | Learned on BRAR |
| --- | --- | --- |
| `efficientnet` | Whole image; ImageNet initialization | Encoder and classification head |
| `dino_global` | Frozen DINOv2, whole image | Standardization and logistic regression |
| `dino_global_regions` | Frozen DINOv2, whole image + mean of three regional embeddings | Standardization and logistic regression |

The metadata is used for joins and audit only. No age, sex, BRAR level, or bone
loss measurement enters these predictors. Ordinal models, metadata fusion, and
auxiliary training on the 736 other images are deferred until this comparison
has been evaluated and annotation inconsistencies resolved.

## Run from the repository root in PowerShell

Use the Python environment that already runs the original CUDA model. The tested
environment is recorded in `artifacts/checks/preflight.json`. Do not replace a
working CUDA PyTorch installation with a CPU build.

```powershell
Set-Location D:\master-thesis\brar_classification
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If dependencies are missing, install with the SAME interpreter:

```powershell
python -m pip install -r small_data_pipeline/requirements.txt
```

For a fresh machine, install a matched CUDA torch/torchvision pair using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) first.

### Non-training preparation

These commands do not fit any classifier or update network parameters:

```powershell
python -m small_data_pipeline prepare
python -m unittest discover -s small_data_pipeline/tests -v
python -m small_data_pipeline check
python -m small_data_pipeline extract
```

`prepare` and `extract` deliberately refuse to overwrite existing artifacts.
If already completed, reuse them; `check` validates the prepared data and any
existing feature cache. For a new experiment, copy the configuration and choose
a new `output` directory. Paths in a config are resolved beside that config.

`check` downloads public pretrained weights into this folder's artifact cache,
runs batches of actual images on CUDA, validates output shapes and finiteness,
and verifies that all model parameters and buffers remain unchanged. It does
not test backpropagation or optimizer memory. `check --offline` uses random
weights solely for architecture diagnostics and does not produce features.

`extract` uses the frozen external DINOv2 model on all labeled images. This is
safe before splitting classifier fits because no BRAR-dependent transformation
is learned here: no fine-tuning, global scaling, PCA, or label use. Standardization
is learned later inside the training folds. Features contain all four views so
both DINO experiments can reuse the same cache.

### Training: run these only when ready

```powershell
python -m small_data_pipeline train --method dino_global
python -m small_data_pipeline train --method dino_global_regions
python -m small_data_pipeline train --method efficientnet
```

Run sequentially. DINO feature extraction uses CUDA; scikit-learn fits the small
logistic regression on CPU. EfficientNet training uses `cuda:0`, mixed precision,
batch size 4, and image size 448 for the RTX 4060 8 GB. Full training memory is
not established by an inference-only preflight. If CUDA runs out of memory,
lower `efficientnet.batch_size` to 2 before starting a new output directory.

Each command runs five folds. To start with a single baseline fold:

```powershell
python -m small_data_pipeline train --method efficientnet --fold 0
```

The all-fold command skips completed folds when their configuration matches.
Interrupted folds restart from external initialization; epoch-level resume is
not implemented. Do not edit hyperparameters between folds of an experiment.
Selection/refit histories are saved every epoch. A fold is complete only when
`complete.json` exists. No runtime or accuracy target is promised.

### After training

Return the command output or notify the assistant when all three commands finish.
The following commands are provided for the subsequent evaluation stage:

```powershell
python -m small_data_pipeline evaluate
python -m small_data_pipeline predict --method dino_global --input "path/to/panorama.jpg"
```

Evaluation creates `artifacts/evaluation/comparison.csv`, per-method metrics,
confusion matrices as JSON arrays, per-image out-of-fold predictions, and paired
macro-F1 differences against EfficientNet. It requires all five folds of each
requested method. To evaluate a completed subset:

```powershell
python -m small_data_pipeline evaluate --methods dino_global dino_global_regions
```

Prediction averages the five fold models. These ensemble probabilities are
uncalibrated. Ensemble inference is not the out-of-fold experiment and must not
be used to report performance on the labeled training images. Load only model
artifacts produced by this project (joblib deserialization requires trusted files).

## Evaluation design

1. Read `2017_classification.csv`, resolve IDs across all three BRAR folders, and
   join metadata. Validate integer labels, unique IDs, readable images, and exact
   duplicate pixel hashes. Preserve professor labels, including disagreements
   with existing numerical bone loss measurements.
2. Create five stratified outer folds with seed 42. Currently each test fold has
   two Stage I cases. Every labeled image receives one out-of-fold prediction.
3. If an image ID is not a unique patient, supply `dataset.patient_groups` with
   `image_id,patient_id`; grouped splits keep patients together. The default
   assumes one patient per image ID. Pixel hashes do not detect near-duplicates.
4. DINO: choose C from `[0.01, 0.1, 1, 10]` using three inner folds and macro F1.
   StandardScaler fits inside each inner fold. Refit the selected pipeline on all
   outer-development images, then predict the outer test fold once.
5. EfficientNet: choose an epoch using a stratified 15% internal validation set,
   class-weighted cross entropy, and early stopping on macro F1. Reinitialize the
   identical external encoder and random head, then refit on ALL outer-development
   images for the selected number of epochs. The outer test fold is never used
   for scheduling, stopping, weighting, or model selection.
6. Report pooled out-of-fold macro F1, balanced accuracy, per-stage recall,
   accuracy, ordinal MAE, quadratic weighted kappa, log loss, and fold metrics.
   Paired stratified patient bootstraps use identical sampled patients across
   methods. Their intervals are conditional on fixed fitted predictions and do
   not capture retraining variability or external validity.

All methods preserve image aspect ratio with black letterboxing. Regional views
are left, central, and right full-height square crops for panoramic images;
they require no tooth localization. The global representation is concatenated
with the MEAN regional representation (768 features for DINOv2 small), rather
than treating crops as independent patients. EfficientNet uses mild flip,
rotation, and brightness/contrast augmentation; DINO extraction is deterministic.

This is a controlled comparison of complete learning approaches, not a claim
that architecture alone causes a difference. It also differs from the historical
EfficientNet run in preprocessing, initialization, and evaluation splits.

## Interpretation and provenance

The local dataset contains 988 images; 252 are stage-labeled and all currently
come from original BRAR `level_3`. Only ten labels are Stage I. The results concern
this selected annotated subset, not full clinical diagnosis or performance on
all BRAR levels. Neither independent external validation nor calibration is
established. Treat model comparisons as exploratory internal validation, because
the dataset and previous test results have already informed study development.

CSV and image hashes, immutable folds, exact cached DINO encoder weights,
experiment settings, and software versions are recorded. Features and completed
folds are rejected when relevant settings or source files change. Never initialize
from `runs/brar_classification/...` for this comparison.

Sources:
- [DINOv2 reference implementation](https://github.com/facebookresearch/dinov2)
- [DINOv2 timm checkpoint and feature extraction](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
- [EfficientNet ImageNet checkpoint](https://huggingface.co/timm/efficientnet_b4.ra2_in1k)

No data or images are uploaded; public model downloads are the only network use.
