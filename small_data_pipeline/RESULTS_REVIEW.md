# Review of completed five-fold experiments

These results pool exactly one held-out prediction per labeled image: 252 images,
five shared outer folds, all four professor-assigned stages. All 15 checkpoints
were reloaded and reproduced their stored predicted classes. Maximum probability
difference was below 3e-8. Recorded train/test memberships and EfficientNet epoch
selection/refit lengths passed the audit. No additional training was performed.

## Main comparison

| Method | Correct / 252 | Accuracy | Balanced accuracy | Macro F1 | Ordinal MAE | Quadratic weighted kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNet | 147 | 58.33% | 51.70% | 0.4926 | 0.4286 | 0.6443 |
| DINOv2 whole image | 153 | 60.71% | 45.89% | 0.4535 | 0.4167 | 0.5948 |
| DINOv2 whole image + regions | 159 | 63.10% | 49.90% | 0.4948 | 0.3889 | 0.6130 |

Always predicting Stage III would give 55.56% accuracy but only 25% balanced
accuracy. This illustrates why ordinary accuracy alone is inadequate here.

The prespecified model-selection measure was macro F1. Regional DINOv2's
difference versus EfficientNet is +0.00214, with a paired conditional 95% bootstrap
interval of [-0.06895, +0.06890]. Whole-image DINOv2's difference is -0.03916,
with interval [-0.11721, +0.03305]. These intervals resample patient groups within
stage from fixed out-of-fold predictions; they do not include retraining
variability or establish external validity. They support neither a superiority
claim nor a formal equivalence claim.

Regional DINOv2 has the highest observed accuracy and lowest ordinal MAE.
EfficientNet has the highest balanced accuracy and quadratic weighted kappa.
Regional DINOv2 improves observed pooled macro F1 over whole-image DINOv2, but a
dedicated paired uncertainty analysis for that contrast was not performed.

## Stage recall

| Stage | Support | EfficientNet | DINOv2 whole | DINOv2 + regions |
| --- | ---: | ---: | ---: | ---: |
| I | 10 | 2/10 (20.0%) | 0/10 (0.0%) | 1/10 (10.0%) |
| II | 53 | 28/53 (52.8%) | 26/53 (49.1%) | 27/53 (50.9%) |
| III | 140 | 79/140 (56.4%) | 94/140 (67.1%) | 97/140 (69.3%) |
| IV | 49 | 38/49 (77.6%) | 33/49 (67.3%) | 34/49 (69.4%) |

Frozen DINOv2 did not solve the rare early-stage problem. Regional DINOv2's
accuracy advantage mainly reflects better Stage III recognition; Stage III is
the largest class. Its fold macro F1 varies from 0.3690 to 0.6314, while
EfficientNet varies from 0.3590 to 0.5456. Do not interpret the pooled near-tie as
evidence of stable performance across samples.

## Recommended thesis interpretation

Retain all three experiments and proceed with an exploratory small-data
comparison. The hypothesis that frozen pretrained features improve balanced
four-stage prediction was not demonstrated in these experiments. The results
remain useful for studying the limits of transfer learning with scarce and
imbalanced expert labels; they do not demonstrate a reliable diagnostic system.

Draft the introduction, research questions, and methodology around comparison,
not an assumed improvement. Include a Results and Discussion section before the
Conclusion. Describe the target as professor-assigned image-based stages within
the annotated BRAR subset. Do not imply full clinical staging or progression
grading was validated. Metadata models and ordinal heads were not tested.

Before finalizing the dataset description, clarify annotation provenance and
the discrepancy between the ten Stage I labels and their BRAR bone-resorption
values (all at least 15%). Preserve both annotation sources pending clarification;
the discrepancy alone does not establish which is correct. All labeled images
remain in original BRAR level_3. The default assumes one patient per image ID;
the audit does not independently establish this or exclude near-duplicate images.

The historical 73.7% accuracy came from a different 38-image split and a different
initialization history, so it is not directly comparable to these five-fold
results. Keep it labeled as a historical pilot rather than selecting it as the
main result because its accuracy is higher.

Further broad model searches would use already-observed outer results to guide
development. Any such extensions should be disclosed as exploratory. They are
not necessary simply to replace an unfavorable result.

## Files and checks

- `artifacts/evaluation/comparison.csv`: pooled comparison.
- `artifacts/evaluation/*_oof.csv`: one held-out prediction per image per method.
- `artifacts/evaluation/paired_comparisons.json`: paired conditional intervals.
- `artifacts/evaluation/*.json`: stage metrics and confusion matrices.
- `artifacts/checks/trained_checkpoints.json`: checkpoint reproduction audit.

An evaluation-only fix normalizes already-validated probability rows to remove
float32 CSV rounding (maximum observed sum error 1.175e-7). It does not change
predicted classes, macro F1, accuracy, or trained models. All nine regression
tests passed after the fix; the evaluation was regenerated.
