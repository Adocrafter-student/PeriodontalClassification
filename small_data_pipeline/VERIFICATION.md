# Verification before training

Completed on 2026-09-12 using the existing Windows Python environment and NVIDIA
GeForce RTX 4060 (8 GB). No classifier fit, backward pass, optimizer step, or
network training was executed.

| Check | Result |
| --- | --- |
| Source audit | 988 images, 252 professor labels, 736 without stage labels |
| Image checks | Every labeled image decoded; no duplicate labeled pixel hashes |
| Shared five folds | Every labeled image assigned once; two Stage I cases per outer test fold |
| Inner partitions | Every training/validation partition contains all four stages; groups disjoint |
| Pretrained DINO CUDA forward | Passed, batch 4, 448 pixels, four views processed serially |
| Pretrained EfficientNet CUDA forward | Passed, batch 4, 448 pixels |
| Model state check | All parameters and buffers unchanged after diagnostic forward passes |
| Frozen feature extraction | Completed for all 252 images; raw shape 252 x 4 x 384 |
| Cache integrity | Encoder/file checksums, image ordering, and both pooled feature shapes verified |
| Regression tests | 8 passed; no model fitting in tests |
| Python compilation / imports | All modules compiled; training and prediction modules imported |
| Training outputs | None created |

Environment: Python 3.12.10, torch 2.6.0+cu124, torchvision 0.21.0+cu124,
timm 1.0.27, scikit-learn 1.5.2, NumPy 2.4.3.

Machine-readable evidence is in `artifacts/checks/preflight.json`,
`artifacts/checks/features.json`, and `artifacts/data/audit.json`. Model download
and frozen-feature caches are already available, so do not repeat `prepare` or
`extract` for this experiment.

Still pending by request: actual classifier fitting, neural-network
backpropagation/optimizer checks, full training memory use, completed-checkpoint
inference, and the real out-of-fold performance comparison. Evaluation logic was
tested with constructed probabilities, not reported as model performance.
