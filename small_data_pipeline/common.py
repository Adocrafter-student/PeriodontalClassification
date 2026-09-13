from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, log_loss,
)

HERE = Path(__file__).resolve().parent
STAGES = [1, 2, 3, 4]
RECIPE_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_config(path=None):
    path = Path(path or HERE / 'config.yaml').resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    for key in ('root', 'labels', 'metadata', 'patient_groups'):
        if cfg['dataset'].get(key):
            cfg['dataset'][key] = str((path.parent / cfg['dataset'][key]).resolve())
    cfg['output'] = str((path.parent / cfg['output']).resolve())
    cache = Path(cfg['output']) / 'cache'
    os.environ['TORCH_HOME'] = str(cache / 'torch')
    os.environ['HF_HOME'] = str(cache / 'huggingface')
    return cfg


def device_for(cfg):
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Activate your CUDA PyTorch environment; '
                           'use --device cpu only for diagnostic checks.')
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime():
    import platform
    import sklearn
    import timm
    import torchvision
    return dict(python=platform.python_version(), torch=torch.__version__,
                torchvision=torchvision.__version__, timm=timm.__version__,
                sklearn=sklearn.__version__, numpy=np.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def source_signature(cfg):
    d = cfg['dataset']
    return {k: file_hash(d[k]) for k in ('labels', 'metadata', 'patient_groups') if d.get(k)}


def read_prepared(cfg):
    folder = Path(cfg['output']) / 'data'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['sources'] != source_signature(cfg):
        raise ValueError('Source CSVs changed. Use a new output directory and prepare again.')
    expected = {k: cfg[k] for k in ('seed', 'outer_folds', 'inner_folds')}
    if manifest['split_settings'] != expected:
        raise ValueError('Split settings changed. Use a new output directory.')
    if file_hash(folder / 'samples.csv') != manifest['samples_sha256']:
        raise ValueError('Prepared sample manifest was edited. Prepare in a new output directory.')
    df = pd.read_csv(folder / 'samples.csv', dtype={'group_id': str})
    # Detect image replacement without decoding the dataset at every command.
    root = Path(cfg['dataset']['root'])
    for row in df.itertuples():
        if file_hash(root / row.image_path) != row.file_sha256:
            raise ValueError(f'Image changed: {row.image_path}')
    return df, manifest


def metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 4) or not np.isfinite(p).all():
        raise ValueError('Expected finite N x 4 probabilities.')
    if (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probability distribution.')
    # CSV-loaded float32 softmax values may sum to 1 +/- 1e-7. Normalize only
    # after validation so log_loss does not mistake rounding for invalid scores.
    p = p / p.sum(1, keepdims=True)
    pred = p.argmax(1) + 1
    return dict(
        num_samples=len(y), accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=STAGES, average='macro', zero_division=0)),
        ordinal_mae=float(np.abs(y - pred).mean()),
        quadratic_weighted_kappa=float(cohen_kappa_score(y, pred, labels=STAGES, weights='quadratic')),
        log_loss=float(log_loss(y, p, labels=STAGES)),
        confusion_matrix=confusion_matrix(y, pred, labels=STAGES).tolist(),
        per_stage=classification_report(y, pred, labels=STAGES,
                                       output_dict=True, zero_division=0),
    )


def prediction_frame(samples, probabilities):
    metrics(samples.stage_2017, probabilities)  # validate before writing
    result = samples[['image_id', 'group_id', 'stage_2017', 'fold']].copy()
    result['pred_stage'] = np.asarray(probabilities).argmax(1) + 1
    for j in range(4):
        result[f'prob_stage_{j + 1}'] = probabilities[:, j]
    return result


def experiment_signature(cfg, manifest, method):
    keys = ('dino',) if method.startswith('dino') else ('efficientnet',)
    return digest(dict(recipe=RECIPE_VERSION, manifest=manifest,
                       settings={k: cfg[k] for k in (*keys, 'seed', 'device', 'num_workers')}))
