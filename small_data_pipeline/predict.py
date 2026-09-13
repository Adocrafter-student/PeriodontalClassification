from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import device_for, experiment_signature, file_hash, read_prepared
from .data import Images
from .features import encode, read_features
from .models import EfficientNet, make_dino, pool_features, transform_for


@torch.inference_mode()
def predict(cfg, method, input_path):
    samples, manifest = read_prepared(cfg)
    path = Path(input_path).resolve()
    if not path.is_file():
        raise ValueError('Supply one panoramic image file.')
    device = device_for(cfg)
    is_dino = method.startswith('dino_')
    mode = method.removeprefix('dino_') if is_dino else 'global'
    folder = Path(cfg['output'])
    if is_dino:
        read_features(cfg, samples, manifest, mode)  # provenance and exact encoder integrity
        model = make_dino(cfg, pretrained=False)
        model.load_state_dict(torch.load(folder / 'features' / 'encoder.pt', map_location='cpu', weights_only=True))
    else:
        model = EfficientNet(cfg, pretrained=False)
    model = model.eval().to(device)
    dummy = pd.DataFrame([dict(image_path=path.name, stage_2017=1)])
    dataset = Images(dummy, path.parent, transform_for(model, cfg['dino' if is_dino else 'efficientnet']['image_size']), mode)
    loader = DataLoader(dataset, batch_size=1)
    if is_dino:
        raw = encode(model, loader, device)
        features = pool_features(torch.from_numpy(raw), mode).numpy()
    probabilities = []
    for fold in range(cfg['outer_folds']):
        run = folder / 'runs' / method / f'fold_{fold}'
        info = json.loads((run / 'complete.json').read_text())
        if info['signature'] != experiment_signature(cfg, manifest, method):
            raise ValueError('Model does not match current experiment settings.')
        if is_dino:
            classifier = joblib.load(run / 'classifier.joblib')
            if not np.array_equal(classifier.classes_, [1, 2, 3, 4]):
                raise ValueError('Unexpected classifier class order.')
            p = classifier.predict_proba(features)[0]
        else:
            model.load_state_dict(torch.load(run / 'model.pt', map_location=device, weights_only=True))
            x, _ = next(iter(loader))
            p = model(x[:, 0].to(device)).softmax(1)[0].cpu().numpy()
        probabilities.append(p)
    mean = np.mean(probabilities, axis=0)
    print(json.dumps(dict(predicted_stage=int(mean.argmax() + 1),
                         probabilities={str(i + 1): float(p) for i, p in enumerate(mean)},
                         known_dataset_image=bool((samples.file_sha256 == file_hash(path)).any()),
                         note='Mean of five fold models; uncalibrated probabilities. This ensemble is not '
                              'the out-of-fold evaluation and must not be used to score training images.'), indent=2))
