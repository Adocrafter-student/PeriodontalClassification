"""Non-training checks: no fit(), backward(), optimizer step, or BatchNorm updates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import device_for, read_prepared, runtime, write_json
from .data import Images, inner_splits, validation_split
from .features import read_features
from .models import EfficientNet, make_dino, pool_features, transform_for


def check(cfg, offline=False):
    samples, manifest = read_prepared(cfg)
    device = device_for(cfg)
    report = dict(runtime=runtime(), device=str(device), pretrained_weights=not offline,
                  training_performed=False, samples=len(samples), folds=[])
    for fold in range(cfg['outer_folds']):
        dev = samples[samples.fold != fold].reset_index(drop=True)
        splits = inner_splits(dev, cfg['inner_folds'], cfg['seed'] + fold)
        a, b = validation_split(dev, cfg['efficientnet']['validation_fraction'], cfg['seed'] + fold)
        report['folds'].append(dict(fold=fold, development=len(dev),
                                   test=int((samples.fold == fold).sum()),
                                   selection_train=len(a), selection_val=len(b), inner_splits=len(splits)))
    for name in ('dino', 'efficientnet'):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model = (make_dino(cfg, pretrained=not offline) if name == 'dino'
                 else EfficientNet(cfg, pretrained=not offline).eval()).to(device)
        mode = cfg['dino']['views'] if name == 'dino' else 'global'
        dataset = Images(samples.iloc[:cfg[name]['batch_size']], cfg['dataset']['root'],
                         transform_for(model, cfg[name]['image_size']), mode)
        loader = DataLoader(dataset, batch_size=cfg[name]['batch_size'], num_workers=cfg['num_workers'])
        x, labels = next(iter(loader))
        # Include buffers: this catches accidental running-stat changes as well as parameter updates.
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            outputs = torch.stack([model(x[:, j].to(device)) for j in range(x.shape[1])], dim=1)
            if name == 'dino':
                assert not any(p.requires_grad for p in model.parameters())
                features = pool_features(outputs, mode)
                assert features.shape[0] == len(labels)
            else:
                assert outputs.shape == (len(labels), 1, 4)
                assert torch.isfinite(torch.nn.functional.cross_entropy(outputs[:, 0], labels.to(device)))
        assert torch.isfinite(outputs).all()
        assert all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items())
        report[name] = dict(input_shape=list(x.shape), output_shape=list(outputs.shape),
                            state_unchanged=True,
                            peak_cuda_allocated_mb=round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                            if device.type == 'cuda' else None)
        print(f'{name}: {report[name]}', flush=True)
        del model, outputs, before, x
        if name == 'dino':
            del features
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if (Path(cfg['output']) / 'features' / 'manifest.json').exists():
        cached = read_features(cfg, samples, manifest, 'global')
        report['feature_cache'] = dict(shape=list(cached.shape), finite=bool(np.isfinite(cached).all()))
    write_json(Path(cfg['output']) / 'checks' / ('offline.json' if offline else 'preflight.json'), report)
    print('Checks passed. No training or parameter updates performed.')
