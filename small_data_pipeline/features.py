from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import (RECIPE_VERSION, device_for, digest, file_hash, read_prepared,
                     runtime, write_json)
from .data import Images
from .models import make_dino, pool_features, transform_for


def feature_signature(cfg, manifest):
    return digest(dict(recipe=RECIPE_VERSION, samples=manifest['samples_sha256'],
                       model=cfg['dino']['model'], size=cfg['dino']['image_size'],
                       views=cfg['dino']['views'], preprocessing='letterbox-model-normalization'))


@torch.inference_mode()
def encode(model, loader, device):
    result = []
    for images, _ in tqdm(loader, desc='Frozen feature extraction'):
        # Process views serially: batch size 4 means four views on GPU, not sixteen.
        batch_views = []
        for index in range(images.shape[1]):
            batch_views.append(model(images[:, index].to(device)).float().cpu())
        result.append(torch.stack(batch_views, dim=1))
    return torch.cat(result).numpy()


def extract(cfg):
    samples, manifest = read_prepared(cfg)
    out = Path(cfg['output']) / 'features'
    if out.exists():
        raise FileExistsError(f'{out} exists. Validate/reuse it or select a new output directory.')
    device = device_for(cfg)
    model = make_dino(cfg).to(device)
    transform = transform_for(model, cfg['dino']['image_size'])
    dataset = Images(samples, cfg['dataset']['root'], transform, cfg['dino']['views'])
    loader = DataLoader(dataset, batch_size=cfg['dino']['batch_size'], shuffle=False,
                        num_workers=cfg['num_workers'], pin_memory=device.type == 'cuda')
    features = encode(model, loader, device)
    if not np.isfinite(features).all():
        raise ValueError('Non-finite image features.')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'dino.npz', image_id=samples.image_id.to_numpy(), features=features)
    # Preserve the exact external encoder used; inference never needs a changed remote checkpoint.
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / 'encoder.pt')
    write_json(out / 'manifest.json', dict(signature=feature_signature(cfg, manifest),
               feature_sha256=file_hash(out / 'dino.npz'), encoder_sha256=file_hash(out / 'encoder.pt'),
               shape=list(features.shape), runtime=runtime()))
    print(f'Saved frozen features {features.shape}. No classifier was fitted.')


def read_features(cfg, samples, manifest, mode):
    folder = Path(cfg['output']) / 'features'
    info = json.loads((folder / 'manifest.json').read_text())
    if info['signature'] != feature_signature(cfg, manifest):
        raise ValueError('Features do not match the current dataset/model/preprocessing.')
    if file_hash(folder / 'dino.npz') != info['feature_sha256']:
        raise ValueError('Feature cache checksum mismatch.')
    if file_hash(folder / 'encoder.pt') != info['encoder_sha256']:
        raise ValueError('Frozen encoder checksum mismatch.')
    with np.load(folder / 'dino.npz', allow_pickle=False) as cache:
        if not np.array_equal(cache['image_id'], samples.image_id.to_numpy()):
            raise ValueError('Feature/sample ordering mismatch.')
        raw = cache['features']
        if not np.isfinite(raw).all():
            raise ValueError('Non-finite cached features.')
        return pool_features(torch.from_numpy(raw), mode).numpy()
