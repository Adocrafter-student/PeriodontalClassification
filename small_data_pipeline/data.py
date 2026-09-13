from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

from .common import STAGES, file_hash, source_signature, write_json


def image_id(name):
    match = re.search(r'patient_image_(\d+)_', str(name))
    if not match:
        raise ValueError(f'Cannot parse image ID: {name}')
    return int(match[1])


def assert_partition(df, train, test):
    if set(df.iloc[train].group_id) & set(df.iloc[test].group_id):
        raise ValueError('Patient/duplicate group crosses a split.')
    for idx in (train, test):
        if set(df.iloc[idx].stage_2017) != set(STAGES):
            raise ValueError('A partition lacks a stage. Reduce fold count or review grouping.')


def inner_splits(df, count, seed):
    splitter_type = StratifiedKFold if df.group_id.is_unique else StratifiedGroupKFold
    splitter = splitter_type(n_splits=count, shuffle=True, random_state=seed)
    splits = list(splitter.split(df, df.stage_2017) if df.group_id.is_unique
                  else splitter.split(df, df.stage_2017, df.group_id))
    for train, val in splits:
        assert_partition(df, train, val)
    return splits


def validation_split(df, fraction, seed):
    # Stratify unique groups so repeated patient images cannot cross boundaries.
    groups = df.groupby('group_id', sort=True).stage_2017.agg(['first', 'nunique'])
    if (groups['nunique'] != 1).any():
        raise ValueError('Mixed-stage patient groups need a custom validation design.')
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    a, b = next(splitter.split(groups, groups['first']))
    train = np.flatnonzero(df.group_id.isin(groups.index[a]).to_numpy())
    val = np.flatnonzero(df.group_id.isin(groups.index[b]).to_numpy())
    assert_partition(df, train, val)
    return train, val


def prepare(cfg):
    out = Path(cfg['output']) / 'data'
    if out.exists():
        raise FileExistsError(f'{out} exists. Prepared folds are immutable; choose a new output path.')
    d = cfg['dataset']
    labels = pd.read_csv(d['labels'], sep=';', header=None, names=['image_id', 'stage_2017'])
    for col in labels:
        values = pd.to_numeric(labels[col], errors='raise')
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f'Invalid integer values in {col}')
        labels[col] = values.astype(int)
    if labels.image_id.duplicated().any() or not labels.stage_2017.isin(STAGES).all():
        raise ValueError('Duplicate label IDs or stages outside 1..4.')
    metadata = pd.read_csv(d['metadata'])
    metadata['image_id'] = metadata['File name'].map(image_id)
    if metadata.image_id.duplicated().any():
        raise ValueError('Duplicate metadata IDs.')
    records = []
    for path in sorted(Path(d['root']).glob('level_*/*.jpg')):
        records.append(dict(image_id=image_id(path.name), image_path=path.relative_to(d['root']).as_posix(),
                            original_folder=path.parent.name))
    images = pd.DataFrame(records)
    if images.empty or images.image_id.duplicated().any():
        raise ValueError('No images or duplicate image IDs.')
    df = labels.merge(images, on='image_id', how='left', validate='one_to_one')
    df = df.merge(metadata, on='image_id', how='left', validate='one_to_one')
    if df[['image_path', 'File name']].isna().any().any():
        raise ValueError('A professor label lacks an image or metadata row.')
    group_map = None
    if d.get('patient_groups'):
        group_map = pd.read_csv(d['patient_groups'], dtype={'patient_id': str})
        df = df.merge(group_map[['image_id', 'patient_id']], on='image_id', how='left', validate='one_to_one')
        if df.patient_id.isna().any():
            raise ValueError('Patient mapping does not cover every labeled image.')
    hashes, files, widths, heights = [], [], [], []
    for row in df.itertuples():
        path = Path(d['root']) / row.image_path
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            widths.append(im.width)
            heights.append(im.height)
            hashes.append(hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest())
        files.append(file_hash(path))
    df['pixel_sha256'], df['file_sha256'] = hashes, files
    df['width'], df['height'] = widths, heights
    df['group_id'] = df.patient_id if group_map is not None else df.image_id.astype(str)
    # Exact duplicate images are conservatively rejected, including conflicting labels.
    if df.pixel_sha256.duplicated().any():
        raise ValueError('Duplicate decoded images found; resolve duplicates before preparing folds.')
    if (df.groupby('group_id').stage_2017.nunique() > 1).any():
        raise ValueError('Mixed-stage patient groups require review before this experiment.')
    df = df.sort_values('image_id').reset_index(drop=True)
    df['fold'] = -1
    splits = inner_splits(df, cfg['outer_folds'], cfg['seed'])
    for fold, (train, test) in enumerate(splits):
        df.loc[test, 'fold'] = fold
        development = df.iloc[train].reset_index(drop=True)
        inner_splits(development, cfg['inner_folds'], cfg['seed'] + fold)
        validation_split(development, cfg['efficientnet']['validation_fraction'], cfg['seed'] + fold)
    audit = dict(images_on_disk=len(images), metadata_rows=len(metadata), labeled=len(df),
                 without_stage_label=int((~images.image_id.isin(labels.image_id)).sum()),
                 stage_counts=df.stage_2017.value_counts().sort_index().to_dict(),
                 folder_counts=df.original_folder.value_counts().to_dict(),
                 folds=pd.crosstab(df.fold, df.stage_2017).to_dict(),
                 grouping='patient mapping' if group_map is not None else 'image ID assumed to identify one patient',
                 stage_I_metadata_loss_at_least_15pct=int(((df.stage_2017 == 1) & (df['Bone resorption'] >= .15)).sum()),
                 note='Metadata retained for audit only; all implemented models are image-only. '
                      'Pixel hashes detect exact duplicates, not near-duplicates or unrecorded repeat patients.')
    out.mkdir(parents=True)
    df.to_csv(out / 'samples.csv', index=False)
    write_json(out / 'audit.json', audit)
    write_json(out / 'manifest.json', dict(sources=source_signature(cfg),
               samples_sha256=file_hash(out / 'samples.csv'),
               split_settings={k: cfg[k] for k in ('seed', 'outer_folds', 'inner_folds')}))
    print(json.dumps(audit, indent=2))


class Letterbox:
    """Keep the entire field of view and aspect ratio; pad with dark pixels."""
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))


def views(image, mode):
    if mode == 'global':
        return [image]
    if mode != 'global_regions':
        raise ValueError(f'Unknown view mode: {mode}')
    side = min(image.size)
    boxes = [(round((image.width - side) * fraction), (image.height - side) // 2)
             for fraction in (0, .5, 1)]
    return [image] + [image.crop((x, y, x + side, y + side)) for x, y in boxes]


def image_transform(size, mean, std, train=False):
    steps = [Letterbox(size)]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                  transforms.ColorJitter(brightness=.12, contrast=.12)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


class Images(Dataset):
    def __init__(self, df, root, transform, mode='global'):
        self.df, self.root, self.transform, self.mode = df.reset_index(drop=True), Path(root), transform, mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(self.root / row.image_path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            tensors = torch.stack([self.transform(v) for v in views(image, self.mode)])
        return tensors, int(row.stage_2017) - 1
