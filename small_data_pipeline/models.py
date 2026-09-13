from __future__ import annotations

import timm
import torch
from timm.data import resolve_model_data_config
from torch import nn

from .data import image_transform


def make_dino(cfg, pretrained=True):
    settings = cfg['dino']
    if settings['image_size'] % 14:
        raise ValueError('DINOv2 image size must be divisible by 14.')
    model = timm.create_model(settings['model'], pretrained=pretrained, num_classes=0,
                              img_size=settings['image_size'])
    model.requires_grad_(False)
    model.eval()
    return model


class EfficientNet(nn.Module):
    """Same head design as the original project, with external ImageNet initialization."""
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        c = cfg['efficientnet']
        self.encoder = timm.create_model(c['model'], pretrained=pretrained, num_classes=0, global_pool='avg')
        self.head = nn.Sequential(nn.Dropout(c['dropout']), nn.Linear(self.encoder.num_features, 256),
                                  nn.ReLU(inplace=True), nn.Dropout(c['dropout'] * .5), nn.Linear(256, 4))

    def forward(self, images):
        return self.head(self.encoder(images))


def transform_for(model, size, train=False):
    data = resolve_model_data_config(model.encoder if isinstance(model, EfficientNet) else model)
    return image_transform(size, data['mean'], data['std'], train=train)


def pool_features(features, mode):
    """N,V,D -> N,D global or N,2D global + mean regional representation."""
    if features.ndim != 3:
        raise ValueError('Expected N x views x feature dimension.')
    if mode == 'global':
        return features[:, 0]
    if mode == 'global_regions' and features.shape[1] == 4:
        return torch.cat((features[:, 0], features[:, 1:].mean(1)), dim=1)
    raise ValueError('Regional features require all four views.')
