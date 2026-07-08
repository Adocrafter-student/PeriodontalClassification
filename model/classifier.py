"""EfficientNet classifier for 2017 periodontal staging.

The default path is image-only. The age/gender metadata branch remains
available for explicit ablation experiments.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class BRARClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = False,
        num_classes: int = 4,
        dropout: float = 0.4,
        use_metadata: bool = False,
        metadata_dim: int = 2,
    ) -> None:
        super().__init__()
        self.use_metadata = use_metadata

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        embed_dim = self.encoder.num_features

        if use_metadata:
            self.meta_fc = nn.Sequential(
                nn.Linear(metadata_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
            head_in = embed_dim + 32
        else:
            self.meta_fc = None
            head_in = embed_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encoder(image)

        if self.use_metadata:
            if metadata is None:
                raise ValueError("metadata tensor is required when use_metadata=True")
            meta_features = self.meta_fc(metadata)
            features = torch.cat([features, meta_features], dim=1)

        return self.head(features)
