from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor, nn
from torchvision.models import resnet18

try:
    from torchvision.models import ResNet18_Weights
except ImportError:  # pragma: no cover
    ResNet18_Weights = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FeatureSpec:
    names: Tuple[str, ...] = ("layer1", "layer2", "layer3")
    channels: Tuple[int, ...] = (64, 128, 256)


class ResNet18FeatureExtractor(nn.Module):
    """ResNet18 backbone that returns intermediate feature maps."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        trainable: bool = True,
        out_layers: Iterable[str] = ("layer1", "layer2", "layer3"),
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            if ResNet18_Weights is not None:
                weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_layers = tuple(out_layers)
        self.feature_spec = FeatureSpec()

        if not trainable:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def train(self, mode: bool = True) -> "ResNet18FeatureExtractor":
        super().train(mode)
        # Keep BN layers frozen when the backbone is not trainable.
        if not any(p.requires_grad for p in self.parameters()):
            super().train(False)
        return self

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64x64 for 256x256 inputs
        x = self.layer1(x)
        if "layer1" in self.out_layers:
            out["layer1"] = x
        x = self.layer2(x)
        if "layer2" in self.out_layers:
            out["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.out_layers:
            out["layer3"] = x
        x = self.layer4(x)
        if "layer4" in self.out_layers:
            out["layer4"] = x
        return out


def flatten_views(x: Tensor) -> Tuple[Tensor, int, int]:
    if x.ndim != 5:
        raise ValueError(f"Expected [B, V, C, H, W], got {tuple(x.shape)}")
    b, v, c, h, w = x.shape
    return x.view(b * v, c, h, w), b, v


def unflatten_views(x: Tensor, batch_size: int, num_views: int) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B*V, C, H, W], got {tuple(x.shape)}")
    _, c, h, w = x.shape
    return x.view(batch_size, num_views, c, h, w)

