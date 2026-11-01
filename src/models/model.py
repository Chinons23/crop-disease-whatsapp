"""Classifier architectures used by training and inference."""

import torch
import timm
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class TinyVGGClassifier(nn.Module):
    """Small VGG-style CNN used as a from-scratch comparison baseline."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.block_1(images)
        features = self.block_2(features)
        return self.classifier(self.pool(features))


def build_mobilenet_classifier(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build torchvision MobileNetV3-Small with a task-specific classifier head."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(input_features, num_classes),
    )
    return model


def build_model(
    num_classes: int,
    backbone: str = "mobilenetv3_small_100",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a classifier from either the project's timm or torchvision backbones.

    Good backbone choices for this use case (small, fast, mobile/edge-friendly):
        - mobilenetv3_small_100
        - efficientnet_lite0
        - resnet18 (heavier, higher accuracy ceiling)
        - mobilenet_v3_small (torchvision MobileNetV3-Small)
        - tiny_vgg (from-scratch experiment baseline)
    """
    if backbone == "tiny_vgg":
        return TinyVGGClassifier(num_classes)
    if backbone == "mobilenet_v3_small":
        return build_mobilenet_classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
