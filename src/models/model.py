"""Model definition: a timm pretrained backbone fine-tuned for crop disease classes."""

import timm
import torch.nn as nn


def build_model(
    num_classes: int,
    backbone: str = "mobilenetv3_small_100",
    pretrained: bool = True,
) -> nn.Module:
    """Builds a classifier by swapping the head of a pretrained timm backbone.

    Good backbone choices for this use case (small, fast, mobile/edge-friendly):
        - mobilenetv3_small_100
        - efficientnet_lite0
        - resnet18 (heavier, higher accuracy ceiling)
    """
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
