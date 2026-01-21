"""
Model architecture module for the M7 Project.

This module provides a flexible factory function to build ResNet models
with custom classification heads and support for various freezing and
fine-tuning strategies.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models  # type: ignore[import-untyped]


def build_resnet(
    num_classes: int,
    arch: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    unfreeze_from: str | None = None,
) -> nn.Module:
    """
    Constructs a ResNet model with a custom classification head.

    Args:
        num_classes: The number of output neurons for the final linear layer.
        arch: The ResNet variant to use ('resnet18', 'resnet34', 'resnet50').
        pretrained: If True, loads weights trained on ImageNet.
        freeze_backbone: If True, sets all layers except the head to non-trainable.
        unfreeze_from: The name of the layer from which training should begin
            (e.g., 'layer4'). If set, overrides freeze_backbone for these layers.

    Returns:
        A PyTorch nn.Module representing the configured ResNet model.

    Raises:
        ValueError: If an unsupported architecture string is provided.
    """
    # 1. Architecture selection and weight loading
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Supported: resnet18, resnet34, resnet50")

    # 2. Classifier head replacement
    # ResNet models use 'fc' (Fully Connected) as the final layer name
    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 3. Backbone freezing logic
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Ensure the custom head is always trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    # 4. Selective unfreezing for fine-tuning
    if unfreeze_from is not None:
        for name, param in model.named_parameters():
            # Enable gradients if the layer matches the unfreeze prefix or is the head
            should_unfreeze = name.startswith(unfreeze_from) or name.startswith("fc")
            param.requires_grad = should_unfreeze

    return model
