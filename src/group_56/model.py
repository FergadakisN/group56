# src/project_name/models.py
import torch 
import torch.nn as nn
from torchvision import models


def build_resnet(
    num_classes: int,
    arch: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    unfreeze_from: str | None = None,  # e.g. "layer4" for partial fine-tune
):
    # Pick weights
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
        raise ValueError(f"Unknown arch: {arch}")

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze logic
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    # Partial unfreeze (common: unfreeze layer4 + fc)
    if unfreeze_from is not None:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(unfreeze_from) or name.startswith("fc")

    return model


def resnet_preprocess(arch: str = "resnet18"):
    """Return the torchvision-recommended preprocessing for the chosen weights."""
    if arch == "resnet18":
        return models.ResNet18_Weights.DEFAULT.transforms()
    if arch == "resnet34":
        return models.ResNet34_Weights.DEFAULT.transforms()
    if arch == "resnet50":
        return models.ResNet50_Weights.DEFAULT.transforms()
    raise ValueError(f"Unknown arch: {arch}")
