import sys
from pathlib import Path

import pytest
import torch

from tests import SRC_ROOT

sys.path.insert(0, str(SRC_ROOT))

from group_56.model import build_resnet
from group_56.train import resolve_device, save_checkpoint, set_seed, train_one_epoch, validate_one_epoch


def test_build_resnet_output_shape() -> None:
    """Ensure model outputs match the requested number of classes."""
    model = build_resnet(num_classes=10, arch="resnet18", pretrained=False)
    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
    assert outputs.shape == (2, 10)


def test_freeze_backbone_only_fc_trainable() -> None:
    """Ensure backbone parameters are frozen while the head stays trainable."""
    model = build_resnet(num_classes=5, arch="resnet18", pretrained=False, freeze_backbone=True)
    for name, param in model.named_parameters():
        if name.startswith("fc"):
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_unfreeze_from_layer() -> None:
    """Ensure only the specified layer prefix and head are trainable."""
    model = build_resnet(
        num_classes=5,
        arch="resnet18",
        pretrained=False,
        freeze_backbone=True,
        unfreeze_from="layer4",
    )
    has_layer4 = False
    for name, param in model.named_parameters():
        if name.startswith("layer4"):
            has_layer4 = True
            assert param.requires_grad
        elif name.startswith("fc"):
            assert param.requires_grad
        else:
            assert not param.requires_grad
    assert has_layer4


def test_invalid_arch_raises() -> None:
    """Ensure unsupported architectures raise a ValueError."""
    with pytest.raises(ValueError):
        build_resnet(num_classes=3, arch="resnet101", pretrained=False)


def test_resolve_device_cpu() -> None:
    """Ensure CPU device resolution returns a CPU device."""
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_save_checkpoint_writes_expected_keys(tmp_path: Path) -> None:
    """Ensure save_checkpoint writes required metadata."""
    model = build_resnet(num_classes=2, arch="resnet18", pretrained=False)
    checkpoint_path = tmp_path / "ckpt.pt"
    class_to_idx = {"class_a": 0, "class_b": 1}

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        class_to_idx=class_to_idx,
        epoch=3,
        arch="resnet18",
        num_classes=2,
    )

    data = torch.load(checkpoint_path, map_location="cpu")
    assert data["epoch"] == 3
    assert data["arch"] == "resnet18"
    assert data["num_classes"] == 2
    assert data["class_to_idx"] == class_to_idx
    assert "model_state_dict" in data


def test_train_one_epoch_updates_parameters() -> None:
    """Ensure a training step updates model parameters."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    inputs = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.5, 0.5], [0.5, 0.5]]]])
    labels = torch.tensor([0, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels),
        batch_size=2,
        shuffle=False,
    )

    before = [p.detach().clone() for p in model.parameters()]
    loss, acc = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        scaler=None,
        use_amp=False,
    )
    after = list(model.parameters())

    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=False))


def test_set_seed_deterministic() -> None:
    """Ensure set_seed makes torch RNG deterministic."""
    set_seed(123)
    first = torch.rand(3)
    set_seed(123)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_validate_one_epoch_basic() -> None:
    """Ensure validation loop returns expected metrics."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2, bias=False))
    torch.nn.init.zeros_(model[1].weight)
    criterion = torch.nn.CrossEntropyLoss()
    inputs = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.5, 0.5], [0.5, 0.5]]]])
    labels = torch.tensor([0, 0])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels),
        batch_size=2,
        shuffle=False,
    )

    loss, acc = validate_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert loss >= 0.0
    assert acc == 1.0
