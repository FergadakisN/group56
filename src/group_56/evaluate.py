"""
Evaluation module for the M7 Project.

This script provides utilities to load a trained model checkpoint and
perform a final evaluation on the test or validation datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Annotated,  # Add this import at the top
    Any,
)

import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

from .data import DataConfig, make_dataloaders
from .model import build_resnet


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    """
    Computes the loss and accuracy of the model on a given dataset.

    Args:
        model: The trained PyTorch model.
        loader: DataLoader for the dataset split to evaluate.
        device: The hardware device (cpu, cuda, or mps).
        criterion: Optional loss function. If None, loss is returned as NaN.

    Returns:
        A tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits: torch.Tensor = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

    acc = correct / total if total > 0 else 0.0
    if criterion is None:
        return float("nan"), acc

    avg_loss = total_loss / total
    return avg_loss, acc


def resolve_device(device: str) -> torch.device:
    """
    Translates a device string into a torch.device object.

    Args:
        device: Device string ('auto', 'cpu', 'cuda', 'mps').

    Returns:
        The resolved torch.device.
    """
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    """
    Loads a saved PyTorch checkpoint from disk.

    Args:
        ckpt_path: Path to the .pt or .pth file.
        device: The device to map the saved weights to.

    Returns:
        A dictionary containing the state_dict and metadata.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint: dict[str, Any] = torch.load(ckpt_path, map_location=device)
    return checkpoint


def main(
    processed_dir: str = "data/processed",
    arch: str = "resnet18",
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
    # Change these two lines:
    ckpt_path: Annotated[str, typer.Option(help="Path to checkpoint (.pt)")] = "outputs/resnet_run/best.pt",
    split: Annotated[str, typer.Option(help="Split to evaluate: test | val")] = "test",
    compute_loss: bool = True,
) -> None:
    # Now Path(ckpt_path) will correctly receive a string
    dev = resolve_device(device)
    ckpt = load_checkpoint(Path(ckpt_path), dev)
    typer.echo(f"Using device: {dev}")

    # Load checkpoint metadata
    ckpt = load_checkpoint(Path(ckpt_path), dev)
    ckpt_arch: str = ckpt.get("arch", arch)
    class_to_idx_ckpt: dict[str, int] | None = ckpt.get("class_to_idx")

    # Determine number of classes
    num_classes: int | None = ckpt.get("num_classes")
    if num_classes is None and class_to_idx_ckpt is not None:
        num_classes = len(class_to_idx_ckpt)

    if num_classes is None:
        raise ValueError("num_classes not found in checkpoint.")

    # Prepare DataLoaders
    data_cfg = DataConfig(
        processed_dir=processed_dir,
        arch=ckpt_arch,
        batch_size=batch_size,
        num_workers=num_workers,
        rebuild_processed=False,
        wipe_output_dir=False,
    )
    loaders = make_dataloaders(data_cfg)
    train_loader, val_loader, test_loader, class_to_idx_data = loaders

    # Select Split
    split_map = {"train": train_loader, "val": val_loader, "validation": val_loader, "test": test_loader}

    if split not in split_map:
        raise ValueError(f"Invalid split '{split}'. Use train, validation, or test.")

    loader = split_map[split]

    # Warning for class index mismatch
    if class_to_idx_ckpt is not None and class_to_idx_ckpt != class_to_idx_data:
        typer.echo("Warning: Checkpoint class mapping differs from dataset mapping!")

    # Initialize and load model
    model = build_resnet(
        num_classes=num_classes,
        arch=ckpt_arch,
        pretrained=False,
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss() if compute_loss else None

    # Run Evaluation
    loss, acc = evaluate(model=model, loader=loader, device=dev, criterion=criterion)

    result_str = f"{split} acc: {acc:.4f}"
    if compute_loss:
        result_str = f"{split} loss: {loss:.4f} | " + result_str

    typer.echo(result_str)


if __name__ == "__main__":
    typer.run(main)
