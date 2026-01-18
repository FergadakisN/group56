from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import typer

from data import DataConfig, make_dataloaders
from model import build_resnet


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """
    Returns (avg_loss, accuracy). If criterion is None, loss is returned as NaN.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

    acc = correct / total if total > 0 else 0.0
    if criterion is None:
        return float("nan"), acc
    return total_loss / total, acc


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def main(
    processed_dir: str = "data/processed",
    arch: str = "resnet18",
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
    ckpt_path: str = typer.Option(..., help="Path to checkpoint (.pt), e.g. outputs/run_name/best.pt"),
    split: str = typer.Option("test", help="Which split to evaluate: test | validation | train"),
    compute_loss: bool = True,
):
    """
    Evaluate a saved checkpoint on the selected split.
    Recommended: split='test' and ckpt_path pointing to best.pt.
    """
    dev = resolve_device(device)
    typer.echo(f"Using device: {dev}")

    ckpt = load_checkpoint(Path(ckpt_path), dev)

    # If checkpoint contains arch/num_classes, prefer those (safer)
    ckpt_arch = ckpt.get("arch", arch)
    num_classes = ckpt.get("num_classes", None)
    class_to_idx = ckpt.get("class_to_idx", None)

    if num_classes is None and class_to_idx is not None:
        num_classes = len(class_to_idx)
    if num_classes is None:
        raise ValueError(
            "num_classes not found in checkpoint. Ensure train.py saves it (recommended)."
        )

    # Data (use same arch transforms as training, to match preprocessing)
    data_cfg = DataConfig(
        processed_dir=processed_dir,
        arch=ckpt_arch,
        batch_size=batch_size,
        num_workers=num_workers,
        rebuild_processed=False,
        wipe_output_dir=False,
    )
    train_loader, val_loader, test_loader, class_to_idx_data = make_dataloaders(data_cfg)

    if split == "train":
        loader = train_loader
    elif split in ("val", "validation"):
        loader = val_loader
    elif split == "test":
        loader = test_loader
    else:
        raise ValueError("split must be one of: train, validation, test")

    # Optional safety check: mapping consistency
    if class_to_idx is not None and class_to_idx != class_to_idx_data:
        typer.echo(
            "Warning: class_to_idx in checkpoint differs from current data folder mapping.\n"
            "This can cause incorrect label interpretation if folders changed."
        )

    # Model
    model = build_resnet(
        num_classes=num_classes,
        arch=ckpt_arch,
        pretrained=False,
        freeze_backbone=False,
        unfreeze_from=None,
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss() if compute_loss else None

    loss, acc = evaluate(model=model, loader=loader, device=dev, criterion=criterion)

    if compute_loss:
        typer.echo(f"{split} loss: {loss:.4f} | {split} acc: {acc:.4f}")
    else:
        typer.echo(f"{split} acc: {acc:.4f}")


if __name__ == "__main__":
    typer.run(main)
