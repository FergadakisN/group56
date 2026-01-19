"""
Training and validation orchestration for the M7 Project.

This script handles the training lifecycle, including device selection,
mixed-precision training (AMP), validation, and checkpointing.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from typing import Annotated, Optional
import typer
import wandb
from model import build_resnet
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from .data import DataConfig, make_dataloaders

# ============================================================
# CONFIGS
# ============================================================

@dataclass(frozen=True)
class TrainConfig:
    """Configuration for model training hyperparameters and environment."""

    arch: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    unfreeze_from: str | None = None
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    device: str = "auto"   # auto | cpu | cuda | mps
    amp: bool = True
    seed: int = 42
    out_dir: str = "outputs"
    run_name: str = "resnet_run"


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int) -> None:
    """
    Sets the seed for all relevant libraries to ensure reproducibility.

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """
    Resolves a string identifier to a torch.device object.

    Args:
        device: 'cpu', 'cuda', 'mps', or 'auto'.

    Returns:
        The resolved torch.device.
    """
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Auto-detection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    epoch: int,
    arch: str,
    num_classes: int,
) -> None:
    """
    Saves the model state and metadata to a checkpoint file.

    Args:
        path: Destination Path for the checkpoint.
        model: The model whose state_dict will be saved.
        class_to_idx: Mapping of class names to indices.
        epoch: The current epoch index.
        arch: The architecture name string.
        num_classes: Number of classes in the classifier head.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "arch": arch,
            "num_classes": num_classes,
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
        },
        path,
    )


# ============================================================
# TRAINING / VALIDATION
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> Tuple[float, float]:
    """
    Runs one full training epoch.

    Args:
        model: The network to train.
        loader: Training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: Hardware device to use.
        scaler: GradScaler for AMP.
        use_amp: Whether to use mixed precision.

    Returns:
        A tuple containing (average_loss, accuracy).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The network to evaluate.
        loader: Validation DataLoader.
        criterion: The loss function.
        device: Hardware device to use.

    Returns:
        A tuple containing (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
# CLI ENTRYPOINT
# ============================================================

def main(
    processed_dir: str = "data/processed",
    batch_size: int = 32,
    num_workers: int = 4,
    arch: str = "resnet18",
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    unfreeze_from: Annotated[Optional[str], typer.Option(help="Layer name to unfreeze")] = None,
    device: str = "auto",
    amp: bool = True,
    seed: int = 42,
    out_dir: str = "outputs",
    run_name: str = "resnet_run",
    ckpt_name: str = "last.pt",
    save_best: bool = True,
) -> None:
    """
    Starts the training and validation process via the command line.
    """
    set_seed(seed)
    dev = resolve_device(device)
    typer.echo(f"Using device: {dev}")

    # W&B init
    run = wandb.init(
        project="group56-fish",   # change to your project name
        name=run_name,
        config={
            "processed_dir": processed_dir,
            "arch": arch,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "unfreeze_from": unfreeze_from,
            "amp": amp,
            "seed": seed,
            "ckpt_name": ckpt_name,
            "save_best": save_best,
        },
    )

    cfg = wandb.config
    lr = cfg.get("lr", lr)
    batch_size = cfg.get("batch_size", batch_size)
    weight_decay = cfg.get("weight_decay", weight_decay)

    freeze_backbone = cfg.get("freeze_backbone", freeze_backbone)
    unfreeze_from = cfg.get("unfreeze_from", unfreeze_from)

    if freeze_backbone:
        unfreeze_from = None



    # Data Initialization
    data_cfg = DataConfig(
        processed_dir=processed_dir,
        arch=arch,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader, val_loader, _, class_to_idx = make_dataloaders(data_cfg)
    num_classes = len(class_to_idx)

    # Model Initialization
    model = build_resnet(
        num_classes=num_classes,
        arch=arch,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        unfreeze_from=unfreeze_from,
    ).to(dev)

    # Optimization Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and dev.type == "cuda"))

    # Path Setup
    out_path = Path(out_dir) / run_name
    out_path.mkdir(parents=True, exist_ok=True)

    last_ckpt = out_path / ckpt_name
    best_ckpt = out_path / "best.pt"
    best_val_acc = -1.0

    # Training Loop
    #[Image of deep learning training process flowchart]
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=dev,
            scaler=scaler,
            use_amp=amp,
        )

        va_loss, va_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=dev,
        )

        typer.echo(
            f"Epoch {epoch:02d}/{epochs} | "
            f"tr_loss: {tr_loss:.4f} | tr_acc: {tr_acc:.4f} | "
            f"va_loss: {va_loss:.4f} | va_acc: {va_acc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": va_loss,
                "val/acc": va_acc,
            }
        )

        save_checkpoint(
            path=last_ckpt,
            model=model,
            class_to_idx=class_to_idx,
            epoch=epoch,
            arch=arch,
            num_classes=num_classes,
        )

        if save_best and va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(
                path=best_ckpt,
                model=model,
                class_to_idx=class_to_idx,
                epoch=epoch,
                arch=arch,
                num_classes=num_classes,
            )
            typer.echo(f"New best: {best_ckpt} (acc={best_val_acc:.4f})")

        wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["best_epoch"] = epoch

    typer.echo("Training process complete.")
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
