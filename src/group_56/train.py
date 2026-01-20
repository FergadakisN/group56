"""
Training and validation orchestration for the M7 Project.

This script handles the training lifecycle, including device selection,
mixed-precision training (AMP), validation, and checkpointing.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn as nn
import typer
import wandb
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from .data import DataConfig, make_dataloaders
from .model import build_resnet

logger = logging.getLogger(__name__)


def _setup_logging(run_name: str, out_dir: str) -> None:
    """Configure logging to file and console."""

    log_dir = Path(out_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


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
    device: str = "auto"  # auto | cpu | cuda | mps
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
    logger.info(f"Set random seed to {seed}")


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
    resolved = torch.device("cpu")
    logger.debug(f"Resolved device '{device}' to {resolved}")
    return resolved


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: dict[str, int],
    epoch: int,
    arch: str,
    num_classes: int,
) -> None:
    """Save the model state and metadata to a checkpoint file."""

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
    logger.debug(f"Saved checkpoint to {path} at epoch {epoch}")


# ============================================================
# TRAINING / VALIDATION
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
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
) -> tuple[float, float]:
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
    unfreeze_from: Annotated[str | None, typer.Option(help="Layer name to unfreeze")] = None,
    device: str = "auto",
    amp: bool = True,
    seed: int = 42,
    out_dir: str = "outputs",
    run_name: str = "resnet_run",
    ckpt_name: str = "last.pt",
    save_best: bool = True,
    config_path: Annotated[str | None, typer.Option(help="Path to JSON config with training overrides")] = None,
) -> None:
    """
    Starts the training and validation process via the command line.
    """
    # Setup logging
    _setup_logging(run_name, out_dir)
    logger.info("=== Starting Training ===")
    logger.info(f"Run name: {run_name}, Output dir: {out_dir}")

    # Load optional JSON config overrides
    if config_path:
        logger.info(f"Loading config from {config_path}")
        with open(config_path) as f:
            cfg_json = json.load(f)
        processed_dir = cfg_json.get("processed_dir", processed_dir)
        batch_size = cfg_json.get("batch_size", batch_size)
        num_workers = cfg_json.get("num_workers", num_workers)
        arch = cfg_json.get("arch", arch)
        epochs = cfg_json.get("epochs", epochs)
        lr = cfg_json.get("lr", lr)
        weight_decay = cfg_json.get("weight_decay", weight_decay)
        pretrained = cfg_json.get("pretrained", pretrained)
        freeze_backbone = cfg_json.get("freeze_backbone", freeze_backbone)
        unfreeze_from = cfg_json.get("unfreeze_from", unfreeze_from)
        device = cfg_json.get("device", device)
        amp = cfg_json.get("amp", amp)
        seed = cfg_json.get("seed", seed)
        out_dir = cfg_json.get("out_dir", out_dir)
        run_name = cfg_json.get("run_name", run_name)
        ckpt_name = cfg_json.get("ckpt_name", ckpt_name)
        save_best = cfg_json.get("save_best", save_best)

    set_seed(seed)
    dev = resolve_device(device)
    logger.info(f"Using device: {dev}")

    # W&B init
    logger.debug("Initializing Weights & Biases")

    # W&B init
    run = wandb.init(
        project="group56-fish",  # change to your project name
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

    cfg = run.config
    lr = cfg.get("lr", lr)
    batch_size = cfg.get("batch_size", batch_size)
    weight_decay = cfg.get("weight_decay", weight_decay)

    freeze_backbone = cfg.get("freeze_backbone", freeze_backbone)
    unfreeze_from = cfg.get("unfreeze_from", unfreeze_from)

    if freeze_backbone:
        unfreeze_from = None

    # Data Initialization
    logger.info(f"Loading data from {processed_dir}")
    data_cfg = DataConfig(
        processed_dir=processed_dir,
        arch=arch,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader, val_loader, _, class_to_idx = make_dataloaders(data_cfg)
    num_classes = len(class_to_idx)
    logger.info(f"Loaded {num_classes} classes: {list(class_to_idx.keys())[:5]}...")

    # Model Initialization
    logger.info(f"Building {arch} model with {num_classes} classes")
    model = build_resnet(
        num_classes=num_classes,
        arch=arch,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        unfreeze_from=unfreeze_from,
    ).to(dev)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {trainable_params:,} trainable parameters")

    # Optimization Setup
    logger.debug(f"Optimizer setup: lr={lr}, weight_decay={weight_decay}")
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
    logger.info(f"Starting training for {epochs} epochs")
    # [Image of deep learning training process flowchart]
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

        epoch_msg = (
            f"Epoch {epoch:02d}/{epochs} | "
            f"tr_loss: {tr_loss:.4f} | tr_acc: {tr_acc:.4f} | "
            f"va_loss: {va_loss:.4f} | va_acc: {va_acc:.4f}"
        )
        logger.info(epoch_msg)

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
            logger.info(f"New best checkpoint: {best_ckpt} (acc={best_val_acc:.4f})")

        wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["best_epoch"] = epoch

    logger.info("Training process complete.")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Checkpoints saved to {out_path}")
    run.finish()


if __name__ == "__main__":
    typer.run(main)
