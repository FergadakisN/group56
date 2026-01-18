from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
import typer

from project_name.data import DataConfig, make_dataloaders
from project_name.models import build_resnet


# ============================================================
# CONFIGS
# ============================================================

@dataclass(frozen=True)
class TrainConfig:
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
    import random, numpy as np, os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: dict,
    epoch: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
        },
        path,
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
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

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


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
    unfreeze_from: str = typer.Option(None),
    device: str = "auto",
    amp: bool = True,
    seed: int = 42,
    out_dir: str = "outputs",
    run_name: str = "resnet_run",
):
    set_seed(seed)
    dev = resolve_device(device)
    typer.echo(f"Using device: {dev}")

    # Data
    data_cfg = DataConfig(
        processed_dir=processed_dir,
        arch=arch,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader, _, _, class_to_idx = make_dataloaders(data_cfg)
    num_classes = len(class_to_idx)

    # Model
    model = build_resnet(
        num_classes=num_classes,
        arch=arch,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        unfreeze_from=unfreeze_from,
    ).to(dev)

    # Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and dev.type == "cuda"))

    # Output
    out_path = Path(out_dir) / run_name
    last_ckpt = out_path / "last.pt"

    # Training loop
    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=dev,
            scaler=scaler,
            use_amp=amp,
        )

        typer.echo(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss: {loss:.4f} | train acc: {acc:.4f}"
        )

        save_checkpoint(
            path=last_ckpt,
            model=model,
            class_to_idx=class_to_idx,
            epoch=epoch,
        )

    typer.echo(f"Training finished. Last checkpoint saved to {last_ckpt}")


if __name__ == "__main__":
    typer.run(main)
