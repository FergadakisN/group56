"""
Data preprocessing and loading module for the M7 Project.

This module provides utilities to split raw image datasets into training,
validation, and test sets, as well as custom PyTorch Dataset and DataLoader
implementations tailored for ResNet architectures.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# ============================================================
# PART A) PREPROCESSING
# ============================================================


def _extract_class_name_from_filename(image_path: Path) -> str:
    """
    Extracts the class name from a filename (e.g., class_0123.jpg -> class).

    Args:
        image_path: The Path object of the image file.

    Returns:
        The extracted class name as a string.
    """
    stem = image_path.stem
    if "_" not in stem:
        return stem
    class_name, _ = stem.rsplit("_", 1)
    return class_name


def split_dataset_by_class(
    raw_dir: str = "data/raw/cropped",
    output_dir: str = "data/processed",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    low_count_threshold: int = 5,
    seed: int = 42,
    extensions: Iterable[str] = (".png", ".jpg", ".jpeg"),
    wipe_output_dir: bool = True,
) -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    """
    Split images per class into train/validation/test folders and emit metadata.

    Args:
        raw_dir: Source directory containing raw images.
        output_dir: Destination directory for processed splits.
        train_ratio: Proportion of images for the training set.
        validation_ratio: Proportion of images for the validation set.
        test_ratio: Proportion of images for the test set.
        low_count_threshold: Minimum images required per class to split.
        seed: Random seed for reproducibility.
        extensions: Allowed file extensions.
        wipe_output_dir: Whether to clear the output directory before starting.

    Returns:
        Tuple of (split_counts per class, split_assignment metadata DataFrame).

    Raises:
        ValueError: If split ratios do not sum to 1.0.
        FileNotFoundError: If the source directory is missing or empty.
    """
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_path}")

    output_path = Path(output_dir)
    if output_path.exists() and wipe_output_dir:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = {e.lower() for e in extensions}
    rng = random.Random(seed)

    image_paths = [p for p in raw_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {raw_path}")

    class_to_images: dict[str, list[Path]] = defaultdict(list)
    for p in image_paths:
        class_to_images[_extract_class_name_from_filename(p)].append(p)

    split_counts: dict[str, dict[str, int]] = {}
    copy_tasks: list[tuple[Path, str, str, Path, str]] = []
    split_records: list[dict] = []

    for class_name, images in class_to_images.items():
        images = sorted(images)

        n_images = len(images)

        if n_images <= low_count_threshold:
            # Rare class: all samples to train for better learning
            split_map = {"train": images, "validation": [], "test": []}
            reason = "rare_class"
        else:
            # Random split with ratios
            rng.shuffle(images)
            n_train = max(1, int(n_images * train_ratio))
            n_validation = max(1, int(n_images * validation_ratio))
            split_map = {
                "train": images[:n_train],
                "validation": images[n_train : n_train + n_validation],
                "test": images[n_train + n_validation :],
            }
            reason = "random_split"

        split_counts[class_name] = {k: len(v) for k, v in split_map.items()}

        copy_tasks.extend(
            (
                output_path / split_name / class_name / img_path.name,
                class_name,
                split_name,
                img_path,
                reason,
            )
            for split_name, split_images in split_map.items()
            for img_path in split_images
        )

    for dest, class_name, split_name, img_path, reason in copy_tasks:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest)
        split_records.append(
            {
                "image_id": img_path.stem,
                "species_id": class_name,
                "split": split_name,
                "reason": reason,
                "source_path": str(img_path),
                "dest_path": str(dest),
            }
        )

    split_df = pd.DataFrame(split_records)
    return split_counts, split_df


# ============================================================
# PART B) TRAINING INPUT
# ============================================================


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset splitting and dataloaders."""

    raw_dir: str = "data/raw/cropped"
    processed_dir: str = "data/processed"
    arch: str = "resnet18"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    rebuild_processed: bool = True
    wipe_output_dir: bool = True


def get_official_transform(arch: str) -> Any:
    """
    Retrieves the official torchvision transforms for pretrained ResNet weights.

    Args:
        arch: Model architecture name (e.g., 'resnet18').

    Returns:
        A torchvision transforms object.
    """
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
    elif arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    return weights.transforms()


class FolderSplitDataset(Dataset):
    """Custom Dataset to load images from a structured split directory."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        transform: Any = None,
        extensions: Iterable[str] = (".png", ".jpg", ".jpeg"),
        return_path: bool = False,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        """Initializes the dataset by scanning split folders."""
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.transform = transform
        self.return_path = return_path
        exts = {e.lower() for e in extensions}

        split_dir = self.processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

        if class_to_idx is None:
            self.classes = [p.name for p in class_dirs]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            # Reuse a shared mapping (e.g., from the train split) so label indices stay aligned.
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())

        samples: list[tuple[Path, int]] = []
        for cls_dir in class_dirs:
            cls_name = cls_dir.name
            if cls_name not in self.class_to_idx:
                # Skip classes not present in the shared mapping (e.g., rare classes kept only in train).
                continue
            label_idx = self.class_to_idx[cls_name]
            for img_path in cls_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in exts:
                    samples.append((img_path, label_idx))

        self.samples = sorted(samples, key=lambda x: str(x[0]))

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int] | tuple[Any, int, str]:
        """Returns the (image, label) or (image, label, path) at the given index."""
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img, label, str(path)
        return img, label


def make_dataloaders(
    config: DataConfig | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """
    Prepares DataLoaders for the training, validation, and test sets.

    Returns:
        A tuple of (train_loader, val_loader, test_loader, class_to_idx).
    """
    if config is None:
        config = DataConfig()

    if config.rebuild_processed:
        _counts, _ = split_dataset_by_class(
            raw_dir=config.raw_dir,
            output_dir=config.processed_dir,
            wipe_output_dir=config.wipe_output_dir,
        )

    transform = get_official_transform(config.arch)

    # Build train dataset first to establish a stable class_to_idx mapping.
    train_dataset = FolderSplitDataset(config.processed_dir, "train", transform=transform)
    shared_class_to_idx = train_dataset.class_to_idx

    datasets = {
        "train": train_dataset,
        "validation": FolderSplitDataset(
            config.processed_dir, "validation", transform=transform, class_to_idx=shared_class_to_idx
        ),
        "test": FolderSplitDataset(config.processed_dir, "test", transform=transform, class_to_idx=shared_class_to_idx),
    }

    persistent = config.persistent_workers and config.num_workers > 0

    loaders = []
    for split_name, ds in datasets.items():
        loaders.append(
            DataLoader(
                ds,
                batch_size=config.batch_size,
                shuffle=(split_name == "train"),
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=persistent,
            )
        )

    return (*loaders, shared_class_to_idx)


# ============================================================
# CLI ENTRYPOINT
# ============================================================


def build_splits_cli(
    raw_dir: str = "data/raw/cropped",
    output_dir: str = "data/processed",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    low_count_threshold: int = 5,
    seed: int = 42,
    wipe_output_dir: bool = True,
) -> None:
    """Command-line interface to trigger the dataset splitting process."""
    counts, split_df = split_dataset_by_class(
        raw_dir=raw_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        low_count_threshold=low_count_threshold,
        seed=seed,
        wipe_output_dir=wipe_output_dir,
    )

    split_csv_path = Path(output_dir) / "split_assignment.csv"
    split_df.to_csv(split_csv_path, index=False)
    typer.echo(f"Saved split assignments to {split_csv_path}")

    # Prepare summary statistics
    split_value_counts = split_df["split"].value_counts().to_dict()
    summary = {
        "total_records": len(split_df),
        "total_classes": len(counts),
        "split_totals": split_value_counts,
        "per_class_counts": counts,
        "split_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "low_count_threshold": low_count_threshold,
        "seed": seed,
    }

    # Save summary to JSON
    summary_json_path = Path(output_dir) / "split_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Saved split summary to {summary_json_path}")

    # Print summary to terminal
    typer.echo(f"\nTotal records tracked: {len(split_df)}")
    typer.echo(f"Total classes: {len(counts)}")
    typer.echo("\nSplit distribution:")
    for split_name, count in split_value_counts.items():
        typer.echo(f"  {split_name}: {count}")


if __name__ == "__main__":
    typer.run(build_splits_cli)
