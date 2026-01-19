"""
Data preprocessing and loading module for the M7 Project.

This module provides utilities to split raw image datasets into training,
validation, and test sets, as well as custom PyTorch Dataset and DataLoader
implementations tailored for ResNet architectures.
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    low_count_threshold: int = 3,
    seed: int = 42,
    extensions: Iterable[str] = (".png", ".jpg", ".jpeg"),
    wipe_output_dir: bool = True,
) -> Dict[str, Dict[str, int]]:
    """
    Reads images from raw_dir and copies them into split folders by class.

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
        A dictionary mapping class names to counts in each split.

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

    image_paths = [
        p for p in raw_path.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {raw_path}")

    class_to_images: Dict[str, List[Path]] = defaultdict(list)
    for p in image_paths:
        class_to_images[_extract_class_name_from_filename(p)].append(p)

    split_counts: Dict[str, Dict[str, int]] = {}
    copy_tasks: List[Tuple[str, str, Path]] = []

    for class_name, images in class_to_images.items():
        images = sorted(images)

        # Ensure directory structure exists
        for split_name in ("train", "validation", "test"):
            (output_path / split_name / class_name).mkdir(parents=True, exist_ok=True)

        if len(images) <= low_count_threshold:
            split_map = {"train": images, "validation": [], "test": []}
        else:
            rng.shuffle(images)
            n_total = len(images)
            n_train = max(1, int(n_total * train_ratio))
            n_validation = max(1, int(n_total * validation_ratio))
            split_map = {
                "train": images[:n_train],
                "validation": images[n_train: n_train + n_validation],
                "test": images[n_train + n_validation:],
            }

        split_counts[class_name] = {k: len(v) for k, v in split_map.items()}
        copy_tasks.extend(
            (split_name, class_name, img_path)
            for split_name, split_images in split_map.items()
            for img_path in split_images
        )

    for split_name, class_name, img_path in copy_tasks:
        target_dir = output_path / split_name / class_name
        dest = target_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)

    return split_counts


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
        self.classes = [p.name for p in class_dirs]
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = sorted(
            [
                (img_path, self.class_to_idx[cls])
                for cls in self.classes
                for img_path in (split_dir / cls).rglob("*")
                if img_path.is_file() and img_path.suffix.lower() in exts
            ],
            key=lambda x: str(x[0]),
        )

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int] | Tuple[Any, int, str]:
        """Returns the (image, label) or (image, label, path) at the given index."""
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img, label, str(path)
        return img, label


def make_dataloaders(
    config: DataConfig = DataConfig(),
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Prepares DataLoaders for the training, validation, and test sets.

    Returns:
        A tuple of (train_loader, val_loader, test_loader, class_to_idx).
    """
    if config.rebuild_processed:
        split_dataset_by_class(
            raw_dir=config.raw_dir,
            output_dir=config.processed_dir,
            wipe_output_dir=config.wipe_output_dir,
        )

    transform = get_official_transform(config.arch)
    splits = ["train", "validation", "test"]
    datasets = {
        s: FolderSplitDataset(config.processed_dir, s, transform=transform)
        for s in splits
    }

    persistent = config.persistent_workers and config.num_workers > 0

    loaders = []
    for s in splits:
        loaders.append(
            DataLoader(
                datasets[s],
                batch_size=config.batch_size,
                shuffle=(s == "train"),
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=persistent,
            )
        )

    return (*loaders, datasets["train"].class_to_idx)


# ============================================================
# CLI ENTRYPOINT
# ============================================================

def build_splits_cli(
    raw_dir: str = "data/raw/cropped",
    output_dir: str = "data/processed",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    low_count_threshold: int = 3,
    seed: int = 42,
    wipe_output_dir: bool = True,
) -> None:
    """Command-line interface to trigger the dataset splitting process."""
    counts = split_dataset_by_class(
        raw_dir=raw_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        low_count_threshold=low_count_threshold,
        seed=seed,
        wipe_output_dir=wipe_output_dir,
    )

    for cls, splits in counts.items():
        typer.echo(f"{cls}: {splits}")


if __name__ == "__main__":
    typer.run(build_splits_cli)
