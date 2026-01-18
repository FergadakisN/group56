from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import typer
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# ============================================================
# PART A) PREPROCESSING: build data/processed/train|validation|test/<class>/
# ============================================================

def _extract_class_name_from_filename(image_path: Path) -> str:
    """
    Extract class name from filename like: class_0123.jpg -> class
    If no underscore is present: class.jpg -> class
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
    Reads images from raw_dir and copies them into:

      output_dir/train/<class>/
      output_dir/validation/<class>/
      output_dir/test/<class>/

    Classes are inferred from the filename.
    Returns per-class counts in each split.
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

    # Collect images by class (flattened filtering)
    image_paths = [
        p for p in raw_path.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {raw_path} with extensions {sorted(exts)}")

    class_to_images: Dict[str, List[Path]] = defaultdict(list)
    for p in image_paths:
        class_to_images[_extract_class_name_from_filename(p)].append(p)

    split_counts: Dict[str, Dict[str, int]] = {}
    copy_tasks: List[Tuple[str, str, Path]] = []  # (split_name, class_name, img_path)

    for class_name, images in class_to_images.items():
        images = sorted(images)

        # If too few samples, keep everything in train
        if len(images) <= low_count_threshold:
            split_map = {"train": images, "validation": [], "test": []}
        else:
            rng.shuffle(images)
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_validation = int(n_total * validation_ratio)
            split_map = {
                "train": images[:n_train],
                "validation": images[n_train : n_train + n_validation],
                "test": images[n_train + n_validation :],
            }

        # Save counts
        split_counts[class_name] = {k: len(v) for k, v in split_map.items()}

        # Flatten all file copy operations into one list
        copy_tasks.extend(
            (split_name, class_name, img_path)
            for split_name, split_images in split_map.items()
            for img_path in split_images
        )

    # Execute copy tasks with a single loop
    for split_name, class_name, img_path in copy_tasks:
        target_dir = output_path / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        dest = target_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)

    return split_counts


# ============================================================
# PART B) TRAINING INPUT: Custom Dataset + official transforms
# ============================================================

@dataclass(frozen=True)
class DataConfig:
    processed_dir: str = "data/processed"
    arch: str = "resnet18"  # resnet18/resnet34/resnet50
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


def get_official_transform(arch: str):
    """
    Official torchvision preprocessing for pretrained weights.
    Deterministic and correct (val/test style).
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
    """
    Reads images from:
      processed_dir/<split>/<class_name>/*.{jpg,png,...}

    Returns:
      image_tensor, label_int   (optionally also path)
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        transform=None,
        extensions: Iterable[str] = (".png", ".jpg", ".jpeg"),
        return_path: bool = False,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.transform = transform
        self.return_path = return_path
        exts = {e.lower() for e in extensions}

        split_dir = self.processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        if not class_dirs:
            raise FileNotFoundError(
                f"No class subfolders found in {split_dir}. Expected {split_dir}/<class_name>/image.jpg"
            )

        self.classes = [p.name for p in class_dirs]
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        # Flattened sample collection (no nested for blocks)
        self.samples: List[Tuple[Path, int]] = sorted(
            [
                (img_path, self.class_to_idx[cls])
                for cls in self.classes
                for img_path in (split_dir / cls).rglob("*")
                if img_path.is_file() and img_path.suffix.lower() in exts
            ],
            key=lambda x: str(x[0]),
        )

        if not self.samples:
            raise FileNotFoundError(f"No images found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img, label, str(path)
        return img, label


def make_dataloaders(
    config: DataConfig = DataConfig(),
):
    """
    Returns train/val/test DataLoaders + class_to_idx mapping.
    Uses official torchvision transforms for the chosen ResNet weights.
    """
    transform = get_official_transform(config.arch)

    train_ds = FolderSplitDataset(config.processed_dir, "train", transform=transform)
    val_ds = FolderSplitDataset(config.processed_dir, "validation", transform=transform)
    test_ds = FolderSplitDataset(config.processed_dir, "test", transform=transform)

    # Ensure consistent label mapping across splits
    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError(
            "Class folders differ across splits. Ensure train/validation/test contain the same class subfolders.\n"
            f"train: {train_ds.class_to_idx}\n"
            f"validation: {val_ds.class_to_idx}\n"
            f"test: {test_ds.class_to_idx}"
        )

    persistent = config.persistent_workers and config.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


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
):
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
