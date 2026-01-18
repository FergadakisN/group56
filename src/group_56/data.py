from __future__ import annotations
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import typer
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir: str | Path, extensions: Iterable[str] = (".png", ".jpg", ".jpeg")) -> None:
        self.data_dir = Path(data_dir)
        image_dir = self.data_dir / "cropped" if (self.data_dir / "cropped").exists() else self.data_dir
        self.image_paths = sorted(
            path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Path:
        return self.image_paths[index]


def _extract_class_name(image_path: Path) -> str:
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
) -> dict[str, dict[str, int]]:
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_path}")

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    class_to_images: dict[str, list[Path]] = defaultdict(list)
    for image_path in raw_path.rglob("*"):
        if image_path.is_file():
            class_name = _extract_class_name(image_path)
            class_to_images[class_name].append(image_path)

    split_counts: dict[str, dict[str, int]] = {}
    for class_name, image_paths in class_to_images.items():
        images = sorted(image_paths)
        if len(images) <= low_count_threshold:
            split_map = {"train": images, "validation": [], "test": []}
        else:
            rng.shuffle(images)
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_validation = int(n_total * validation_ratio)
            n_test = n_total - (n_train + n_validation)
            split_map = {
                "train": images[:n_train],
                "validation": images[n_train : n_train + n_validation],
                "test": images[n_train + n_validation : n_train + n_validation + n_test],
            }

        split_counts[class_name] = {}
        for split_name, split_images in split_map.items():
            target_dir = output_path / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                destination = target_dir / image_path.name
                if not destination.exists():
                    shutil.copy2(image_path, destination)
            split_counts[class_name][split_name] = len(split_images)

    return split_counts


def main(   
    raw_dir: str = "data/raw/cropped",
    output_dir: str = "data/processed",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    low_count_threshold: int = 3,
    seed: int = 42,) -> None:

    counts = split_dataset_by_class(
        raw_dir=raw_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        low_count_threshold=low_count_threshold,
        seed=seed,
    )

    for cls,splits in counts.items():
        typer.echo(f"{cls}: {splits}")

if __name__ == "__main__":
    typer.run(main)