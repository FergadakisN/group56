import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tests import SRC_ROOT

sys.path.insert(0, str(SRC_ROOT))

from group_56.data import DataConfig, _extract_class_name_from_filename, make_dataloaders, split_dataset_by_class


def test_make_dataloaders_loads_tensors() -> None:
    """Ensure dataloaders return tensors and labels from the processed splits."""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        pytest.skip("data/processed not found; generate splits before running this test.")

    config = DataConfig(
        processed_dir=str(processed_dir),
        batch_size=2,
        num_workers=0,
        rebuild_processed=False,
        wipe_output_dir=False,
    )
    train_loader, val_loader, test_loader, class_to_idx = make_dataloaders(config)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert isinstance(class_to_idx, dict)
    assert len(class_to_idx) > 0

    def _assert_batch(loader: DataLoader, num_classes: int) -> None:
        images, labels = next(iter(loader))
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.ndim == 4
        assert images.shape[0] == labels.shape[0]
        assert labels.min().item() >= 0
        assert labels.max().item() < num_classes

    num_classes = len(class_to_idx)
    _assert_batch(train_loader, num_classes)
    _assert_batch(val_loader, num_classes)
    _assert_batch(test_loader, num_classes)


def test_label_mapping_matches_folder() -> None:
    """Ensure labels align with class folder names."""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        pytest.skip("data/processed not found; generate splits before running this test.")

    config = DataConfig(
        processed_dir=str(processed_dir),
        batch_size=2,
        num_workers=0,
        rebuild_processed=False,
        wipe_output_dir=False,
    )
    train_loader, val_loader, test_loader, class_to_idx = make_dataloaders(config)

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    def _assert_samples(loader: DataLoader) -> None:
        dataset = loader.dataset
        if not hasattr(dataset, "samples"):
            pytest.skip("Dataset does not expose samples for label verification.")
        samples = dataset.samples
        limit = min(20, len(samples))
        for image_path, label in samples[:limit]:
            class_name = Path(image_path).parent.name
            assert idx_to_class[label] == class_name

    _assert_samples(train_loader)
    _assert_samples(val_loader)
    _assert_samples(test_loader)


def test_no_data_leakage_between_splits() -> None:
    """Ensure no image appears in more than one split."""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        pytest.skip("data/processed not found; generate splits before running this test.")

    config = DataConfig(
        processed_dir=str(processed_dir),
        batch_size=2,
        num_workers=0,
        rebuild_processed=False,
        wipe_output_dir=False,
    )
    train_loader, val_loader, test_loader, _ = make_dataloaders(config)

    def _paths(loader: DataLoader) -> set[str]:
        dataset = loader.dataset
        if not hasattr(dataset, "samples"):
            pytest.skip("Dataset does not expose samples for leakage verification.")
        return {str(path) for path, _ in dataset.samples}

    train_paths = _paths(train_loader)
    val_paths = _paths(val_loader)
    test_paths = _paths(test_loader)

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_extract_class_name_from_filename() -> None:
    """Ensure class name parsing matches the filename format."""
    assert _extract_class_name_from_filename(Path("some_fish_12.png")) == "some_fish"
    assert _extract_class_name_from_filename(Path("singleclass.png")) == "singleclass"


def test_split_dataset_invalid_ratios(tmp_path: Path) -> None:
    """Ensure invalid split ratios raise an error."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        split_dataset_by_class(
            raw_dir=str(raw_dir),
            output_dir=str(tmp_path / "processed"),
            train_ratio=0.5,
            validation_ratio=0.3,
            test_ratio=0.3,
        )
