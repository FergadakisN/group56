# group_56 Documentation

## Overview

This project delivers an image-classification pipeline for fish species with reproducible data handling, training, and
experiment tracking. Tooling includes DVC for data, Typer CLIs, Invoke tasks, W&B logging, and MkDocs for docs.

## Workflow snapshot

1. Install: `pip install -r requirements.txt && pip install -e .` (or `pip install -e .["dev"]` for docs/tests).
2. Pull data: `dvc pull` (prompts Google auth on first run).
3. Build splits: `data-split --raw-dir data/raw/cropped --output-dir data/processed --low-count-threshold 5`.
4. Train: `python -m group_56.train --config configs/train_example.json` (or `train-model ...`).

## Data splitting

- Classes are derived from filename prefixes before the final underscore.
- Classes with ≤5 images are kept entirely in the training split; others use a 70/15/15 train/val/test split.
- Outputs: `data/processed/` with per-split class folders, `split_assignment.csv`, and `split_summary.json` detailing
  counts, ratios, threshold, and seed.

## Training

- ResNet 18/34/50 backbones with optional backbone freezing and AMP.
- W&B project: `group56-fish`; set `WANDB_DISABLED=true` or `WANDB_MODE=offline` to avoid network logging.
- Checkpoints land in `outputs/<run_name>/` (`last.pt`, `best.pt`).
- Override hyperparameters via JSON config (see `configs/train_example.json`) or CLI flags.

## Useful commands

- Data split: `data-split ...` or `invoke preprocess_data`.
- Train: `python -m group_56.train ...` or `train-model ...`.
- Evaluate: `evaluate-model --split validation` (default split is test).
- Tests: `pytest tests/` or `invoke test` for coverage.
- Docs: `mkdocs serve --config-file docs/mkdocs.yaml` or `invoke serve_docs`.
