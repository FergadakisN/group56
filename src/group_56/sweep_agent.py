"""
W&B Sweep Agent Runner.

This script initializes and runs a W&B hyperparameter sweep.
The default sweep configuration is configs/sweep.yaml.

Usage:
    python src/group_56/sweep_agent.py --sweep-id <SWEEP_ID> --count 20

Or:
    python src/group_56/sweep_agent.py --config configs/sweep.yaml
"""

import logging
import sys
from pathlib import Path

import typer
import wandb

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for sweep agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main(
    config_path: str = typer.Option(
        "configs/sweep.yaml",
        "--config",
        help="Path to sweep configuration YAML",
    ),
    count: int = typer.Option(
        10,
        "--count",
        help="Number of sweep runs to execute",
    ),
    sweep_id: str = typer.Option(
        None,
        "--sweep-id",
        help="Pre-existing sweep ID to join (if provided, config_path is ignored)",
    ),
) -> None:
    """
    Run a W&B hyperparameter sweep.

    Args:
        config_path: Path to sweep configuration YAML file.
        count: Number of runs to execute in this sweep.
        sweep_id: Optional existing sweep ID to join instead of creating a new one.
    """
    setup_logging()
    logger.info("=== Starting W&B Sweep Agent ===")

    if sweep_id:
        logger.info(f"Joining existing sweep: {sweep_id}")
    else:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        logger.info(f"Creating new sweep from config: {config_path}")
        sweep_id = wandb.sweep(str(config_file), project="group56-fish")
        logger.info(f"Created sweep with ID: {sweep_id}")

    logger.info(f"Running {count} sweep trials...")
    wandb.agent(sweep_id, count=count)
    logger.info("Sweep completed.")


if __name__ == "__main__":
    typer.run(main)
