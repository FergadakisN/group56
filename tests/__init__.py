from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
DATA_PATH = PROJECT_ROOT / "data"
SRC_ROOT = PROJECT_ROOT / "src"

__all__ = ["DATA_PATH", "PROJECT_ROOT", "SRC_ROOT", "TEST_ROOT"]
