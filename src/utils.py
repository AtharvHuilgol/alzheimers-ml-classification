"""Shared utilities for reproducibility and path management."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set global random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path | str) -> Path:
    """Create directory if it does not exist and return its path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_path(*parts: str) -> Path:
    """Resolve a path under the data/ directory."""
    return get_project_root() / "data" / Path(*parts)


def results_path(*parts: str) -> Path:
    """Resolve a path under the results/ directory."""
    return get_project_root() / "results" / Path(*parts)
