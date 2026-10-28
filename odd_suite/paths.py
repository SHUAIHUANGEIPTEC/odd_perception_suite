"""Repository-rooted path resolution (overridable via environment)."""

from __future__ import annotations

import os
from pathlib import Path

# odd_suite/paths.py -> odd_suite/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def weights_dir() -> Path:
    return _env_path("ODD_WEIGHTS_ROOT", REPO_ROOT / "weights")


def configs_dir() -> Path:
    return REPO_ROOT / "configs"


def profiles_dir() -> Path:
    return REPO_ROOT / "profiles"


def switches_dir() -> Path:
    return REPO_ROOT / "switches"


def mapping_tables_dir() -> Path:
    return Path(__file__).resolve().parent / "mapping" / "tables"


def vendor_sparse4d_dir() -> Path:
    return REPO_ROOT / "vendor" / "sparse4d"


def examples_dir() -> Path:
    return REPO_ROOT / "examples"
