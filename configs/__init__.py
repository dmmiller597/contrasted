"""Hydra configuration package for contrasted."""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR_ENV = "CONTRASTED_CONFIG_DIR"


def hydra_config_dir() -> str:
    """Directory Hydra should load recipes from.

    Uses ``$CONTRASTED_CONFIG_DIR`` when set, otherwise the ``configs``
    package next to this file (editable checkout or the wheel copy).
    Set the env var before launching the process; Hydra reads this path
    when the console script is imported.
    """
    raw = os.environ.get(CONFIG_DIR_ENV, "").strip()
    if raw:
        override = Path(raw).expanduser().resolve()
        if not override.is_dir():
            raise FileNotFoundError(
                f"${CONFIG_DIR_ENV} is {override}, which is not a directory"
            )
        return str(override)
    return str(Path(__file__).resolve().parent)
