"""Hydra configuration package for contrasted."""

from pathlib import Path


def hydra_config_dir() -> str:
    """Directory Hydra should load recipes from.

    Console scripts import the installed ``configs`` copy, which is only the
    wheel allowlist. An editable checkout keeps extra lab recipes (dualfilt,
    held-out) under ``configs/``. Prefer that overlay when the process is
    running from the checkout or any subdirectory of it.
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        overlay = candidate / "configs"
        if (overlay / "train.yaml").is_file() and (overlay / "annotate.yaml").is_file():
            return str(overlay)
    return str(Path(__file__).resolve().parent)
