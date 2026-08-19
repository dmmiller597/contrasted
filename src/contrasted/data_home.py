"""Locate downloaded projection heads and indexes.

Weights are not in the wheel. The usual scientific-Python layout applies:
``get_data_home()`` is ``$CONTRASTED_DATA_DIR`` or ``~/.cache/contrasted``,
and a packaged registry names the expected files and hashes. ``locate()``
turns a Hydra filename into a path in that directory.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

DATA_HOME_ENV = "CONTRASTED_DATA_DIR"
_DEFAULT_DATA_HOME = Path.home() / ".cache" / "contrasted"


@lru_cache(maxsize=1)
def registry() -> dict[str, Any]:
    """Return the packaged file registry (filenames, hashes, cutoffs)."""
    text = (files("contrasted") / "registry.json").read_text(encoding="utf-8")
    return json.loads(text)


def get_data_home() -> Path:
    """Directory for downloaded heads and indexes.

    Uses ``$CONTRASTED_DATA_DIR`` when set, otherwise ``~/.cache/contrasted``.
    Creates the directory if it does not exist.
    """
    raw = os.environ.get(DATA_HOME_ENV, "").strip()
    home = Path(raw).expanduser() if raw else _DEFAULT_DATA_HOME
    home.mkdir(parents=True, exist_ok=True)
    return home


def locate(raw: str | Path, *, kind: str) -> Path:
    """Return an existing weights path, or raise with download instructions.

    An existing path wins. A bare filename then looks in ``get_data_home()``.
    A missing path that already names a directory is not rewritten to a
    same-basename file in the data home.
    """
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    if path.parent == Path("."):
        cached = get_data_home() / path.name
        if cached.exists():
            return cached.resolve()
    raise FileNotFoundError(_missing_message(path, kind=kind))


def _missing_message(path: Path, *, kind: str) -> str:
    home = get_data_home()
    tried = f"{path} and {home / path.name}"
    entry = registry().get("files", {}).get(path.name, {})
    lines = [
        f"{kind} not found. Looked in {tried}.",
        f"Put the file in ${DATA_HOME_ENV} (currently {home}).",
    ]
    url = registry().get("zenodo_url")
    if url:
        lines.append(f"Download the user bundle from {url}.")
    else:
        lines.append(
            "Download the user bundle from Zenodo (URL in the README once the "
            "DOI is minted) and place the files in that directory."
        )
    if entry:
        lines.append(f"Expected file: {path.name}")
        sha = entry.get("sha256")
        if sha:
            lines.append(f"sha256: {sha}")
    return " ".join(lines)
