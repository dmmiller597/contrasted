#!/usr/bin/env python3
"""Assert the built wheel and sdist carry exactly the public method surface.

uv build -o dist
python tests/assert_wheel_contents.py dist/*.whl dist/*.tar.gz
"""

from __future__ import annotations

import fnmatch
import re
import sys
import tarfile
import zipfile
from pathlib import Path

HOST_PATH_RE = re.compile(r"/(?:scratch\d+|SAN|Users)/")

WHEEL_BAN_GLOBS = (
    "contrasted/tm_graph.py",
    "contrasted/tm_graph/*",
)

CONFIG_ALLOW = {
    "configs/__init__.py",
    "configs/annotate.yaml",
    "configs/embed.yaml",
    "configs/make_db.yaml",
    "configs/train.yaml",
    "configs/train/__init__.py",
    "configs/train/cath_s20_aa3di.yaml",
    "configs/logger/csv.yaml",
    "configs/model/loss/ccl.yaml",
}

CONFIG_BAN_GLOBS = (
    "configs/*revision*",
    "configs/figure45_*",
    "configs/robin_*",
    "configs/novel_sf*",
    "configs/tm_graph*",
    "configs/*discovery*",
    "configs/*holdout*",
    "configs/*_split.yaml",
    "configs/train/*dualfilt*",
    "configs/train/cath_ted_s20_*",
    "configs/train/held_out_*",
    "configs/model/loss/discovery_*",
    "configs/model/loss/hierarchical_*",
    "configs/model/loss/parent_*",
    "configs/model/loss/ccl_full.yaml",
    "configs/model/loss/ccl_center_only.yaml",
    "configs/model/loss/center_contrastive.yaml",
)

WHEEL_REQUIRED = {
    "configs/__init__.py",
    "configs/train.yaml",
    "configs/annotate.yaml",
    "configs/make_db.yaml",
    "configs/embed.yaml",
    "configs/model/loss/ccl.yaml",
    "configs/train/__init__.py",
    "configs/train/cath_s20_aa3di.yaml",
    "contrasted/__init__.py",
    "contrasted/concat.py",
    "contrasted/data_home.py",
    "contrasted/registry.json",
}

SDIST_BAN_GLOBS = (
    "*/checkpoints/*",
    "*/analysis/*",
    "*/benchmarks/*",
    "*/scripts/*",
    "*/reproduce/*",
    "*/_archive/*",
    "*/_contrasted-manuscript/*",
    "*/.uv-cache/*",
    "*/.claude/*",
    "*/outputs/*",
)

SDIST_REQUIRED = {
    "configs/train.yaml",
    "configs/train/cath_s20_aa3di.yaml",
    "configs/annotate.yaml",
    "src/contrasted/concat.py",
    "src/contrasted/registry.json",
    "tests/assert_wheel_contents.py",
    "README.md",
    "LICENSE",
    "pyproject.toml",
}


def _names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    with tarfile.open(path) as tf:
        return tf.getnames()


def _read_members(path: Path, names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as zf:
            for name in names:
                out[name] = zf.read(name).decode("utf-8", "replace")
        return out
    with tarfile.open(path) as tf:
        for name in names:
            member = tf.extractfile(name)
            if member is not None:
                out[name] = member.read().decode("utf-8", "replace")
    return out


def _strip_sdist_prefix(names: list[str]) -> set[str]:
    """Drop the leading ``contrasted-<version>/`` component from sdist paths."""
    return {n.split("/", 1)[1] for n in names if "/" in n}


def _host_path_errors(path: Path, names: list[str], label: str) -> list[str]:
    yamls = [n for n in names if n.endswith(".yaml") and "__pycache__" not in n]
    errors = []
    for name, text in _read_members(path, yamls).items():
        if HOST_PATH_RE.search(text):
            errors.append(f"{label}: host absolute path in {name}")
    return errors


def check_wheel(wheel: Path) -> list[str]:
    errors: list[str] = []
    names = _names(wheel)
    label = wheel.name

    for pattern in WHEEL_BAN_GLOBS + CONFIG_BAN_GLOBS:
        hits = [n for n in names if fnmatch.fnmatch(n, pattern)]
        if hits:
            errors.append(f"{label}: banned member {pattern!r}: {hits[:5]}")

    configs = {
        n
        for n in names
        if n.startswith("configs/") and not n.endswith("/") and "__pycache__" not in n
    }
    if extra := sorted(configs - CONFIG_ALLOW):
        errors.append(f"{label}: configs outside the allowlist: {extra}")

    if not any(n.startswith("contrasted/") for n in names):
        errors.append(f"{label}: no contrasted/ package in the wheel")

    if missing := sorted(WHEEL_REQUIRED - set(names)):
        errors.append(f"{label}: missing required members: {missing}")

    errors += _host_path_errors(wheel, names, label)
    return errors


def check_sdist(sdist: Path) -> list[str]:
    errors: list[str] = []
    names = _names(sdist)
    label = sdist.name

    for pattern in SDIST_BAN_GLOBS:
        hits = [n for n in names if fnmatch.fnmatch(n, pattern)]
        if hits:
            errors.append(f"{label}: banned path {pattern!r}: {hits[:5]}")

    stripped = _strip_sdist_prefix(names)
    if missing := sorted(SDIST_REQUIRED - stripped):
        errors.append(f"{label}: missing required members: {missing}")

    errors += _host_path_errors(sdist, names, label)
    return errors


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    errors: list[str] = []
    for arg in argv:
        path = Path(arg)
        if not path.exists():
            errors.append(f"missing artifact: {path}")
        elif path.suffix == ".whl":
            errors += check_wheel(path)
        elif path.name.endswith(".tar.gz"):
            errors += check_sdist(path)
        else:
            errors.append(f"unrecognized artifact: {path}")

    for error in errors:
        print(error)
    if errors:
        print(f"FAILED: {len(errors)} problem(s)")
        return 1
    print(f"OK {' '.join(Path(a).name for a in argv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
