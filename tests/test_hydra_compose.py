"""Compose every allowlisted Hydra entry config from the CLI config dir."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from configs import hydra_config_dir

ENTRY_CONFIGS = (
    "train",
    "embed",
    "make_db",
    "annotate",
    "make_db_s20",
    # Training recipes live in configs/train/. They use an absolute default
    # (- /train) so the group name does not shadow configs/train.yaml.
    "train/cath_s20_aa3di",
    "train/cath_s40_aa3di",
)

HOST_PATH_RE = re.compile(r"/(?:scratch\d+|SAN|Users)/")
_LAB_RECIPE = Path("configs/train/cath_ted_dualfilt_s40_aa3di.yaml")


@pytest.mark.parametrize("config_name", ENTRY_CONFIGS)
def test_entry_config_composes(config_name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=hydra_config_dir()):
        try:
            cfg = compose(config_name=config_name)
        except ConfigCompositionException as exc:
            pytest.fail(f"{config_name} failed to compose: {exc}")
    assert cfg is not None
    dumped = str(OmegaConf.to_container(cfg, throw_on_missing=False))
    assert HOST_PATH_RE.search(dumped) is None, dumped
    if config_name.startswith("train/"):
        assert "datamodule" in cfg, f"{config_name} did not flatten to the job root"


def test_lab_dualfilt_recipe_composes_via_cli() -> None:
    """Gitignored production YAML must still compose from a repo-root CLI."""
    if not _LAB_RECIPE.is_file():
        pytest.skip("lab dualfilt recipe is not on disk")
    exe = shutil.which("contrasted-train")
    if exe is None:
        pytest.skip("contrasted-train is not on PATH")
    result = subprocess.run(
        [exe, "--config-name=train/cath_ted_dualfilt_s40_aa3di", "--cfg", "job"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "embedding_dir" in result.stdout
    assert re.search(r"(?m)^datamodule:", result.stdout)


def test_lab_dualfilt_recipe_composes_from_a_subdirectory() -> None:
    """Analysis scripts often invoke the CLI from analysis/, not the repo root."""
    if not _LAB_RECIPE.is_file():
        pytest.skip("lab dualfilt recipe is not on disk")
    exe = shutil.which("contrasted-train")
    if exe is None:
        pytest.skip("contrasted-train is not on PATH")
    nested = Path("analysis")
    if not nested.is_dir():
        pytest.skip("analysis/ is not on disk")
    result = subprocess.run(
        [exe, "--config-name=train/cath_ted_dualfilt_s40_aa3di", "--cfg", "job"],
        check=False,
        capture_output=True,
        text=True,
        cwd=nested,
    )
    assert result.returncode == 0, result.stderr
    assert re.search(r"(?m)^datamodule:", result.stdout)
