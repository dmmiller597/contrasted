"""Compose every allowlisted Hydra entry config from the CLI config dir."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from configs import CONFIG_DIR_ENV, hydra_config_dir

ENTRY_CONFIGS = (
    "train",
    "embed",
    "make_db",
    "annotate",
    # Training recipes live in configs/train/. They use an absolute default
    # (- /train) so the group name does not shadow configs/train.yaml.
    "train/cath_s20_aa3di",
)

HOST_PATH_RE = re.compile(r"/(?:scratch\d+|SAN|Users)/")


@pytest.fixture(autouse=True)
def _clear_config_dir_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)


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


def test_s20_recipe_writes_under_outputs_train() -> None:
    with initialize_config_dir(version_base=None, config_dir=hydra_config_dir()):
        cfg = compose(config_name="train/cath_s20_aa3di", return_hydra_config=True)
    run_dir = str(cfg.hydra.run.dir)
    assert run_dir.startswith("outputs/train/ccl_s20/seed_40/")
    assert cfg.hydra.job.chdir is False


def test_hydra_config_dir_ignores_ancestor_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packaged = hydra_config_dir()
    overlay = tmp_path / "pipeline" / "configs"
    overlay.mkdir(parents=True)
    (overlay / "train.yaml").write_text("input: fake\n")
    (overlay / "annotate.yaml").write_text("input: fake\n")
    work = tmp_path / "pipeline" / "work" / "ab" / "cdef"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    assert hydra_config_dir() == packaged


def test_hydra_config_dir_uses_contrasted_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "extra-configs"
    override.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(override))
    assert Path(hydra_config_dir()) == override.resolve()


def test_hydra_config_dir_rejects_missing_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "gone"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(missing))
    with pytest.raises(FileNotFoundError, match=CONFIG_DIR_ENV):
        hydra_config_dir()
