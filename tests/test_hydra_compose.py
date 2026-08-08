"""Compose every allowlisted Hydra entry config from pkg://configs."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from hydra.errors import ConfigCompositionException

ENTRY_CONFIGS = (
    "train",
    "embed",
    "make_db",
    "annotate",
    "make_db_cath_s100_s20",
    "cath_s20_aa3di_concat_ccl",
    "cath_s40_aa3di_concat_ccl",
)


@pytest.mark.parametrize("config_name", ENTRY_CONFIGS)
def test_entry_config_composes(config_name: str) -> None:
    with initialize_config_module(version_base=None, config_module="configs"):
        try:
            cfg = compose(config_name=config_name)
        except ConfigCompositionException as exc:
            pytest.fail(f"{config_name} failed to compose: {exc}")
    assert cfg is not None
