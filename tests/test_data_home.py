"""Locate projection heads and indexes under $CONTRASTED_DATA_DIR."""

from __future__ import annotations

from pathlib import Path

import pytest

from contrasted.data_home import DATA_HOME_ENV, get_data_home, locate, registry


def test_registry_names_the_default_s20_head() -> None:
    reg = registry()
    assert reg["default_head"] == "aa3di_s20_seed40_head.pt"
    assert reg["default_index"] == "cath_s20_centroids.pt"
    head = reg["files"][reg["default_head"]]
    assert head["sha256"]
    assert head["distance_cutoff"] == 0.24048042
    assert head.get("canonical", True) is not False


def test_get_data_home_reads_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "cache"
    monkeypatch.setenv(DATA_HOME_ENV, str(home))
    assert get_data_home() == home
    assert home.is_dir()


def test_locate_prefers_an_existing_path(tmp_path: Path) -> None:
    f = tmp_path / "head.pt"
    f.write_bytes(b"ok")
    assert locate(f, kind="Projection head") == f.resolve()


def test_locate_falls_back_to_data_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "cache"
    home.mkdir()
    target = home / "custom_head.pt"
    target.write_bytes(b"ok")
    monkeypatch.setenv(DATA_HOME_ENV, str(home))
    found = locate("custom_head.pt", kind="Projection head")
    assert found == target.resolve()


def test_locate_rejects_registry_sha256_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "cache"
    home.mkdir()
    target = home / "aa3di_s20_seed40_head.pt"
    target.write_bytes(b"wrong-hash")
    monkeypatch.setenv(DATA_HOME_ENV, str(home))
    with pytest.raises(ValueError, match="does not match registry"):
        locate("aa3di_s20_seed40_head.pt", kind="Projection head")


def test_locate_does_not_reuse_cache_for_a_missing_directory_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "cache"
    home.mkdir()
    (home / "head.pt").write_bytes(b"cached")
    monkeypatch.setenv(DATA_HOME_ENV, str(home))
    missing = tmp_path / "gone" / "head.pt"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        locate(missing, kind="Projection head")


def test_locate_missing_known_file_names_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="sha256: d3018916") as exc:
        locate("aa3di_s20_seed40_head.pt", kind="Projection head")
    msg = str(exc.value)
    assert DATA_HOME_ENV in msg
    assert "aa3di_s20_seed40_head.pt" in msg
