"""CATH S20 split invariants.

CI does not ship ``data/cath_s20_split/``. Count constants are always checked.
FASTA-level checks run when the lock files are on disk.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from contrasted.datamodule import S20_SPLIT_EXPECTATIONS, SPLIT_EXPECTATIONS

LOCK_PATH = Path("data/cath_s20_split/LOCK.json")
SPLIT_DIR = Path("data/cath_s20_split")
SF_MAP_PATH = Path("data/cath-domain-sf-list-c123.txt")


def _load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _extract_domain_id(header: str) -> str:
    token = header.lstrip(">").split()[0]
    if "|" in token:
        token = token.split("|")[-1]
    if "/" in token:
        token = token.split("/")[0]
    return token


def _fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(_extract_domain_id(line))
    return ids


def test_s20_expectations_are_registered() -> None:
    assert SPLIT_EXPECTATIONS["s20"] is S20_SPLIT_EXPECTATIONS
    assert S20_SPLIT_EXPECTATIONS["train_h"] == 5_659
    assert S20_SPLIT_EXPECTATIONS["val_queries"] == 514
    assert S20_SPLIT_EXPECTATIONS["test_queries"] == 1_028
    assert S20_SPLIT_EXPECTATIONS["train_domains"] == 116_301
    assert S20_SPLIT_EXPECTATIONS["num_c"] == 3
    assert S20_SPLIT_EXPECTATIONS["num_a"] == 40
    assert S20_SPLIT_EXPECTATIONS["num_t"] == 1_282
    assert S20_SPLIT_EXPECTATIONS["num_h"] == 5_659


@pytest.mark.skipif(not LOCK_PATH.is_file(), reason="S20 split is not on disk")
def test_lock_json_matches_datamodule_expectations() -> None:
    lock = json.loads(LOCK_PATH.read_text())
    assert lock["protocol"] == "cath_s20"
    counts = lock["counts"]
    assert counts["train_domains"] == S20_SPLIT_EXPECTATIONS["train_domains"]
    assert counts["train_h"] == S20_SPLIT_EXPECTATIONS["train_h"]
    assert counts["val_queries"] == S20_SPLIT_EXPECTATIONS["val_queries"]
    assert counts["test_queries"] == S20_SPLIT_EXPECTATIONS["test_queries"]
    assert counts["num_c"] == S20_SPLIT_EXPECTATIONS["num_c"]
    assert counts["num_a"] == S20_SPLIT_EXPECTATIONS["num_a"]
    assert counts["num_t"] == S20_SPLIT_EXPECTATIONS["num_t"]
    assert counts["num_h"] == S20_SPLIT_EXPECTATIONS["num_h"]
    assert lock["mmseqs"]["min_seq_id"] == 0.2
    assert lock["mmseqs"]["cov_mode"] == "0"
    assert set(lock["failed_remote_superfamilies"]) == {
        "1.20.1050.20",
        "2.130.10.130",
        "3.30.20.10",
        "3.40.190.120",
        "3.40.50.10310",
    }


@pytest.mark.skipif(not LOCK_PATH.is_file(), reason="S20 split is not on disk")
def test_lock_fasta_hashes_and_disjoint_ids() -> None:
    lock = json.loads(LOCK_PATH.read_text())
    for name, expected in lock["sha256"].items():
        digest = hashlib.sha256((SPLIT_DIR / name).read_bytes()).hexdigest()
        assert digest == expected, name

    train = set(_load_ids(SPLIT_DIR / "train_ids.txt"))
    val = set(_load_ids(SPLIT_DIR / "val_ids.txt"))
    test = set(_load_ids(SPLIT_DIR / "test_ids.txt"))
    assert len(train) == S20_SPLIT_EXPECTATIONS["train_domains"]
    assert len(val) == S20_SPLIT_EXPECTATIONS["val_queries"]
    assert len(test) == S20_SPLIT_EXPECTATIONS["test_queries"]
    assert not train & val
    assert not train & test
    assert not val & test
    assert _fasta_ids(SPLIT_DIR / "train.fasta") == _load_ids(
        SPLIT_DIR / "train_ids.txt"
    )
    assert _fasta_ids(SPLIT_DIR / "val.fasta") == _load_ids(SPLIT_DIR / "val_ids.txt")
    assert _fasta_ids(SPLIT_DIR / "test.fasta") == _load_ids(SPLIT_DIR / "test_ids.txt")


@pytest.mark.skipif(
    not (LOCK_PATH.is_file() and SF_MAP_PATH.is_file()),
    reason="S20 split or SF map is not on disk",
)
def test_every_class_1_3_superfamily_is_in_train() -> None:
    sf_map = {}
    with SF_MAP_PATH.open() as handle:
        for line in handle:
            domain_id, superfamily = line.rstrip().split("\t")
            sf_map[domain_id] = superfamily
    universe = set(sf_map.values())
    train_sfs = {sf_map[i] for i in _load_ids(SPLIT_DIR / "train_ids.txt")}
    val_sfs = {sf_map[i] for i in _load_ids(SPLIT_DIR / "val_ids.txt")}
    test_sfs = {sf_map[i] for i in _load_ids(SPLIT_DIR / "test_ids.txt")}
    assert train_sfs == universe
    assert len(universe) == 5_659
    assert not val_sfs & test_sfs
    assert val_sfs <= train_sfs
    assert test_sfs <= train_sfs
    assert len(val_sfs) == len(_load_ids(SPLIT_DIR / "val_ids.txt"))
    assert len(test_sfs) == len(_load_ids(SPLIT_DIR / "test_ids.txt"))

    size = Counter(sf_map.values())
    giants = {sf for sf, n in size.items() if n >= 100}
    eval_sfs = val_sfs | test_sfs
    probes = defaultdict(list)
    with (SPLIT_DIR / "chosen_probes.tsv").open() as handle:
        next(handle)
        for line in handle:
            superfamily, _split, _query, n_s100 = line.rstrip().split("\t")
            probes[superfamily].append(int(n_s100))
    cheap_giants = {sf for sf, ns in probes.items() if ns[0] >= 100}
    assert cheap_giants <= eval_sfs
    assert len(cheap_giants) == 204
    assert cheap_giants <= giants


HOLDOUT_PATH = SPLIT_DIR / "holdout_summary.json"


@pytest.mark.skipif(not HOLDOUT_PATH.is_file(), reason="H50 holdout is not on disk")
def test_h50_holdout_is_train_only_and_hashed() -> None:
    holdout = json.loads(HOLDOUT_PATH.read_text())
    assert holdout["protocol"] == "train_only_ge11_uniform"
    assert holdout["seed"] == 42
    assert holdout["n_holdout"] == 50
    assert holdout["min_members"] == 11
    sfs = _load_ids(SPLIT_DIR / "holdout_superfamilies.txt")
    assert len(sfs) == 50
    assert len(set(sfs)) == 50
    assert holdout["holdout_superfamilies"] == sfs
    for name, expected in holdout["sha256"].items():
        digest = hashlib.sha256((SPLIT_DIR / name).read_bytes()).hexdigest()
        assert digest == expected, name

    train = set(_load_ids(SPLIT_DIR / "train_ids.txt"))
    val = set(_load_ids(SPLIT_DIR / "val_ids.txt"))
    test = set(_load_ids(SPLIT_DIR / "test_ids.txt"))
    hold_ids = set(_load_ids(SPLIT_DIR / "holdout_ids.txt"))
    keep_ids = set(_load_ids(SPLIT_DIR / "train_excl_holdouts_ids.txt"))
    assert hold_ids.isdisjoint(val)
    assert hold_ids.isdisjoint(test)
    assert hold_ids.isdisjoint(keep_ids)
    assert hold_ids | keep_ids == train
    assert len(hold_ids) + len(keep_ids) == len(train)


@pytest.mark.skipif(
    not (HOLDOUT_PATH.is_file() and SF_MAP_PATH.is_file()),
    reason="H50 holdout or SF map is not on disk",
)
def test_h50_holdout_sizes_and_no_eval_superfamilies() -> None:
    sf_map = {}
    with SF_MAP_PATH.open() as handle:
        for line in handle:
            domain_id, superfamily = line.split()
            sf_map[domain_id] = superfamily
    val = _load_ids(SPLIT_DIR / "val_ids.txt")
    test = _load_ids(SPLIT_DIR / "test_ids.txt")
    train = _load_ids(SPLIT_DIR / "train_ids.txt")
    eval_sfs = {sf_map[i] for i in val} | {sf_map[i] for i in test}
    train_n = Counter(sf_map[i] for i in train)
    sfs = _load_ids(SPLIT_DIR / "holdout_superfamilies.txt")
    assert not set(sfs) & eval_sfs
    assert all(train_n[sf] >= 11 for sf in sfs)
    hold_ids = _load_ids(SPLIT_DIR / "holdout_ids.txt")
    assert {sf_map[i] for i in hold_ids} == set(sfs)
    assert _fasta_ids(SPLIT_DIR / "holdouts.fasta") == hold_ids


_RETIRED_S20_ALIASES = (
    "remote-v2",
    "remote_v2",
    "lockedA",
    "locked-A",
    "cath_s20_remote",
    "S20 S20",
)
_HOST_MARKERS = ("/SAN/", "/scratch")
_TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".toml",
    ".txt",
    ".tex",
    ".sh",
}


def test_live_tree_has_no_retired_s20_aliases() -> None:
    roots = (
        Path("src"),
        Path("tests"),
        Path("configs"),
        Path("README.md"),
        Path("analysis"),
        Path("data/cath_s20_split"),
        Path("_contrasted-manuscript/contrasted-article.tex"),
        Path("_contrasted-manuscript/contrasted-supplementary.tex"),
        Path("_contrasted-manuscript/resources/generated"),
    )
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
                continue
            if path.name == Path(__file__).name:
                continue
            if any(part in {"runs", "_archive"} for part in path.parts):
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if any(marker in line for marker in _HOST_MARKERS):
                    continue
                for token in _RETIRED_S20_ALIASES:
                    if token in line:
                        hits.append(f"{path}:{lineno}: {token}")
    assert not hits, "retired S20 aliases remain:\n" + "\n".join(hits)
