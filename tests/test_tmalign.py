"""TM-align helpers: path resolution and retrieve-k then TM-best rerank."""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import pytest

from contrasted.tmalign import (
    pick_tm_best,
    rerank_by_tmalign,
    resolve_structure_path,
    run_tmalign,
)


def test_pick_tm_best_returns_highest_query_tm() -> None:
    assert pick_tm_best([]) is None
    hits = [("a", 0.40), ("b", 0.81), ("c", 0.55)]
    assert pick_tm_best(hits) == ("b", 0.81)


def test_pick_tm_best_keeps_first_on_tie() -> None:
    hits = [("first", 0.70), ("second", 0.70), ("third", 0.10)]
    assert pick_tm_best(hits) == ("first", 0.70)


def test_resolve_structure_path_bare_cath_filename(tmp_path: Path) -> None:
    bare = tmp_path / "1abcA00"
    bare.write_text("ATOM\n")
    assert resolve_structure_path("1abcA00", tmp_path) == bare


def test_resolve_structure_path_pdb_extension(tmp_path: Path) -> None:
    pdb = tmp_path / "1abcA00.pdb"
    pdb.write_text("ATOM\n")
    assert resolve_structure_path("1abcA00", tmp_path) == pdb


def test_resolve_structure_path_skips_directory_with_extension(tmp_path: Path) -> None:
    (tmp_path / "1abcA00.pdb").mkdir()
    cif = tmp_path / "1abcA00.cif"
    cif.write_text("ATOM\n")
    assert resolve_structure_path("1abcA00", tmp_path) == cif
    (tmp_path / "2xyzA00.pdb").mkdir()
    assert resolve_structure_path("2xyzA00", tmp_path) is None


def _write_fake_tmalign(tmp_path: Path) -> Path:
    """Stdout matches Zhang-Skolnick TMalign; TM-score is the target file body."""
    binary = tmp_path / "TMalign"
    binary.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import sys
            from pathlib import Path
            tm = Path(sys.argv[2]).read_text().strip()
            print("Aligned length= 10, RMSD= 1.00")
            print(f"TM-score= {tm} (normalized by length of Chain_1)")
            print("Length of Chain_1: 100 residues")
            """
        )
    )
    binary.chmod(0o755)
    return binary


def test_rerank_by_tmalign_assigns_tm_best(tmp_path: Path) -> None:
    queries = tmp_path / "q"
    targets = tmp_path / "t"
    queries.mkdir()
    targets.mkdir()
    (queries / "q1.pdb").write_text("QUERY\n")
    (targets / "near.pdb").write_text("0.42\n")
    (targets / "best.pdb").write_text("0.91\n")
    (targets / "mid.pdb").write_text("0.60\n")
    binary = _write_fake_tmalign(tmp_path)

    chosen = rerank_by_tmalign(
        "q1",
        ["near", "best", "mid"],
        queries,
        targets,
        binary=str(binary),
    )
    assert chosen == ("best", 0.91)


def test_run_tmalign_parses_fake_binary(tmp_path: Path) -> None:
    query = tmp_path / "q.pdb"
    target = tmp_path / "t.pdb"
    query.write_text("QUERY\n")
    target.write_text("0.55\n")
    result = run_tmalign(query, target, binary=str(_write_fake_tmalign(tmp_path)))
    assert result.tm_score == pytest.approx(0.55)
    assert result.rmsd == pytest.approx(1.0)
    assert result.aligned_length == 10
    assert result.query_length == 100


@pytest.mark.skipif(os.name == "nt", reason="chmod +x fake binary")
def test_rerank_skips_missing_target_structures(tmp_path: Path) -> None:
    queries = tmp_path / "q"
    targets = tmp_path / "t"
    queries.mkdir()
    targets.mkdir()
    (queries / "q1.pdb").write_text("QUERY\n")
    (targets / "only.pdb").write_text("0.33\n")
    chosen = rerank_by_tmalign(
        "q1",
        ["missing", "only"],
        queries,
        targets,
        binary=str(_write_fake_tmalign(tmp_path)),
    )
    assert chosen == ("only", 0.33)
