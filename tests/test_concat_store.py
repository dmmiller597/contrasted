"""Unit tests for AA∥3Di concat store construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from contrasted.concat import (
    ConcatStoreRequest,
    build_aa_3di_concat_store,
    concat_aa_di_rows,
    main,
    normalize_foldseek_3di_id,
)
from contrasted.data import EmbeddingStore


def _aa_store(
    tmp_path: Path,
    *,
    n: int = 4,
    dim: int = 8,
    with_labels: bool = True,
) -> Path:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(n, dim)).astype(np.float16)
    labels = np.arange(n, dtype=np.int64) % 2 if with_labels else None
    ids = [f"d{i}" for i in range(n)]
    store = EmbeddingStore(
        embeddings=embeddings,
        ids=ids,
        labels=labels,
        id_to_idx={d: i for i, d in enumerate(ids)},
        idx_to_label={0: "sf_a", 1: "sf_b"} if with_labels else None,
    )
    out = tmp_path / "aa_store"
    store.save(out, source="test-aa")
    return out


def test_concat_aa_di_rows_shape():
    aa = np.zeros((3, 4), dtype=np.float16)
    di = np.ones((3, 4), dtype=np.float16)
    out = concat_aa_di_rows(aa, di)
    assert out.shape == (3, 8)
    assert np.all(out[:, :4] == 0)
    assert np.all(out[:, 4:] == 1)


def test_build_aa_3di_concat_store_from_matrices(tmp_path):
    aa_dir = _aa_store(tmp_path, n=4, dim=8)
    di = np.arange(4 * 8, dtype=np.float16).reshape(4, 8)
    di_ids = [f"d{i}" for i in range(4)]
    out_dir = tmp_path / "concat"
    store = build_aa_3di_concat_store(
        ConcatStoreRequest(
            aa_store=aa_dir,
            output_dir=out_dir,
            domain_ids=di_ids,
            di_embeddings=di,
            di_ids=di_ids,
        )
    )
    assert store.embeddings.shape == (4, 16)
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["modality"] == "aa_3di_concat"
    assert meta["aa_dim"] == 8
    assert meta["di_dim"] == 8
    reloaded = EmbeddingStore.from_dir(out_dir, require_labels=True)
    assert reloaded.embeddings.shape == (4, 16)


def test_build_aa_3di_concat_store_without_labels(tmp_path):
    aa_dir = _aa_store(tmp_path, n=2, dim=4, with_labels=False)
    ids = ["d0", "d1"]
    out_dir = tmp_path / "concat"

    store = build_aa_3di_concat_store(
        ConcatStoreRequest(
            aa_store=aa_dir,
            output_dir=out_dir,
            domain_ids=ids,
            di_embeddings=np.ones((2, 4), dtype=np.float16),
            di_ids=ids,
        )
    )

    assert store.labels is None
    assert not (out_dir / "labels.npy").exists()
    assert EmbeddingStore.from_dir(out_dir).labels is None


@pytest.mark.parametrize(
    ("header", "known_id"),
    [
        ("1abcA00.pdb_A", "1abcA00"),
        (
            "AF-Q9Y6K1-F1-model_v4_TED03.pdb_A",
            "AF-Q9Y6K1-F1-model_v4_TED03",
        ),
    ],
)
def test_normalize_foldseek_3di_id_matches_known_id(header, known_id):
    assert normalize_foldseek_3di_id(header, {known_id}) == known_id


def test_normalize_foldseek_3di_id_preserves_afdb_ted_id_without_known_ids():
    header = "AF-Q9Y6K1-F1-model_v4_TED03.pdb_A"

    assert normalize_foldseek_3di_id(header) == "AF-Q9Y6K1-F1-model_v4_TED03"


def test_build_raises_when_no_requested_domains_can_be_kept(tmp_path, monkeypatch):
    aa_dir = _aa_store(tmp_path, n=2, dim=4)
    foldseek = tmp_path / "foldseek"
    foldseek.touch()
    monkeypatch.setattr("contrasted.concat.extract_3di_from_pdbs", lambda **_: {})

    with pytest.raises(RuntimeError, match="No requested domains"):
        build_aa_3di_concat_store(
            ConcatStoreRequest(
                aa_store=aa_dir,
                output_dir=tmp_path / "concat",
                domain_ids=["d0", "d1"],
                di_embeddings=np.empty((0, 4), dtype=np.float16),
                di_ids=[],
                pdb_dir=tmp_path,
                foldseek=foldseek,
            )
        )


def test_build_refuses_populated_dir(tmp_path):
    aa_dir = _aa_store(tmp_path, n=2, dim=4)
    di = np.zeros((2, 4), dtype=np.float16)
    ids = ["d0", "d1"]
    out_dir = tmp_path / "concat"
    build_aa_3di_concat_store(
        ConcatStoreRequest(
            aa_store=aa_dir,
            output_dir=out_dir,
            domain_ids=ids,
            di_embeddings=di,
            di_ids=ids,
        )
    )
    with pytest.raises(FileExistsError):
        build_aa_3di_concat_store(
            ConcatStoreRequest(
                aa_store=aa_dir,
                output_dir=out_dir,
                domain_ids=ids,
                di_embeddings=di,
                di_ids=ids,
            )
        )


def test_main_defaults_roster_to_aa_store_ids(tmp_path: Path) -> None:
    aa_dir = _aa_store(tmp_path, n=3, dim=4)
    ids = ["d0", "d1", "d2"]
    npy = tmp_path / "di.npy"
    np.save(npy, np.ones((3, 4), dtype=np.float16))
    id_file = tmp_path / "di_ids.txt"
    id_file.write_text("\n".join(ids) + "\n")
    out_dir = tmp_path / "concat"

    main(
        [
            "--aa-store",
            str(aa_dir),
            "--output-dir",
            str(out_dir),
            "--di-cache-npy",
            str(npy),
            "--di-cache-ids",
            str(id_file),
        ]
    )

    store = EmbeddingStore.from_dir(out_dir)
    assert store.ids == ids
    assert store.embeddings.shape == (3, 8)
