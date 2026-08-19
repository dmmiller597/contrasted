"""Tests for contrasted.embed: batching, pooling correctness, store I/O."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from contrasted.data import (
    EmbeddingStore,
    read_fasta_sequences,
    resolve_store,
    validate_store_for_projection_head,
)
from contrasted.embed import (
    PROSTT5_DIM,
    EncodeConfig,
    ProstT5Encoder,
    _build_batches,
    _format_sequence,
)


def test_format_sequence_uppercases_and_masks_non_standard():
    # Lower-case (soft-masked) residues must be up-cased, and U/Z/O/B mapped
    # to X, so nothing tokenizes to <unk>.
    out = _format_sequence("mseUzObG")
    assert out == "<AA2fold> M S E X X X X G"


def test_format_3di_sequence_lowercases_and_uses_fold_prefix():
    out = _format_sequence("AcDeF", modality="3di")
    assert out == "<fold2AA> a c d e f"


def test_format_sequence_rejects_unknown_modality():
    with pytest.raises(ValueError, match="Unknown ProstT5 modality"):
        _format_sequence("ACDE", modality="dna")


def test_build_batches_flushes_on_max_batch():
    items = [(f"id{i}", "A" * 10) for i in range(5)]
    batches = _build_batches(items, max_residues=10_000, max_batch=2, max_seq_len=100)
    assert [len(b) for b in batches] == [2, 2, 1]


def test_build_batches_flushes_on_max_residues():
    items = [("id1", "A" * 50), ("id2", "A" * 60), ("id3", "A" * 30)]
    batches = _build_batches(items, max_residues=100, max_batch=99, max_seq_len=1000)
    # 50 fits; 50+60=110 >=100 flushes after id2; then id3 alone.
    assert [len(b) for b in batches] == [2, 1]


def test_build_batches_long_seq_isolated():
    items = [("short", "A" * 5), ("long", "A" * 2000)]
    batches = _build_batches(items, max_residues=10_000, max_batch=99, max_seq_len=1000)
    # long (>max_seq_len) flushes the batch it lands in.
    assert len(batches) == 1
    assert len(batches[0]) == 2


def test_read_fasta_sequences_joins_multiline_and_strips_gaps():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">cath|4_4_0|aaaA01/1-10\nMVL\nSPA-DK\n")
        f.write(">cath|4_4_0|bbbB01/1-5\nWGKV\n")
        path = Path(f.name)
    seqs = read_fasta_sequences(path)
    assert seqs == {"aaaA01": "MVLSPADK", "bbbB01": "WGKV"}


def test_read_fasta_sequences_drops_duplicates():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">cath|1|aaaA01/1-3\nMVL\n")
        f.write(">cath|1|aaaA01/1-3\nWGK\n")
        path = Path(f.name)
    seqs = read_fasta_sequences(path)
    assert seqs == {"aaaA01": "MVL"}


def test_read_fasta_sequences_rejects_header_only(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text(">cath|1|aaaA01/1-3\n")
    with pytest.raises(ValueError, match="no residues"):
        read_fasta_sequences(fasta)


def test_read_fasta_sequences_rejects_gap_only(tmp_path):
    fasta = tmp_path / "gaps.fasta"
    fasta.write_text(">cath|1|bbbB01/1-5\n----\n")
    with pytest.raises(ValueError, match="no residues"):
        read_fasta_sequences(fasta)


# ---------------------------------------------------------------------------
# Mocked encoder: assert pooling skips the <AA2fold> prefix token.
# ---------------------------------------------------------------------------


def _make_fake_hf_modules(dim: int = PROSTT5_DIM):
    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def half(self):
            return self

        def __call__(self, input_ids, attention_mask=None):
            bsz, seqlen = input_ids.shape
            pos = torch.arange(seqlen, dtype=torch.float32).view(1, seqlen, 1)
            hidden = pos.expand(bsz, seqlen, dim).clone()
            return MagicMock(last_hidden_state=hidden)

    class FakeEncoding:
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def to(self, _device):
            return self

    class FakeTokenizer:
        def batch_encode_plus(self, seqs, **_kwargs):
            token_lens = [1 + len(s.split()) - 1 for s in seqs]
            maxlen = max(token_lens)
            input_ids = torch.zeros(len(seqs), maxlen, dtype=torch.long)
            attn = torch.zeros_like(input_ids)
            for i, n in enumerate(token_lens):
                input_ids[i, :n] = 1
                attn[i, :n] = 1
            return FakeEncoding(input_ids, attn)

    return FakeModel, FakeTokenizer


@contextmanager
def _patched_hf_components(fake_model_cls, fake_tokenizer_cls):
    model_cls = MagicMock()
    tokenizer_cls = MagicMock()
    model_cls.from_pretrained.return_value = fake_model_cls()
    tokenizer_cls.from_pretrained.return_value = fake_tokenizer_cls()
    with patch("contrasted.embed._load_t5_components") as load_components:
        load_components.return_value = (model_cls, tokenizer_cls)
        yield model_cls, tokenizer_cls


def test_encoder_pools_from_residue_tokens_only():
    """The [1:s_len+1] slice must skip the prefix token."""
    FakeModel, FakeTokenizer = _make_fake_hf_modules(dim=4)
    with _patched_hf_components(FakeModel, FakeTokenizer):
        enc = ProstT5Encoder(EncodeConfig(device=torch.device("cpu"), dtype="float32"))
        out = enc.encode({"a": "ACG"})
    # Hidden states are position-indexed. For s_len=3, mean of hidden[1:4] = 2.0.
    # If the slice wrongly started at 0 (prefix included), it'd be 1.5.
    np.testing.assert_allclose(out["a"], np.full(4, 2.0, dtype=np.float32))


def test_encoder_local_files_only_passed_to_hf_loaders():
    FakeModel, FakeTokenizer = _make_fake_hf_modules(dim=4)
    with _patched_hf_components(FakeModel, FakeTokenizer) as (
        model_cls,
        tokenizer_cls,
    ):
        enc = ProstT5Encoder(
            EncodeConfig(
                device=torch.device("cpu"),
                dtype="float32",
                local_files_only=True,
            )
        )
        enc.encode({"a": "ACG"})

    model_cls.from_pretrained.assert_called_once_with(
        "Rostlab/ProstT5", local_files_only=True
    )
    tokenizer_cls.from_pretrained.assert_called_once_with(
        "Rostlab/ProstT5", do_lower_case=False, local_files_only=True
    )


# ---------------------------------------------------------------------------
# EmbeddingStore.save() round-trip and resolve_store dispatcher.
# ---------------------------------------------------------------------------


def test_embedding_store_save_roundtrip(tmp_path):
    embeddings = np.arange(24, dtype=np.float32).reshape(4, 6)
    ids = [f"d{i}" for i in range(4)]
    labels = np.arange(4, dtype=np.int64)
    store = EmbeddingStore(
        embeddings=embeddings,
        ids=ids,
        labels=labels,
        id_to_idx={d: i for i, d in enumerate(ids)},
        idx_to_label={i: f"sf_{i}" for i in range(4)},
    )

    out_dir = tmp_path / "store"
    store.save(out_dir, source="unit-test")

    loaded = EmbeddingStore.from_dir(out_dir)
    np.testing.assert_array_equal(np.asarray(loaded.embeddings), embeddings)
    assert loaded.ids == ids
    np.testing.assert_array_equal(loaded.labels, labels)
    assert loaded.idx_to_label == {i: f"sf_{i}" for i in range(4)}


def test_embedding_store_save_refuses_overwrite(tmp_path):
    store = EmbeddingStore(
        embeddings=np.zeros((2, 3), dtype=np.float32),
        ids=["a", "b"],
        labels=None,
        id_to_idx={"a": 0, "b": 1},
    )
    out = tmp_path / "store"
    store.save(out)
    with pytest.raises(FileExistsError):
        store.save(out)


def test_resolve_store_loads_existing_dir(tmp_path):
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    store = EmbeddingStore(
        embeddings=embeddings,
        ids=["a", "b", "c"],
        labels=None,
        id_to_idx={"a": 0, "b": 1, "c": 2},
    )
    out = tmp_path / "cache"
    store.save(out)

    resolved = resolve_store(embedding_dir=out)
    np.testing.assert_array_equal(np.asarray(resolved.embeddings), embeddings)


def test_resolve_store_errors_on_partial_dir(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "ids.txt").write_text("a\n")
    with pytest.raises(FileExistsError):
        resolve_store(embedding_dir=partial)


def test_resolve_store_errors_when_dir_missing(tmp_path):
    cache = tmp_path / "new_cache"
    with pytest.raises(FileNotFoundError, match="Embedding store not found"):
        resolve_store(embedding_dir=cache)


@pytest.mark.parametrize("embedding_dir", [None, ""])
def test_resolve_store_requires_embedding_dir(embedding_dir):
    with pytest.raises(
        ValueError,
        match=r"embedding_dir is required.*contrasted-build-concat-store",
    ):
        resolve_store(embedding_dir=embedding_dir)


def test_validate_store_for_projection_head_rejects_explicit_aa_modality(tmp_path):
    store = EmbeddingStore(
        embeddings=np.zeros((1, 2048), dtype=np.float32),
        ids=["a"],
        labels=None,
        id_to_idx={"a": 0},
    )
    store.save(tmp_path / "aa-store", extra_metadata={"modality": "aa"})
    loaded = resolve_store(embedding_dir=tmp_path / "aa-store")

    with pytest.raises(ValueError, match="requires an aa_3di_concat store"):
        validate_store_for_projection_head(loaded, input_dim=2048)


def test_validate_store_for_projection_head_accepts_aa_3di_concat_modality():
    store = EmbeddingStore(
        embeddings=np.zeros((1, 2048), dtype=np.float32),
        ids=["a"],
        labels=None,
        id_to_idx={"a": 0},
        metadata={"modality": "aa_3di_concat"},
    )

    validate_store_for_projection_head(store, input_dim=2048)


def test_validate_store_for_projection_head_rejects_missing_modality():
    store = EmbeddingStore(
        embeddings=np.zeros((1, 2048), dtype=np.float32),
        ids=["a"],
        labels=None,
        id_to_idx={"a": 0},
        metadata={},
    )

    with pytest.raises(ValueError, match="requires an aa_3di_concat store"):
        validate_store_for_projection_head(store, input_dim=2048)
