"""Tests for contrasted.embed: batching, pooling correctness, store I/O."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from contrasted.data import EmbeddingStore, read_fasta_sequences, resolve_store
from contrasted.embed import (
    PROSTT5_DIM,
    EncodeConfig,
    ProstT5Encoder,
    _build_batches,
)


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
        yield


def test_encoder_pools_from_residue_tokens_only():
    """The [1:s_len+1] slice must skip the prefix token."""
    FakeModel, FakeTokenizer = _make_fake_hf_modules(dim=4)
    with _patched_hf_components(FakeModel, FakeTokenizer):
        enc = ProstT5Encoder(EncodeConfig(device=torch.device("cpu"), dtype="float32"))
        out = enc.encode({"a": "ACG"})
    # Hidden states are position-indexed. For s_len=3, mean of hidden[1:4] = 2.0.
    # If the slice wrongly started at 0 (prefix included), it'd be 1.5.
    np.testing.assert_allclose(out["a"], np.full(4, 2.0, dtype=np.float32))


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

    resolved = resolve_store(embedding_dir=out, fasta_paths=None)
    np.testing.assert_array_equal(np.asarray(resolved.embeddings), embeddings)


def test_resolve_store_errors_on_partial_dir(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "ids.txt").write_text("a\n")
    with pytest.raises(FileExistsError):
        resolve_store(embedding_dir=partial, fasta_paths=None)


def test_resolve_store_encodes_and_caches_when_dir_missing(tmp_path):
    FakeModel, FakeTokenizer = _make_fake_hf_modules(dim=4)
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">cath|1|alpha/1-3\nACG\n")
    cache = tmp_path / "new_cache"

    with _patched_hf_components(FakeModel, FakeTokenizer):
        store = resolve_store(
            embedding_dir=cache,
            fasta_paths=[fasta],
            encode_config=EncodeConfig(device=torch.device("cpu"), dtype="float32"),
        )

    assert store.ids == ["alpha"]
    assert (cache / "embeddings.npy").exists()
    assert (cache / "metadata.json").exists()
