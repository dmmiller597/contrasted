import pytest
import torch

from contrasted.search import VectorIndex


def test_vector_index_search_top1():
    embeddings = torch.eye(3, dtype=torch.float32)
    ids = ["a", "b", "c"]
    index = VectorIndex(embeddings, ids=ids)

    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scores, indices = index.search(queries, k=1)

    assert indices.tolist() == [[0], [1]]
    assert torch.allclose(scores.squeeze(), torch.tensor([1.0, 1.0]), atol=1e-6)


def test_vector_index_save_load(tmp_path):
    embeddings = torch.randn(5, 4)
    ids = [f"id_{i}" for i in range(5)]
    labels = [f"label_{i}" for i in range(5)]
    index = VectorIndex(embeddings, ids=ids, labels=labels)

    path = tmp_path / "index.pt"
    index.save(path)
    loaded = VectorIndex.load(path)

    assert len(loaded) == 5
    assert loaded.ids == ids
    assert loaded.labels == labels


def test_vector_index_save_load_preserves_none_labels(tmp_path):
    # The weights_only=True load path must round-trip None (unlabeled) entries.
    index = VectorIndex(
        torch.randn(3, 4), ids=["a", "b", "c"], labels=["sf_a", None, "sf_c"]
    )
    path = tmp_path / "index.pt"
    index.save(path)
    loaded = VectorIndex.load(path)

    assert loaded.labels == ["sf_a", None, "sf_c"]
    assert loaded.labels[1] is None


def test_search_empty_index_raises():
    index = VectorIndex(torch.empty(0, 4))
    with pytest.raises(ValueError, match="empty index"):
        index.search(torch.randn(2, 4), k=1)


def test_search_rejects_nonpositive_k():
    index = VectorIndex(torch.eye(3))
    with pytest.raises(ValueError, match="k must be positive"):
        index.search(torch.randn(1, 3), k=0)


def test_search_empty_queries_ok():
    index = VectorIndex(torch.eye(3))
    scores, indices = index.search(torch.empty(0, 3), k=2)
    assert scores.shape == (0, 2)
    assert indices.shape == (0, 2)
