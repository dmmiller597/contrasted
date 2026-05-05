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
