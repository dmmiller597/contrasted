import numpy as np
import torch

from contrasted.search import FaissIndex, VectorIndex, normalize_numpy


def test_normalize_numpy():
    embeddings = np.random.randn(32, 128).astype(np.float32)
    normalized = normalize_numpy(embeddings)

    assert normalized.shape == (32, 128)
    assert normalized.dtype == np.float32

    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_vector_index_search_top1():
    embeddings = torch.eye(3, dtype=torch.float32)
    ids = ["a", "b", "c"]
    index = VectorIndex(embeddings, ids=ids, dtype=torch.float32)

    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scores, indices = index.search(queries, k=1)

    assert indices.tolist() == [[0], [1]]
    assert torch.allclose(scores.squeeze(), torch.tensor([1.0, 1.0]), atol=1e-6)


def test_vector_index_save_load(tmp_path):
    embeddings = torch.randn(5, 4)
    ids = [f"id_{i}" for i in range(5)]
    labels = [f"label_{i}" for i in range(5)]
    index = VectorIndex(embeddings, ids=ids, labels=labels, dtype=torch.float32)

    path = tmp_path / "index.pt"
    index.save(path)
    loaded = VectorIndex.load(path)

    assert len(loaded) == 5
    assert loaded.ids == ids
    assert loaded.labels == labels


def test_vector_index_search_chunked_matches():
    embeddings = torch.randn(128, 32)
    queries = torch.randn(50, 32)
    index = VectorIndex(embeddings, dtype=torch.float32)

    scores_full, indices_full = index.search(queries, k=5)
    scores_chunked, indices_chunked = index.search(queries, k=5, chunk_size=16)

    assert torch.allclose(scores_full, scores_chunked, atol=1e-6)
    assert torch.equal(indices_full, indices_chunked)


def test_faiss_index_matches_vector_index():
    embeddings = torch.eye(3, dtype=torch.float32)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vector_index = VectorIndex(embeddings, dtype=torch.float32)
    faiss_index = FaissIndex(embeddings)

    vec_scores, vec_indices = vector_index.search(queries, k=2)
    faiss_scores, faiss_indices = faiss_index.search(queries, k=2)

    assert faiss_indices.tolist() == vec_indices.tolist()
    assert torch.allclose(faiss_scores, vec_scores, atol=1e-5)


def test_faiss_index_save_load(tmp_path):
    embeddings = torch.randn(5, 4)
    ids = [f"id_{i}" for i in range(5)]
    labels = [f"label_{i}" for i in range(5)]
    index = FaissIndex(embeddings, ids=ids, labels=labels)

    path = tmp_path / "faiss_index.pt"
    index.save(path)
    loaded = FaissIndex.load(path)

    assert len(loaded) == 5
    assert loaded.ids == ids
    assert loaded.labels == labels
