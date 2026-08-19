"""Unit tests for the annotation pipeline helpers."""

import pytest
import torch
from omegaconf import DictConfig

from contrasted.annotate import (
    MISSING_EMBEDDING,
    UNKNOWN,
    Prediction,
    _build_annotation_tables,
    attach_true_annotations,
    compute_metrics,
    knn_vote,
    run,
    summarize,
    write_predictions_tsv,
)
from contrasted.search import VectorIndex, as_centroid_index


def test_run_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be 'knn' or 'centroid'"):
        run(DictConfig({"method": "unsupported"}))


def test_run_rejects_centroid_with_tmalign():
    with pytest.raises(
        ValueError,
        match=r"centroid.*superfamily.*domain structure ids",
    ):
        run(DictConfig({"method": "centroid", "tm_align": True}))


def test_run_requires_embedding_dir():
    with pytest.raises(
        ValueError,
        match="embedding_dir is required for contrasted-annotate",
    ):
        run(DictConfig({"method": "knn", "embedding_dir": None}))


def test_knn_vote_majority_with_labels():
    embs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.95, 0.05, 0.0],
        ]
    )
    index = VectorIndex(embs, ids=["x0", "x1", "x2", "x3"], labels=["A", "A", "B", "B"])

    queries = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    predictions = knn_vote(
        queries,
        ["q_A", "q_B"],
        ["q_missing"],
        index,
        k=2,
        distance_cutoff=1.0,
        id_to_annotation={},
        idx_to_annotation={},
    )

    assert len(predictions) == 3
    by_id = {p.query_id: p for p in predictions}
    assert by_id["q_missing"].predicted_annotation == MISSING_EMBEDDING
    assert by_id["q_A"].predicted_annotation == "A"
    assert by_id["q_B"].predicted_annotation == "B"
    assert by_id["q_A"].confidence == 1.0
    assert by_id["q_A"].distance is not None


def test_knn_vote_defaults_to_bounded_search_chunks(monkeypatch):
    index = VectorIndex(torch.eye(2), ids=["x0", "x1"], labels=["A", "B"])
    search = index.search
    observed = {}

    def tracked_search(queries, k, *, chunk_size):
        observed["chunk_size"] = chunk_size
        return search(queries, k, chunk_size=chunk_size)

    monkeypatch.setattr(index, "search", tracked_search)

    knn_vote(
        torch.tensor([[1.0, 0.0]]),
        ["q"],
        [],
        index,
        k=1,
        distance_cutoff=1.0,
        id_to_annotation={},
        idx_to_annotation={},
    )

    assert observed["chunk_size"] == 4096


def test_knn_vote_applies_distance_cutoff():
    embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex(embs, ids=["x0", "x1"], labels=["A", "B"])

    queries = torch.tensor([[1.0, 0.0]])

    predictions = knn_vote(
        queries,
        ["q"],
        [],
        index,
        k=1,
        distance_cutoff=-0.5,
        id_to_annotation={},
        idx_to_annotation={},
    )

    assert predictions[0].predicted_annotation == UNKNOWN
    assert predictions[0].confidence == 0.0


def test_centroid_index_predicts_nearest_class_mean():
    index = VectorIndex(
        torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.8, 0.2],
            ]
        ),
        ids=["b0", "a0", "unlabeled", "a1"],
        labels=["B", "A", None, "A"],
    )

    centroid_index = as_centroid_index(index)

    assert len(centroid_index) == 2
    assert centroid_index.ids == ["A", "B"]
    assert centroid_index.labels == ["A", "B"]
    assert centroid_index.embeddings.dtype == torch.float64
    expected_a = index.embeddings[[1, 3]].to(torch.float64).mean(dim=0)
    expected_a /= expected_a.norm()
    assert torch.allclose(centroid_index.embeddings[0], expected_a)
    assert torch.allclose(
        centroid_index.embeddings.norm(dim=1),
        torch.ones(2, dtype=torch.float64),
    )

    predictions = knn_vote(
        expected_a.unsqueeze(0),
        ["query"],
        [],
        centroid_index,
        k=1,
        distance_cutoff=0.2,
        id_to_annotation={},
        idx_to_annotation={},
    )

    assert predictions[0].predicted_annotation == "A"


def test_centroid_index_respects_distance_cutoff():
    centroid_index = as_centroid_index(
        VectorIndex(torch.tensor([[1.0, 0.0]]), labels=["A"])
    )

    predictions = knn_vote(
        torch.tensor([[0.0, 1.0]]),
        ["query"],
        [],
        centroid_index,
        k=1,
        distance_cutoff=0.5,
        id_to_annotation={},
        idx_to_annotation={},
    )

    assert predictions[0].predicted_annotation == UNKNOWN
    assert predictions[0].confidence == 0.0


@pytest.mark.parametrize("labels", [None, [None, None]])
def test_centroid_index_rejects_missing_labeled_rows(labels):
    index = VectorIndex(torch.eye(2), labels=labels)

    with pytest.raises(ValueError, match=r"requires labels|labeled row"):
        as_centroid_index(index)


def test_knn_vote_uses_id_to_annotation_when_no_labels():
    embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex(embs, ids=["ref0", "ref1"])

    queries = torch.tensor([[1.0, 0.0]])

    predictions = knn_vote(
        queries,
        ["q"],
        [],
        index,
        k=1,
        distance_cutoff=1.0,
        id_to_annotation={"ref0": 7, "ref1": 8},
        idx_to_annotation={7: "sf_seven", 8: "sf_eight"},
    )

    assert predictions[0].predicted_annotation == "sf_seven"


def test_write_predictions_tsv_atomic(tmp_path):
    preds = [
        Prediction(
            query_id="q1", predicted_annotation="A", distance=0.1, confidence=1.0
        ),
        Prediction(
            query_id="q2", predicted_annotation=UNKNOWN, distance=0.9, confidence=0.0
        ),
    ]
    out = tmp_path / "out.tsv"
    write_predictions_tsv(preds, out, return_distance=True, return_confidence=True)
    lines = out.read_text().strip().splitlines()
    assert lines[0].split("\t") == [
        "query_id",
        "predicted_annotation",
        "distance",
        "confidence",
    ]
    assert lines[1].split("\t") == ["q1", "A", "0.1", "1.0"]
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_summarize_counts():
    preds = [
        Prediction(query_id="a", predicted_annotation="X", confidence=1.0),
        Prediction(query_id="b", predicted_annotation=UNKNOWN, confidence=0.0),
        Prediction(query_id="c", predicted_annotation=MISSING_EMBEDDING),
    ]
    s = summarize(preds)
    assert s["total"] == 3
    assert s["annotated"] == 1
    assert s["unknown"] == 1
    assert s["missing"] == 1
    assert s["mean_confidence"] == 1.0
    assert s["median_confidence"] == 1.0


def test_summarize_confidence_stats_empty():
    preds = [Prediction(query_id="b", predicted_annotation=UNKNOWN, confidence=0.0)]
    s = summarize(preds)
    assert s["annotated"] == 0
    assert s["mean_confidence"] == 0.0
    assert s["median_confidence"] == 0.0


def test_build_annotation_tables_none_vs_real_unknown():
    index = VectorIndex(
        torch.randn(3, 4), ids=["a", "b", "c"], labels=["unknown", None, "B"]
    )
    row_to_ann, decoder = _build_annotation_tables(index, {}, {})
    assert row_to_ann[1] == -1
    assert row_to_ann[0] >= 0
    assert decoder[int(row_to_ann[0])] == "unknown"
    assert decoder[int(row_to_ann[2])] == "B"
    assert None not in decoder.values()


def test_unlabeled_db_rows_do_not_vote():
    embs = torch.tensor([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.98, 0.02, 0.0]])
    index = VectorIndex(embs, ids=["u0", "u1", "b0"], labels=[None, None, "B"])
    preds = knn_vote(
        torch.tensor([[1.0, 0.0, 0.0]]),
        ["q"],
        [],
        index,
        k=3,
        distance_cutoff=1.0,
        id_to_annotation={},
        idx_to_annotation={},
    )
    assert preds[0].predicted_annotation == "B"
    assert preds[0].confidence == pytest.approx(1 / 3)


def test_all_unlabeled_neighbours_resolve_unknown():
    embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex(embs, ids=["u0", "u1"], labels=[None, None])
    preds = knn_vote(
        torch.tensor([[1.0, 0.0]]),
        ["q"],
        [],
        index,
        k=2,
        distance_cutoff=1.0,
        id_to_annotation={},
        idx_to_annotation={},
    )
    assert preds[0].predicted_annotation == UNKNOWN
    assert preds[0].confidence == 0.0


def test_real_unknown_label_still_votes():
    embs = torch.tensor([[1.0, 0.0], [0.99, 0.01]])
    index = VectorIndex(embs, ids=["x0", "x1"], labels=["unknown", "unknown"])
    preds = knn_vote(
        torch.tensor([[1.0, 0.0]]),
        ["q"],
        [],
        index,
        k=2,
        distance_cutoff=1.0,
        id_to_annotation={},
        idx_to_annotation={},
    )
    assert preds[0].predicted_annotation == "unknown"
    assert preds[0].confidence == 1.0
    assert preds[0].distance is not None


def test_metrics_do_not_conflate_unlabeled_reference_votes():
    embs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.01],
        ]
    )
    index = VectorIndex(
        embs, ids=["a0", "a1", "u0", "u1"], labels=["A", "A", None, None]
    )
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    id_to_ann = {"q_a": 0, "q_u": 0}
    idx_to_ann = {0: "A"}
    preds = knn_vote(
        queries,
        ["q_a", "q_u"],
        [],
        index,
        k=2,
        distance_cutoff=1.0,
        id_to_annotation=id_to_ann,
        idx_to_annotation=idx_to_ann,
    )
    attach_true_annotations(preds, id_to_ann, idx_to_ann)
    by_id = {p.query_id: p for p in preds}
    assert by_id["q_a"].predicted_annotation == "A"
    assert by_id["q_u"].predicted_annotation == UNKNOWN

    assert compute_metrics(preds) == {"accuracy": 1.0}
    s = summarize(preds)
    assert s["annotated"] == 1
    assert s["unknown"] == 1


def test_partial_truth_excludes_queries_without_labels():
    preds = [
        Prediction(
            query_id="labeled",
            predicted_annotation="A",
            true_annotation=None,
        ),
        Prediction(
            query_id="unlabeled",
            predicted_annotation="B",
            true_annotation=None,
        ),
    ]
    id_to_ann = {"labeled": 0}
    idx_to_ann = {0: "A"}
    attach_true_annotations(preds, id_to_ann, idx_to_ann)

    by_id = {p.query_id: p for p in preds}
    assert by_id["labeled"].true_annotation == "A"
    assert by_id["unlabeled"].true_annotation is None

    assert compute_metrics(preds) == {"accuracy": 1.0}
