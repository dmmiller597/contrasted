"""Unit tests for the annotation pipeline helpers."""

import torch

from contrasted.annotate import (
    MISSING_EMBEDDING,
    UNKNOWN,
    Prediction,
    knn_vote,
    summarize,
    write_predictions_tsv,
)
from contrasted.search import VectorIndex


def test_knn_vote_majority_with_labels():
    # Build an index with 4 embeddings assigned to 2 labels.
    embs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # near query 0
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.95, 0.05, 0.0],
        ]
    )
    index = VectorIndex(embs, ids=["x0", "x1", "x2", "x3"], labels=["A", "A", "B", "B"])

    # Two queries: one close to A-cluster, one close to B-cluster.
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


def test_knn_vote_applies_distance_cutoff():
    embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex(embs, ids=["x0", "x1"], labels=["A", "B"])

    queries = torch.tensor([[1.0, 0.0]])

    # Cutoff of 0.0 means even an exact-match (distance=0) passes, but any
    # further neighbour is ruled out. Keep cutoff tight to force UNKNOWN.
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


def test_knn_vote_uses_id_to_annotation_when_no_labels():
    embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex(embs, ids=["ref0", "ref1"])  # no labels on the index

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
    # No leftover .tmp file.
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
    assert s["confidences"] == [1.0]
