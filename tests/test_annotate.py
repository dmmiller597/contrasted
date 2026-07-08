"""Unit tests for the annotation pipeline helpers."""

import pytest
import torch

from contrasted.annotate import (
    MISSING_EMBEDDING,
    UNKNOWN,
    Prediction,
    _build_annotation_tables,
    attach_true_annotations,
    compute_metrics,
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
    # Confidence stats are over annotated preds only (the lone "X" at 1.0).
    assert s["mean_confidence"] == 1.0
    assert s["median_confidence"] == 1.0


def test_summarize_confidence_stats_empty():
    # No annotated predictions -> stats fall back to 0.0 (no statistics error).
    preds = [Prediction(query_id="b", predicted_annotation=UNKNOWN, confidence=0.0)]
    s = summarize(preds)
    assert s["annotated"] == 0
    assert s["mean_confidence"] == 0.0
    assert s["median_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Unlabeled reference rows (None) must not vote or collide with the sentinel
# ---------------------------------------------------------------------------


def test_build_annotation_tables_none_vs_real_unknown():
    # None -> -1 (cannot vote); a real superfamily literally named "unknown"
    # stays a genuine, decodable class.
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
    # Three neighbours near the query: two unlabeled (None), one labeled "B".
    # The single labeled neighbour must win; the None rows form no class.
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
    # Only 1 of k=3 neighbours is labeled.
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
    # The real label wins by actual voting (full confidence), not by falling
    # through to the no-neighbour sentinel path (which yields confidence 0.0).
    assert preds[0].predicted_annotation == "unknown"
    assert preds[0].confidence == 1.0
    assert preds[0].distance is not None


def test_metrics_do_not_conflate_unlabeled_reference_votes():
    # DB: a labeled "A" cluster plus an unlabeled cluster.
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
    # q_a matches the A cluster; q_u matches only unlabeled rows.
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
    assert by_id["q_u"].predicted_annotation == UNKNOWN  # no labeled neighbour

    # q_a is a correct annotation; q_u is excluded (not scored as a wrong "A").
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
