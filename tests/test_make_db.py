"""Unit tests for make_db label resolution."""

from contrasted.make_db import resolve_labels


def test_resolve_labels_preserves_missing_as_none():
    id_to_label = {"a": 0, "b": 1, "c": 0}
    idx_to_label = {0: "SF_A", 1: "SF_B"}
    labels = resolve_labels(["a", "x", "b", "c"], id_to_label, idx_to_label)
    assert labels == ["SF_A", None, "SF_B", "SF_A"]


def test_resolve_labels_keeps_real_unknown_string():
    # A superfamily literally named "unknown" is a real label, not the sentinel.
    id_to_label = {"a": 0}
    idx_to_label = {0: "unknown"}
    labels = resolve_labels(["a", "missing"], id_to_label, idx_to_label)
    assert labels == ["unknown", None]
