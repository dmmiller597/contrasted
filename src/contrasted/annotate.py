"""Annotate protein sequences using k-NN or centroid search in a vector index.

The pipeline is a chain of composable stages:

1. ``project`` -- project query embeddings through the head.
2. ``knn_vote`` -- search the domain or centroid index for annotations.
3. ``attach_tmalign_scores`` -- (optional) attach TM-align structural scores.
4. ``write_predictions_tsv`` -- atomically write the results as TSV.

``run`` composes everything from a Hydra config.
"""

import csv
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from contrasted.data import (
    load_domain_ids_from_fasta,
    resolve_fasta_paths,
    resolve_store,
    validate_store_for_projection_head,
)
from contrasted.projection import ProjectionHead, project
from contrasted.search import VectorIndex, as_centroid_index
from contrasted.utils import get_device, load_labels, require_exists

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
MISSING_EMBEDDING = "missing_embedding"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Prediction:
    """One row of the annotation TSV."""

    query_id: str
    predicted_annotation: str
    distance: float | None = None
    confidence: float | None = None
    true_annotation: str | None = None
    tm_score: float | None = None
    rmsd: float | None = None
    tm_coverage: float | None = None
    # Neighbour index used for structural rerank (top-1 DB row, or -1).
    top1_db_idx: int = field(default=-1, repr=False)


# ---------------------------------------------------------------------------
# Stage 1: k-NN vote
# ---------------------------------------------------------------------------


def _build_annotation_tables(
    index: VectorIndex,
    id_to_annotation: dict[str, int],
    idx_to_annotation: dict[int, str],
) -> tuple[np.ndarray, dict[int, str]]:
    """Encode each DB row as an int and return the decoder dict.

    If ``index.labels`` is populated, the index's own label strings are
    interned into a local int vocabulary. Otherwise each DB row's ``id`` is
    looked up in ``id_to_annotation`` and the external ``idx_to_annotation``
    is used as the decoder.

    Returns ``(row_to_ann_int, int_to_string)`` where ``row_to_ann_int[i]``
    is the annotation int for DB row ``i`` (``-1`` for unmapped rows).
    """
    if index.labels is not None:
        vocab: dict[str, int] = {}
        arr = np.array(
            [
                -1 if label is None else vocab.setdefault(label, len(vocab))
                for label in index.labels
            ],
            dtype=np.int64,
        )
        return arr, {v: k for k, v in vocab.items()}

    if index.ids is None:
        return np.full(len(index), -1, dtype=np.int64), dict(idx_to_annotation)

    arr = np.asarray(
        [id_to_annotation.get(db_id, -1) for db_id in index.ids],
        dtype=np.int64,
    )
    return arr, dict(idx_to_annotation)


def _decode(decoder: dict[int, str], ann_int: int) -> str:
    return UNKNOWN if ann_int < 0 else decoder.get(ann_int, UNKNOWN)


def _vote(row: np.ndarray) -> tuple[int, int]:
    """Majority-vote helper.

    ``row`` is a 1-D array of annotation indices (``-1`` for invalid).
    Returns ``(winner_ann_idx, vote_count)``. Ties broken by earliest
    (closest) occurrence.
    """
    valid = row[row >= 0]
    if valid.size == 0:
        return -1, 0
    counts = np.bincount(valid)
    max_count = counts.max()
    # Earliest-occurrence tiebreak: scan the row in order (closest neighbour
    # first) and return the first annotation whose count equals the max. Some
    # element always matches, so the loop is guaranteed to return.
    for ann in valid:
        if counts[ann] == max_count:
            return int(ann), int(max_count)
    raise AssertionError("unreachable: no annotation matched the max count")


def knn_vote(
    vectors: torch.Tensor,
    found_ids: list[str],
    missing_ids: list[str],
    index: VectorIndex,
    *,
    k: int,
    distance_cutoff: float,
    id_to_annotation: dict[str, int],
    idx_to_annotation: dict[int, str],
    search_chunk_size: int = 4096,
) -> list[Prediction]:
    """Search the index and aggregate k-NN votes into ``Prediction``s.

    Missing queries are returned with ``predicted_annotation = 'missing_embedding'``;
    queries whose nearest neighbour exceeds ``distance_cutoff`` are returned as
    ``'unknown'`` with confidence ``0.0``.
    """
    predictions: list[Prediction] = [
        Prediction(query_id=qid, predicted_annotation=MISSING_EMBEDDING)
        for qid in missing_ids
    ]

    if not found_ids:
        return predictions

    similarities, neighbor_rows = index.search(
        vectors, k=k, chunk_size=search_chunk_size
    )
    distances = (1.0 - similarities).cpu().numpy()
    neighbor_rows_np = neighbor_rows.cpu().numpy()

    row_to_ann, decoder = _build_annotation_tables(
        index, id_to_annotation, idx_to_annotation
    )

    # (B, k) int array of annotation indices, -1 for invalid entries.
    ann_idx = np.where(
        neighbor_rows_np >= 0, row_to_ann[neighbor_rows_np.clip(min=0)], -1
    )
    # Mask out neighbours beyond the distance cutoff. A query whose nearest
    # neighbour already exceeds the cutoff has every entry masked to -1, so it
    # votes to UNKNOWN with confidence 0.0 through the general path below.
    ann_idx = np.where(distances > distance_cutoff, -1, ann_idx)

    for j, query_id in enumerate(found_ids):
        winner, count = _vote(ann_idx[j])
        predictions.append(
            Prediction(
                query_id=query_id,
                predicted_annotation=_decode(decoder, winner),
                distance=float(distances[j, 0]),
                confidence=count / k if count else 0.0,
                top1_db_idx=int(neighbor_rows_np[j, 0]),
            )
        )

    return predictions


def attach_true_annotations(
    predictions: list[Prediction],
    id_to_annotation: dict[str, int],
    idx_to_annotation: dict[int, str],
) -> None:
    """Populate ``Prediction.true_annotation`` in place from the label tables.

    Queries absent from ``id_to_annotation`` get ``true_annotation = None`` so
    metrics can exclude them rather than treating missing truth as the
    prediction sentinel ``"unknown"``.
    """
    for p in predictions:
        true_idx = id_to_annotation.get(p.query_id)
        p.true_annotation = (
            idx_to_annotation.get(true_idx) if true_idx is not None else None
        )


# ---------------------------------------------------------------------------
# Stage 2: attach TM-align scores
# ---------------------------------------------------------------------------


def attach_tmalign_scores(
    predictions: list[Prediction],
    index: VectorIndex,
    structure_dir: Path,
    *,
    binary: str = "TMalign",
) -> None:
    """Attach TM-align scores for each prediction's top-1 DB neighbour.

    Populates ``tm_score`` / ``rmsd`` / ``tm_coverage`` in place. Does not
    change predicted labels. Logs and skips predictions whose structures
    cannot be located.
    """
    from contrasted.tmalign import resolve_structure_path, run_tmalign

    if index.ids is None:
        logger.warning("Index has no ids; cannot attach TM-align scores.")
        return

    for p in predictions:
        if p.predicted_annotation in {UNKNOWN, MISSING_EMBEDDING} or p.top1_db_idx < 0:
            continue

        target_id = index.ids[p.top1_db_idx]
        query_struct = resolve_structure_path(p.query_id, structure_dir)
        target_struct = resolve_structure_path(target_id, structure_dir)

        if query_struct is None or target_struct is None:
            missing = [
                name
                for name, path in [
                    (p.query_id, query_struct),
                    (target_id, target_struct),
                ]
                if path is None
            ]
            logger.warning("Structure file(s) not found for: %s", ", ".join(missing))
            continue

        try:
            result = run_tmalign(query_struct, target_struct, binary=binary)
        except (RuntimeError, ValueError) as e:
            logger.warning("TMalign failed for %s vs %s: %s", p.query_id, target_id, e)
            continue

        p.tm_score = result.tm_score
        p.rmsd = result.rmsd
        p.tm_coverage = result.coverage


# ---------------------------------------------------------------------------
# Stage 3: TSV writer
# ---------------------------------------------------------------------------


def _format_optional(value: object | None) -> str:
    return "" if value is None else str(value)


def _active_columns(
    *,
    return_true_annotation: bool,
    return_distance: bool,
    return_confidence: bool,
    include_tmalign: bool,
) -> list[str]:
    """Output TSV column names, in order.

    Each name is also a ``Prediction`` attribute, so the same list drives
    both the header row and per-prediction value lookup (keeping them in sync).
    """
    columns = ["query_id", "predicted_annotation"]
    if return_true_annotation:
        columns.append("true_annotation")
    if return_distance:
        columns.append("distance")
    if return_confidence:
        columns.append("confidence")
    if include_tmalign:
        columns += ["tm_score", "rmsd", "tm_coverage"]
    return columns


def write_predictions_tsv(
    predictions: list[Prediction],
    output_path: Path,
    *,
    return_true_annotation: bool = False,
    return_distance: bool = False,
    return_confidence: bool = False,
    include_tmalign: bool = False,
) -> None:
    """Atomically write predictions to a TSV at ``output_path``."""
    columns = _active_columns(
        return_true_annotation=return_true_annotation,
        return_distance=return_distance,
        return_confidence=return_confidence,
        include_tmalign=include_tmalign,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=output_path.parent, suffix=".tmp", prefix=output_path.stem
    )
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_fd, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(columns)
            for p in predictions:
                writer.writerow(
                    [_format_optional(getattr(p, name)) for name in columns]
                )
        tmp_path.replace(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def summarize(predictions: list[Prediction]) -> dict:
    """Compute counts and confidence summary stats for a list of predictions."""
    unknown = missing = 0
    confidences: list[float] = []
    for p in predictions:
        ann = p.predicted_annotation
        if ann == UNKNOWN:
            unknown += 1
        elif ann == MISSING_EMBEDDING:
            missing += 1
        elif p.confidence is not None:
            confidences.append(p.confidence)
    total = len(predictions)
    annotated = total - unknown - missing
    return {
        "total": total,
        "annotated": annotated,
        "unknown": unknown,
        "missing": missing,
        "mean_confidence": mean(confidences) if confidences else 0.0,
        "median_confidence": median(confidences) if confidences else 0.0,
    }


# ---------------------------------------------------------------------------
# Metrics (used when labels are available)
# ---------------------------------------------------------------------------


def compute_metrics(predictions: list[Prediction]) -> dict[str, float]:
    """Accuracy over predictions with a known truth annotation.

    Queries labelled ``unknown`` / ``missing_embedding`` or lacking a
    ``true_annotation`` are excluded.
    """
    pairs = [
        (p.predicted_annotation, p.true_annotation)
        for p in predictions
        if p.true_annotation
        and p.predicted_annotation not in {UNKNOWN, MISSING_EMBEDDING}
    ]
    if not pairs:
        return {"accuracy": 0.0}
    correct = sum(pred == true for pred, true in pairs)
    return {"accuracy": correct / len(pairs)}


def selective_curve(
    predictions: list[Prediction], *, num_thresholds: int = 50
) -> list[tuple[float, float, float]]:
    """Accuracy vs. coverage as the distance threshold sweeps [min, max].

    The denominator for both metrics is the set of queries that have *both* a
    top-1 distance and a known truth annotation (queries missing either are
    excluded). Coverage at threshold ``t`` is the fraction of that set with
    distance <= ``t``; accuracy is over that covered subset, counting
    ``unknown`` assignments as wrong.
    """
    rows = [
        (p.distance, p.predicted_annotation == p.true_annotation)
        for p in predictions
        if p.distance is not None and p.true_annotation
    ]
    if not rows:
        return []
    distances = np.fromiter((r[0] for r in rows), dtype=np.float64, count=len(rows))
    correct = np.fromiter((r[1] for r in rows), dtype=bool, count=len(rows))
    thresholds = np.linspace(distances.min(), distances.max(), num_thresholds)
    covered = distances[:, None] <= thresholds[None, :]
    n_covered = covered.sum(axis=0)
    coverage = n_covered / len(distances)
    acc = np.divide(
        (correct[:, None] & covered).sum(axis=0),
        n_covered,
        out=np.zeros_like(coverage, dtype=np.float64),
        where=n_covered > 0,
    )
    return [
        (float(t), float(c), float(a))
        for t, c, a in zip(thresholds, coverage, acc, strict=True)
    ]


def write_metrics_and_curve(
    predictions: list[Prediction],
    output_dir: Path,
    input_name: str,
    *,
    num_thresholds: int = 50,
) -> dict[str, float]:
    """Write ``{input_name}`` metrics.json + selective_curve.tsv; return the metrics."""
    metrics = compute_metrics(predictions)
    (output_dir / f"{input_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    curve_path = output_dir / f"{input_name}_selective_curve.tsv"
    with open(curve_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["threshold", "coverage", "accuracy"])
        writer.writerows(selective_curve(predictions, num_thresholds=num_thresholds))
    return metrics


def log_summary(output_path: Path, summary: dict, elapsed: float) -> None:
    """Log the per-input annotation counts and timing."""
    total = summary["total"]
    logger.info(f"Saved annotations to: {output_path}")
    logger.info(f"  Total: {total}")
    if total == 0:
        logger.warning("  No sequences were processed")
        return
    annotated, unknown = summary["annotated"], summary["unknown"]
    logger.info(f"  Annotated: {annotated} ({100 * annotated / total:.1f}%)")
    logger.info(f"  Unknown: {unknown} ({100 * unknown / total:.1f}%)")
    logger.info(f"  Missing: {summary['missing']}")
    logger.info(
        f"  Confidence (annotated): mean={summary['mean_confidence']:.3f} "
        f"median={summary['median_confidence']:.3f}"
    )
    logger.info(f"  Time: {elapsed:.2f}s ({elapsed / total * 1000:.2f}ms per query)")


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    """Annotate protein sequences using k-NN or centroid search."""
    device = get_device()
    logger.info(f"Using device: {device}")

    method = str(cfg.get("method", "knn"))
    if method not in {"knn", "centroid"}:
        raise ValueError(f"method must be 'knn' or 'centroid', got {method!r}")

    tm_align_enabled = bool(cfg.get("tm_align", False))
    if method == "centroid" and tm_align_enabled:
        raise ValueError(
            "method='centroid' cannot be combined with tm_align=true because "
            "centroid ids are superfamily labels, not domain structure ids; "
            "use method='knn' or disable tm_align"
        )

    embedding_dir = cfg.get("embedding_dir")
    if embedding_dir is None or not str(embedding_dir).strip():
        raise ValueError(
            "embedding_dir is required for contrasted-annotate. Pass a store "
            "built by contrasted-build-concat-store or contrasted-embed."
        )

    input_path = Path(cfg.input)
    model_path = Path(cfg.model_path)
    index_path = Path(cfg.index)
    annotation_path = (
        Path(cfg.id_to_annotation) if cfg.get("id_to_annotation") else None
    )

    input_paths = resolve_fasta_paths(input_path)
    if not input_paths:
        raise FileNotFoundError(f"No FASTA files found at: {input_path}")
    require_exists(model_path, "Model checkpoint")
    require_exists(index_path, "Vector index")
    if annotation_path:
        require_exists(annotation_path, "Annotation file")

    structure_dir = Path(cfg.structure_dir) if cfg.get("structure_dir") else None
    tmalign_binary = str(cfg.get("tmalign_binary", "TMalign"))
    if tm_align_enabled:
        from contrasted.tmalign import find_tmalign_binary

        if structure_dir is None:
            raise ValueError("structure_dir must be set when tm_align=true")
        if not structure_dir.is_dir():
            raise FileNotFoundError(f"Structure directory not found: {structure_dir}")
        tmalign_binary = str(
            find_tmalign_binary(None if tmalign_binary == "TMalign" else tmalign_binary)
        )
        logger.info(f"TM-align enabled, binary: {tmalign_binary}")

    logger.info(f"Loading projection head from: {model_path}")
    head = ProjectionHead.load(model_path).to(device)
    head.eval()

    logger.info(f"Loading vector index from: {index_path}")
    index = VectorIndex.load(index_path, device=device)
    logger.info(f"Loaded index with {len(index)} vectors")
    if method == "centroid":
        index = as_centroid_index(index)
        k = 1
        logger.info(
            "Centroid annotation collapsed the index to %d H-centroids and forces k=1",
            len(index),
        )
    else:
        k = int(cfg.k)

    if annotation_path:
        logger.info(f"Loading annotations from: {annotation_path}")
        id_to_annotation, idx_to_annotation = load_labels(annotation_path)
        logger.info(f"Loaded {len(idx_to_annotation)} annotation classes")
    else:
        id_to_annotation, idx_to_annotation = {}, {}

    # A usable annotation source is required: either the index carries labels,
    # or it carries ids that can be joined against a supplied annotation file.
    # Without one, every query would silently resolve to "unknown".
    has_annotation_source = index.labels is not None or (
        index.ids is not None and annotation_path is not None
    )
    if not has_annotation_source:
        raise ValueError(
            "No annotation source: the index has no labels, and no "
            "id_to_annotation file was supplied to join against index ids "
            "(or the index has no ids). Rebuild the index with labels or pass "
            "id_to_annotation=<file>."
        )

    output_dir = Path(cfg.get("output_dir", "outputs/annotations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    store = resolve_store(embedding_dir=embedding_dir)
    validate_store_for_projection_head(
        store,
        input_dim=head.input_dim,
    )

    return_distance = bool(cfg.get("return_distance", True))
    return_confidence = bool(cfg.get("return_confidence", False))
    batch_size = int(cfg.get("batch_size", 2048))
    search_chunk_size = int(cfg.get("search_chunk_size", 4096))
    distance_cutoff = float(cfg.distance_cutoff)

    # Loop-invariant: truth annotations are only attached/written when both
    # requested and a label file was supplied.
    return_true_annotation = bool(
        cfg.get("return_true_annotation", True) and annotation_path
    )
    compute_metrics_enabled = bool(cfg.get("compute_metrics", False))
    num_thresholds = int(cfg.get("num_thresholds", 50))

    for input_name, fasta_path in input_paths.items():
        logger.info(f"Processing: {input_name} ({fasta_path})")
        domain_ids = load_domain_ids_from_fasta(fasta_path)
        logger.info(f"Processing {len(domain_ids)} query sequences")

        start = time.time()

        found_indices, found_ids, missing_ids = store.resolve(domain_ids)
        vectors = project(
            head,
            store.embeddings,
            found_indices,
            device=device,
            batch_size=batch_size,
            desc="Projecting queries",
        )
        predictions = knn_vote(
            vectors,
            found_ids,
            missing_ids,
            index,
            k=k,
            distance_cutoff=distance_cutoff,
            id_to_annotation=id_to_annotation,
            idx_to_annotation=idx_to_annotation,
            search_chunk_size=search_chunk_size,
        )

        if return_true_annotation:
            attach_true_annotations(predictions, id_to_annotation, idx_to_annotation)
        if tm_align_enabled and structure_dir is not None:
            attach_tmalign_scores(
                predictions, index, structure_dir, binary=tmalign_binary
            )

        # Preserve input ordering in the TSV.
        order = {d: i for i, d in enumerate(domain_ids)}
        predictions.sort(key=lambda p: order.get(p.query_id, len(order)))

        output_path = output_dir / f"{input_name}_annotations.tsv"
        write_predictions_tsv(
            predictions,
            output_path,
            return_true_annotation=return_true_annotation,
            return_distance=return_distance,
            return_confidence=return_confidence,
            include_tmalign=tm_align_enabled,
        )

        if compute_metrics_enabled and return_true_annotation:
            metrics = write_metrics_and_curve(
                predictions, output_dir, input_name, num_thresholds=num_thresholds
            )
            logger.info(f"Metrics: {metrics}")

        log_summary(output_path, summarize(predictions), time.time() - start)


@hydra.main(version_base=None, config_path="pkg://configs", config_name="annotate")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI wrapper
    run(cfg)


if __name__ == "__main__":
    main()
