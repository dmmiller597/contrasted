"""Annotate protein sequences using k-NN search in a vector index.

The pipeline is split into four composable stages:

1. :func:`project_queries` -- project query embeddings through the trained model.
2. :func:`knn_vote` -- search the index and aggregate neighbour annotations.
3. :func:`rerank_with_tmalign` -- (optional) add TM-align structural scores.
4. :func:`write_predictions_tsv` -- atomically write the results as TSV.

:func:`run` composes all four from a Hydra config.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from contrasted.data import (
    EmbeddingStore,
    load_domain_ids_from_fasta,
    resolve_fasta_paths,
)
from contrasted.model import ContrastiveModel
from contrasted.search import VectorIndex
from contrasted.tmalign import (
    find_tmalign_binary,
    resolve_structure_path,
    run_tmalign,
)
from contrasted.utils import get_device, load_labels

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
# Stage 1: projection
# ---------------------------------------------------------------------------


@torch.inference_mode()
def project_queries(
    model: ContrastiveModel,
    store: EmbeddingStore,
    domain_ids: list[str],
    device: torch.device,
    *,
    batch_size: int = 2048,
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Project queries through the model in chunks.

    Returns ``(vectors, found_ids, missing_ids)``.
    """
    model.eval()
    found_indices, found_ids, missing_ids = store.resolve(domain_ids)
    if not found_indices:
        return torch.empty(0, 0), [], missing_ids

    chunks: list[torch.Tensor] = []
    for i in tqdm(
        range(0, len(found_indices), batch_size), desc="Projecting", leave=False
    ):
        rows = found_indices[i : i + batch_size]
        raw = np.ascontiguousarray(store.embeddings[rows])
        batch = torch.as_tensor(raw).float().to(device)
        chunks.append(model(batch).cpu())

    return torch.cat(chunks, dim=0), found_ids, missing_ids


# ---------------------------------------------------------------------------
# Stage 2: k-NN vote
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
        arr = np.empty(len(index.labels), dtype=np.int64)
        for i, label in enumerate(index.labels):
            if label not in vocab:
                vocab[label] = len(vocab)
            arr[i] = vocab[label]
        decoder = {v: k for k, v in vocab.items()}
        return arr, decoder

    if index.ids is None:
        return np.full(len(index), -1, dtype=np.int64), dict(idx_to_annotation)

    arr = np.asarray(
        [id_to_annotation.get(db_id, -1) for db_id in index.ids],
        dtype=np.int64,
    )
    return arr, dict(idx_to_annotation)


def _decode(decoder: dict[int, str], ann_int: int) -> str:
    if ann_int < 0:
        return UNKNOWN
    return decoder.get(ann_int, UNKNOWN)


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
    # Earliest-occurrence tiebreak: scan the row in order and return the
    # first annotation whose count equals the max.
    for ann in valid:
        if counts[ann] == max_count:
            return int(ann), int(max_count)
    # Unreachable.
    return int(valid[0]), int(max_count)


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
    search_chunk_size: int | None = None,
) -> list[Prediction]:
    """Search the index and aggregate k-NN votes into :class:`Prediction`s.

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
    # Mask out neighbours beyond the distance cutoff.
    ann_idx = np.where(distances > distance_cutoff, -1, ann_idx)

    for j, query_id in enumerate(found_ids):
        best_distance = float(distances[j, 0])

        if best_distance > distance_cutoff:
            predictions.append(
                Prediction(
                    query_id=query_id,
                    predicted_annotation=UNKNOWN,
                    distance=best_distance,
                    confidence=0.0,
                    top1_db_idx=int(neighbor_rows_np[j, 0]),
                )
            )
            continue

        winner, count = _vote(ann_idx[j])
        predictions.append(
            Prediction(
                query_id=query_id,
                predicted_annotation=_decode(decoder, winner),
                distance=best_distance,
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
    """Populate ``Prediction.true_annotation`` in place from the label tables."""
    for p in predictions:
        true_idx = id_to_annotation.get(p.query_id, -1)
        p.true_annotation = idx_to_annotation.get(true_idx, UNKNOWN)


# ---------------------------------------------------------------------------
# Stage 3: TM-align rerank
# ---------------------------------------------------------------------------


def rerank_with_tmalign(
    predictions: list[Prediction],
    index: VectorIndex,
    structure_dir: Path,
    *,
    binary: str = "TMalign",
) -> None:
    """Attach TM-align scores for each prediction's top-1 DB neighbour.

    Populates ``tm_score`` / ``rmsd`` / ``tm_coverage`` in place; logs and
    skips predictions whose structures cannot be located.
    """
    if index.ids is None:
        logger.warning("Index has no ids; cannot run TM-align rerank.")
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
# Stage 4: TSV writer
# ---------------------------------------------------------------------------


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
    headers = ["query_id", "predicted_annotation"]
    if return_true_annotation:
        headers.append("true_annotation")
    if return_distance:
        headers.append("distance")
    if return_confidence:
        headers.append("confidence")
    if include_tmalign:
        headers.extend(["tm_score", "rmsd", "tm_coverage"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=output_path.parent, suffix=".tmp", prefix=output_path.stem
    )
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_fd, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(headers)
            for p in predictions:
                row = [p.query_id, p.predicted_annotation]
                if return_true_annotation:
                    row.append(p.true_annotation or UNKNOWN)
                if return_distance:
                    row.append("" if p.distance is None else f"{p.distance}")
                if return_confidence:
                    row.append("" if p.confidence is None else f"{p.confidence}")
                if include_tmalign:
                    row.extend(
                        [
                            "" if p.tm_score is None else f"{p.tm_score}",
                            "" if p.rmsd is None else f"{p.rmsd}",
                            "" if p.tm_coverage is None else f"{p.tm_coverage}",
                        ]
                    )
                writer.writerow(row)
        tmp_path.replace(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def summarize(predictions: list[Prediction]) -> dict:
    """Compute counts/confidences summary for a list of predictions."""
    unknown = sum(1 for p in predictions if p.predicted_annotation == UNKNOWN)
    missing = sum(1 for p in predictions if p.predicted_annotation == MISSING_EMBEDDING)
    annotated = len(predictions) - unknown - missing
    confidences = [
        p.confidence
        for p in predictions
        if p.confidence is not None
        and p.predicted_annotation not in {UNKNOWN, MISSING_EMBEDDING}
    ]
    return {
        "total": len(predictions),
        "annotated": annotated,
        "unknown": unknown,
        "missing": missing,
        "confidences": confidences,
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

    correct = sum(1 for p, t in pairs if p == t)
    return {"accuracy": correct / len(pairs)}


def selective_curve(
    predictions: list[Prediction], *, num_thresholds: int = 50
) -> list[tuple[float, float, float]]:
    """Accuracy vs. coverage as the distance threshold sweeps [min, max].

    Coverage is over *non-missing* queries; accuracy is over queries whose
    top-1 distance is below the threshold (including ``unknown`` assignments,
    which count as wrong when a truth label is known).
    """
    rows = [
        (p.distance, p.predicted_annotation == p.true_annotation)
        for p in predictions
        if p.distance is not None and p.true_annotation
    ]
    if not rows:
        return []
    distances = np.asarray([r[0] for r in rows], dtype=np.float64)
    correct = np.asarray([r[1] for r in rows], dtype=bool)
    thresholds = np.linspace(distances.min(), distances.max(), num_thresholds)
    out: list[tuple[float, float, float]] = []
    for t in thresholds:
        mask = distances <= t
        coverage = float(mask.mean())
        acc = float(correct[mask].mean()) if mask.any() else 0.0
        out.append((float(t), coverage, acc))
    return out


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    """Annotate protein sequences using k-NN search (Hydra config)."""
    device = get_device()
    logger.info(f"Using device: {device}")

    input_path = Path(cfg.input)
    model_path = Path(cfg.model_path)
    index_path = Path(cfg.index)
    annotation_path = (
        Path(cfg.id_to_annotation) if cfg.get("id_to_annotation") else None
    )

    input_paths = resolve_fasta_paths(input_path)
    if not input_paths:
        raise FileNotFoundError(f"No FASTA files found at: {input_path}")
    for path, name in [
        (model_path, "Model checkpoint"),
        (index_path, "Vector index"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if annotation_path and not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    tm_align_enabled = bool(cfg.get("tm_align", False))
    structure_dir = Path(cfg.structure_dir) if cfg.get("structure_dir") else None
    tmalign_binary = str(cfg.get("tmalign_binary", "TMalign"))
    if tm_align_enabled:
        if structure_dir is None:
            raise ValueError("structure_dir must be set when tm_align=true")
        if not structure_dir.is_dir():
            raise FileNotFoundError(f"Structure directory not found: {structure_dir}")
        binary_path = find_tmalign_binary(
            tmalign_binary if tmalign_binary != "TMalign" else None
        )
        tmalign_binary = str(binary_path)
        logger.info(f"TM-align enabled, binary: {tmalign_binary}")

    logger.info(f"Loading model from: {model_path}")
    model = ContrastiveModel.load_from_checkpoint(
        str(model_path), strict=False, weights_only=False
    )
    model.eval().to(device)

    logger.info(f"Loading vector index from: {index_path}")
    index = VectorIndex.load(index_path, device=device)
    logger.info(f"Loaded index with {len(index)} vectors")

    if annotation_path:
        logger.info(f"Loading annotations from: {annotation_path}")
        id_to_annotation, idx_to_annotation = load_labels(annotation_path)
        logger.info(f"Loaded {len(idx_to_annotation)} annotation classes")
    else:
        id_to_annotation, idx_to_annotation = {}, {}

    output_dir = Path(cfg.get("output_dir", "outputs/annotations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_dir = Path(cfg.get("embedding_dir", "data/cath-c123-S100"))
    store = EmbeddingStore.from_dir(embedding_dir)

    return_distance = bool(cfg.get("return_distance", True))
    return_confidence = bool(cfg.get("return_confidence", False))
    batch_size = int(cfg.get("batch_size", 2048))
    search_chunk_size = cfg.get("search_chunk_size")
    k = int(cfg.k)
    distance_cutoff = float(cfg.distance_cutoff)

    for input_name, fasta_path in input_paths.items():
        logger.info(f"Processing: {input_name} ({fasta_path})")
        domain_ids = load_domain_ids_from_fasta(fasta_path)
        logger.info(f"Processing {len(domain_ids)} query sequences")

        start = time.time()

        vectors, found_ids, missing_ids = project_queries(
            model, store, domain_ids, device, batch_size=batch_size
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

        return_true_annotation = bool(
            cfg.get("return_true_annotation", True) and annotation_path
        )
        if return_true_annotation:
            attach_true_annotations(predictions, id_to_annotation, idx_to_annotation)
        if tm_align_enabled and structure_dir is not None:
            rerank_with_tmalign(
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

        # Emit metrics + selective curve when truth labels are available.
        if cfg.get("compute_metrics", False) and return_true_annotation:
            metrics = compute_metrics(predictions)
            (output_dir / f"{input_name}_metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
            curve_path = output_dir / f"{input_name}_selective_curve.tsv"
            with open(curve_path, "w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["threshold", "coverage", "accuracy"])
                for row in selective_curve(
                    predictions, num_thresholds=cfg.get("num_thresholds", 50)
                ):
                    w.writerow(row)
            logger.info(f"Metrics: {metrics}")

        elapsed = time.time() - start
        summary = summarize(predictions)
        total = summary["total"]
        logger.info(f"Saved annotations to: {output_path}")
        logger.info(f"  Total: {total}")
        if total > 0:
            logger.info(
                f"  Annotated: {summary['annotated']} "
                f"({100 * summary['annotated'] / total:.1f}%)"
            )
            logger.info(
                f"  Unknown: {summary['unknown']} "
                f"({100 * summary['unknown'] / total:.1f}%)"
            )
            logger.info(f"  Missing: {summary['missing']}")
            logger.info(
                f"  Time: {elapsed:.2f}s ({elapsed / total * 1000:.2f}ms per query)"
            )
        else:
            logger.warning("  No sequences were processed")


@hydra.main(version_base=None, config_path="../../configs", config_name="annotate")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI wrapper
    run(cfg)


if __name__ == "__main__":
    main()
