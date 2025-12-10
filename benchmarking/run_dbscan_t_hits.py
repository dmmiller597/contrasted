#!/usr/bin/env python
"""Cluster CATH + t_hits in shared space and flag clusters without CATH members."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import cupy as cp
    from cuml.cluster import DBSCAN as cuDBSCAN
    CUML_AVAILABLE = True
except ImportError as e:
    CUML_AVAILABLE = False
    print(f"Warning: cuML not available ({e}). Falling back to CPU DBSCAN.")
    print("To install cuML:")
    print("  Option 1 (conda, recommended): conda install -c rapidsai -c conda-forge -c nvidia cuml cudf")
    print("  Option 2 (pip): pip install --extra-index-url https://pypi.nvidia.com cuml-cu11")

from benchmarking._utils import l2_normalize, get_device
from contrasted.model import CathSupConModel
from contrasted.utils import (
    EmbeddingReader,
    load_h5_keys_from_fasta,
    load_labels,
    extract_domain_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project CATH (H5) and t_hits (LMDB) embeddings through a trained model, "
            "cluster with DBSCAN, and report clusters without CATH members."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/holdouts50.ckpt"))
    parser.add_argument("--cath_h5", type=Path, default=Path("data/cath-domain-seqs-S100.h5"))
    parser.add_argument("--cath_fasta", type=Path, default=Path("data/cath-c123-S100.fasta"))
    parser.add_argument("--labels_file", type=Path, default=Path("data/cath-domain-sf-list.txt"))
    parser.add_argument("--thits_lmdb", type=Path, default=Path("data/t-hits-lmdb"))
    parser.add_argument(
        "--thits_fasta",
        type=Path,
        default=None,
        help="Optional FASTA to select a subset of t_hits (headers -> LMDB keys).",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/t_hits/clustering/"))
    parser.add_argument("--eps", type=float, default=0.14)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--min_cluster_size", type=int, default=5, help="Min members for a cluster to be considered valid.")
    parser.add_argument("--read_batch_size", type=int, default=8192, help="Batch size when reading t_hits LMDB.")
    parser.add_argument("--project_batch_size", type=int, default=4096, help="Batch size for model projection.")
    parser.add_argument("--memmap_path", type=Path, default=None, help="Optional override for projected memmap path.")
    parser.add_argument("--limit_thits", type=int, default=None, help="Optional limit on t_hits embeddings (dry run).")
    return parser.parse_args()


def _alloc_memmap(path: Path, n_rows: int, dim: int) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=np.float32, mode="w+", shape=(n_rows, dim))


@torch.no_grad()
def project_cath(
    model: CathSupConModel,
    cath_h5: Path,
    cath_fasta: Path,
    label_map: dict,
    memmap: np.memmap,
    start_idx: int,
    project_batch_size: int,
) -> Tuple[int, List[int], List[str]]:
    """Project CATH embeddings and write into memmap starting at start_idx."""
    keys = load_h5_keys_from_fasta(cath_fasta)
    reader = EmbeddingReader(cath_h5)
    device = next(model.parameters()).device
    labels: List[int] = []
    domain_ids: List[str] = []
    written = 0

    for i in tqdm(range(0, len(keys), project_batch_size), desc="Projecting CATH", unit="batch"):
        batch_keys = keys[i : i + project_batch_size]
        batch_embs = []
        batch_ids = []
        for k, emb in reader.get_embeddings_batch(batch_keys):
            if emb is None:
                continue
            batch_embs.append(emb.astype(np.float32, copy=False))
            batch_ids.append(extract_domain_id(k))
        if not batch_embs:
            continue

        batch_tensor = torch.from_numpy(np.stack(batch_embs)).to(device)
        proj = model(batch_tensor).cpu().numpy().astype(np.float32)
        proj = l2_normalize(proj)

        end = start_idx + written + len(proj)
        memmap[start_idx + written : end] = proj
        labels.extend([label_map.get(did, -1) for did in batch_ids])
        domain_ids.extend(batch_ids)
        written += len(proj)

    reader.close()
    return written, labels, domain_ids


@torch.no_grad()
def project_thits(
    model: CathSupConModel,
    thits_reader: EmbeddingReader,
    memmap: np.memmap,
    start_idx: int,
    read_batch_size: int,
    project_batch_size: int,
    limit: Optional[int],
    keys_filter: Optional[List[str]] = None,
) -> Tuple[int, List[int], List[str]]:
    """Project t_hits LMDB embeddings and write into memmap after CATH block."""
    device = next(model.parameters()).device
    labels: List[int] = []
    domain_ids: List[str] = []
    written = 0

    # Establish the iteration list if a filter is provided
    if keys_filter is not None:
        keys_filter = list(keys_filter)
        total_allowed = len(keys_filter) if limit is None else min(len(keys_filter), limit)
        key_iter = [keys_filter]
    else:
        total_allowed = len(thits_reader) if limit is None else min(len(thits_reader), limit)
        key_iter = None

    progress = tqdm(total=total_allowed, desc="Projecting t_hits", unit="emb")
    # If filtering, iterate over the explicit key list; else stream entire LMDB
    batches = (
        thits_reader.iter_embeddings(keys=key_iter[0], batch_size=read_batch_size, to_float32=True)
        if key_iter is not None
        else thits_reader.iter_embeddings(batch_size=read_batch_size, to_float32=True)
    )

    for batch in batches:
        if limit is not None and written >= limit:
            break
        batch = [(k, e) for k, e in batch if e is not None]
        if not batch:
            continue

        for j in range(0, len(batch), project_batch_size):
            if limit is not None and written >= limit:
                break
            chunk = batch[j : j + project_batch_size]
            keys, embs = zip(*chunk)

            remaining = total_allowed - written
            keys = keys[:remaining]
            embs = embs[:remaining]

            batch_tensor = torch.from_numpy(np.stack(embs)).to(device)
            proj = model(batch_tensor).cpu().numpy().astype(np.float32)
            proj = l2_normalize(proj)

            end = start_idx + written + len(proj)
            memmap[start_idx + written : end] = proj
            labels.extend([-1] * len(proj))  # unknown superfamily
            domain_ids.extend(keys)
            written += len(proj)
            progress.update(len(proj))

            if limit is not None and written >= limit:
                break

    progress.close()
    return written, labels, domain_ids


def summarize_clusters(
    cluster_labels: np.ndarray,
    is_cath_mask: np.ndarray,
    min_cluster_size: int,
    domain_ids: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cluster_summary, novel_clusters) dataframes."""
    summary_rows = []
    novel_rows = []

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for cid, size in zip(unique_labels.tolist(), counts.tolist()):
        if cid == -1:
            continue
        mask = cluster_labels == cid
        cath_count = int(is_cath_mask[mask].sum())
        thits_count = int(size - cath_count)
        cath_fraction = cath_count / size if size else 0.0
        summary_rows.append(
            {
                "cluster_id": cid,
                "size": size,
                "cath_count": cath_count,
                "thits_count": thits_count,
                "cath_fraction": cath_fraction,
            }
        )
        if size >= min_cluster_size and cath_count == 0:
            sample_ids = domain_ids[np.where(mask)[0][:10]].tolist() if hasattr(domain_ids, "__getitem__") else []
            novel_rows.append(
                {
                    "cluster_id": cid,
                    "size": size,
                    "thits_count": thits_count,
                    "sample_domain_ids": ";".join(sample_ids),
                }
            )

    cluster_df = pd.DataFrame(summary_rows).sort_values("size", ascending=False)
    novel_df = pd.DataFrame(novel_rows).sort_values("size", ascending=False)
    return cluster_df, novel_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    memmap_path = args.memmap_path or args.output_dir / "projected_embeddings.memmap"
    labels_path = args.output_dir / "dbscan_labels.npy"
    domain_ids_path = args.output_dir / "domain_ids.npy"
    cluster_summary_path = args.output_dir / "cluster_summary.csv"
    novel_clusters_path = args.output_dir / "novel_clusters.csv"
    summary_path = args.output_dir / "summary.json"

    device = get_device()
    model = CathSupConModel.load_from_checkpoint(
        str(args.checkpoint),
        map_location=device,
        strict=False,
    ).eval().to(device)

    id_to_idx, _ = load_labels(args.labels_file)
    cath_keys = load_h5_keys_from_fasta(args.cath_fasta)
    thits_reader = EmbeddingReader(args.thits_lmdb)

    # Optional subset of t_hits from FASTA headers
    if args.thits_fasta is not None:
        thits_keys = load_h5_keys_from_fasta(args.thits_fasta)
    else:
        thits_keys = None

    if thits_keys is not None:
        thits_n = len(thits_keys) if args.limit_thits is None else min(len(thits_keys), args.limit_thits)
    else:
        thits_n = len(thits_reader) if args.limit_thits is None else min(len(thits_reader), args.limit_thits)

    total_estimate = len(cath_keys) + thits_n

    # Determine projection dimension from a single forward pass
    dummy = torch.zeros((1, model.hparams.input_dim), dtype=torch.float32, device=device)
    proj_dim = model(dummy).shape[1]
    memmap = _alloc_memmap(memmap_path, total_estimate, proj_dim)

    # Project CATH block
    cath_written, cath_labels, cath_ids = project_cath(
        model=model,
        cath_h5=args.cath_h5,
        cath_fasta=args.cath_fasta,
        label_map=id_to_idx,
        memmap=memmap,
        start_idx=0,
        project_batch_size=args.project_batch_size,
    )

    # Project t_hits block
    thits_start = cath_written
    thits_written, thits_labels, thits_ids = project_thits(
        model=model,
        thits_reader=thits_reader,
        memmap=memmap,
        start_idx=thits_start,
        read_batch_size=args.read_batch_size,
        project_batch_size=args.project_batch_size,
        limit=args.limit_thits,
        keys_filter=thits_keys,
    )

    total_written = cath_written + thits_written
    memmap.flush()

    # Run DBSCAN on written slice
    emb_view = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(total_estimate, proj_dim))
    embeddings = emb_view[:total_written]
    
    if CUML_AVAILABLE and torch.cuda.is_available():
        print(f"\n=== Running GPU DBSCAN (cuML) on {total_written:,} embeddings ===")
        # Convert to cupy array for GPU processing
        # cuML DBSCAN with cosine metric uses inner product for normalized vectors
        # (cosine distance = 1 - dot product for L2-normalized vectors)
        embeddings_gpu = cp.asarray(embeddings)
        dbscan = cuDBSCAN(
            eps=args.eps,
            min_samples=args.min_samples,
            metric='cosine',  # Uses inner product for normalized vectors
            output_type='numpy',
        )
        cluster_labels = dbscan.fit_predict(embeddings_gpu)
        # Ensure it's a numpy array and cleanup GPU memory
        if isinstance(cluster_labels, cp.ndarray):
            cluster_labels = cp.asnumpy(cluster_labels)
        del embeddings_gpu, dbscan
        cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory
    else:
        if not CUML_AVAILABLE:
            print("Warning: cuML not available. Using CPU DBSCAN (slow for large datasets).")
        elif not torch.cuda.is_available():
            print("Warning: CUDA not available. Using CPU DBSCAN.")
        from benchmarking._utils import run_dbscan
        cluster_labels = run_dbscan(embeddings, eps=args.eps, min_samples=args.min_samples)

    # Build masks and outputs
    is_cath_mask = np.zeros(total_written, dtype=bool)
    is_cath_mask[:cath_written] = True
    all_labels = np.array(cath_labels + thits_labels, dtype=np.int64)[:total_written]
    all_domain_ids = np.array(cath_ids + thits_ids, dtype=object)[:total_written]

    np.save(labels_path, cluster_labels)
    np.save(domain_ids_path, all_domain_ids)

    cluster_df, novel_df = summarize_clusters(
        cluster_labels=cluster_labels,
        is_cath_mask=is_cath_mask,
        min_cluster_size=args.min_cluster_size,
        domain_ids=all_domain_ids,
    )
    cluster_df.to_csv(cluster_summary_path, index=False)
    novel_df.to_csv(novel_clusters_path, index=False)

    noise_fraction = float((cluster_labels == -1).mean()) if len(cluster_labels) else 1.0
    summary = {
        "eps": float(args.eps),
        "min_samples": int(args.min_samples),
        "min_cluster_size": int(args.min_cluster_size),
        "total_embeddings_written": int(total_written),
        "cath_embeddings": int(cath_written),
        "thits_embeddings": int(thits_written),
        "n_clusters": int(len(set(cluster_labels) - {-1})),
        "noise_fraction": noise_fraction,
        "memmap_path": str(memmap_path),
        "labels_path": str(labels_path),
        "domain_ids_path": str(domain_ids_path),
        "cluster_summary_path": str(cluster_summary_path),
        "novel_clusters_path": str(novel_clusters_path),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DBSCAN complete ===")
    print(f"Embeddings processed: {total_written} (CATH {cath_written}, t_hits {thits_written})")
    print(f"Clusters (excl. noise): {summary['n_clusters']}")
    print(f"Noise fraction: {noise_fraction:.3f}")
    print(f"Novel clusters (no CATH, size >= {args.min_cluster_size}): {len(novel_df)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

