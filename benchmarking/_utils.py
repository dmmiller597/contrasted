#!/usr/bin/env python
"""Shared utilities for analysis scripts.

Common functions for loading embeddings, projecting datasets, and computing
geometric properties of embedding spaces.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Sequence, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from contrasted.model import CathSupConModel

import numpy as np
import torch
import h5py
from tqdm import tqdm
from scipy.stats import percentileofscore
from dataclasses import dataclass
from collections import Counter
import polars as pl
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, 
    homogeneity_score, completeness_score, v_measure_score,
    f1_score
)

from contrasted.data import CathEmbeddingDataset
from contrasted.utils import load_labels, extract_domain_id, load_h5_keys_from_fasta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatasetSlice:
    """Lightweight container for projected datasets."""
    embeddings: np.ndarray
    labels: np.ndarray
    domain_ids: List[str]
    superfamily_names: Dict[int, str]

    def subset(self, mask: np.ndarray) -> "DatasetSlice":
        return DatasetSlice(
            embeddings=self.embeddings[mask],
            labels=self.labels[mask],
            domain_ids=[d for d, keep in zip(self.domain_ids, mask) if keep],
            superfamily_names=self.superfamily_names,
        )


def get_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def extract_cath_levels(cath_code: str, level: str) -> str:
    """Extract CATH hierarchy level from CATH code.
    
    Args:
        cath_code: CATH code in format "C.A.T.H" (e.g., "1.10.8.10")
        level: One of 'C', 'A', 'T', 'H'
            - 'C': Class (first component)
            - 'A': Architecture (first two components)
            - 'T': Topology (first three components)
            - 'H': Homologous superfamily (full code)
        
    Returns:
        Extracted level code as string, or empty string if invalid
        
    Example:
        >>> extract_cath_levels("1.10.8.10", "C")
        '1'
        >>> extract_cath_levels("1.10.8.10", "A")
        '1.10'
        >>> extract_cath_levels("1.10.8.10", "T")
        '1.10.8'
        >>> extract_cath_levels("1.10.8.10", "H")
        '1.10.8.10'
    """
    if not cath_code or '.' not in str(cath_code):
        return ''
    try:
        parts = str(cath_code).split('.')
        if level == 'C':
            return parts[0] if len(parts) > 0 else ''
        elif level == 'A':
            return '.'.join(parts[:2]) if len(parts) > 1 else ''
        elif level == 'T':
            return '.'.join(parts[:3]) if len(parts) > 2 else ''
        elif level == 'H':
            return cath_code  # Full code is homologous superfamily
        else:
            return ''
    except (AttributeError, IndexError):
        return ''


def load_labels_dict(labels_file: Path, include_uppercase: bool = True) -> Dict[str, str]:
    """Load domain ID -> superfamily mapping from labels file.
    
    Args:
        labels_file: Path to labels file with format: domain_id superfamily
        include_uppercase: If True, also add uppercase keys for case-insensitive lookup
        
    Returns:
        Dictionary mapping domain_id -> superfamily CATH code
    """
    id_to_sf = {}
    with open(labels_file, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    id_to_sf[parts[0]] = parts[1]
                    if include_uppercase:
                        id_to_sf[parts[0].upper()] = parts[1]
    return id_to_sf


def compute_classification_metrics(
    df_results: pl.DataFrame,
    method: str,
    n_total_queries: int,
    runtime_seconds: Optional[float] = None,
) -> Optional[pl.DataFrame]:
    """Compute classification metrics from results DataFrame.
    
    Computes accuracy at CATH hierarchy levels (C, A, T, H), balanced accuracy,
    F1 macro score, and coverage.
    
    Args:
        df_results: Polars DataFrame with columns:
            - 'true_sf': True superfamily CATH code
            - 'predicted_sf': Predicted superfamily CATH code
        method: Method name (e.g., 'Foldseek', 'Contrasted', 'MMseqs2')
        n_total_queries: Total number of queries processed (before threshold filtering)
        runtime_seconds: Optional runtime in seconds to include in results
        
    Returns:
        DataFrame with single row containing metrics, or None if no results
    """
    if len(df_results) == 0:
        return None
    
    n_predictions = len(df_results)
    
    # Convert to string and handle nulls
    df_results = df_results.with_columns([
        pl.col('true_sf').cast(pl.Utf8).fill_null(''),
        pl.col('predicted_sf').cast(pl.Utf8).fill_null(''),
    ])
    
    # Compute matches at each CATH level
    matches = {}
    for level in ['C', 'A', 'T', 'H']:
        true_levels = df_results['true_sf'].map_elements(
            lambda x: extract_cath_levels(x, level), return_dtype=pl.Utf8
        )
        pred_levels = df_results['predicted_sf'].map_elements(
            lambda x: extract_cath_levels(x, level), return_dtype=pl.Utf8
        )
        matches[level] = (true_levels == pred_levels) & (true_levels != '') & (pred_levels != '')
    
    # Compute accuracies
    accuracies = {
        f'accuracy_{level}': float(matches[level].sum() / n_predictions) if n_predictions > 0 else 0.0
        for level in ['C', 'A', 'T', 'H']
    }
    
    # Balanced accuracy (macro-averaged by true class)
    true_sfs = df_results['true_sf'].value_counts()
    balanced_acc_H = 0.0
    if len(true_sfs) > 0:
        for row in true_sfs.iter_rows(named=True):
            sf = row['true_sf']
            count = row['count']
            sf_mask = df_results['true_sf'] == sf
            sf_matches = (matches['H'] & sf_mask).sum()
            balanced_acc_H += (sf_matches / count) if count > 0 else 0.0
        balanced_acc_H /= len(true_sfs)
    
    # F1 macro (computed at H-level, not full CATH codes)
    true_H = df_results['true_sf'].map_elements(
        lambda x: extract_cath_levels(x, 'H'), return_dtype=pl.Utf8
    ).to_numpy()
    pred_H = df_results['predicted_sf'].map_elements(
        lambda x: extract_cath_levels(x, 'H'), return_dtype=pl.Utf8
    ).to_numpy()
    valid_mask = (true_H != '') & (pred_H != '') & (pred_H != 'unknown')
    f1_macro = f1_score(
        true_H[valid_mask], pred_H[valid_mask], 
        average='macro', zero_division=0.0
    ) if valid_mask.sum() > 0 else 0.0
    
    coverage = n_predictions / n_total_queries if n_total_queries > 0 else 0.0
    
    result = {
        'method': method,
        'accuracy': accuracies['accuracy_H'],
        'balanced_accuracy': balanced_acc_H,
        'f1_macro': f1_macro,
        'coverage': coverage,
        **accuracies,
        'n_queries': n_predictions,
        'n_total_queries': n_total_queries,
    }
    
    if runtime_seconds is not None:
        result['runtime_seconds'] = runtime_seconds
    
    return pl.DataFrame([result])


def filter_by_superfamily_size(dataset: DatasetSlice, min_size: int) -> Tuple[DatasetSlice, float]:
    """Return subset containing only superfamilies with ≥min_size sequences.

    Returns:
        filtered_dataset, coverage_ratio (kept / total)
    """
    if len(dataset.labels) == 0:
        return dataset, 0.0
    counts = Counter(map(int, dataset.labels))
    keep_ids = {sf for sf, count in counts.items() if count >= min_size}
    mask = np.array([int(label) in keep_ids for label in dataset.labels], dtype=bool)
    filtered = dataset.subset(mask)
    coverage = float(mask.sum()) / float(len(mask)) if len(mask) else 0.0
    return filtered, coverage


def load_raw_embeddings(
    embedding_h5: Path,
    dataset: CathEmbeddingDataset,
    idx_to_sf: Dict[int, str],
) -> DatasetSlice:
    """Load raw ProstT5 embeddings and normalize them (baseline)."""
    embeddings_list, labels_list = [], []
    domain_ids = []
    superfamily_names = {}
    
    with h5py.File(embedding_h5, 'r') as h5f:
        for h5_key, label in dataset.samples:
            try:
                embedding = h5f[h5_key][:]
                embeddings_list.append(embedding)
                labels_list.append(label)
                domain_id = extract_domain_id(h5_key)
                domain_ids.append(domain_id)
                
                # Store superfamily name mapping
                if int(label) not in superfamily_names:
                    superfamily_names[int(label)] = idx_to_sf.get(int(label), f"SF_{label}")
            except KeyError:
                logger.warning(f"Missing embedding for key: {h5_key}")
                continue
    
    if not embeddings_list:
        raise ValueError("No valid embeddings loaded")
    
    embeddings = np.vstack(embeddings_list).astype(np.float32)
    
    # L2 normalize for cosine similarity (same as ProstT5 baseline)
    embeddings = l2_normalize(embeddings)
    
    labels = np.array(labels_list, dtype=np.int64)
    
    return DatasetSlice(embeddings, labels, domain_ids, superfamily_names)


@torch.no_grad()
def project_dataset(
    model: CathSupConModel,
    dataset: CathEmbeddingDataset,
    batch_size: int,
    num_workers: int = 0,
    idx_to_sf: Dict[int, str] = None,
    show_progress: bool = True,
) -> DatasetSlice:
    """Project dataset through model and return embeddings, labels, and domain IDs."""
    model.eval()
    device = next(model.parameters()).device
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    embeddings_list, labels_list = [], []
    domain_ids = [extract_domain_id(k) for k, _ in dataset.samples]
    
    iterator = tqdm(loader, desc="Projecting embeddings", unit="batch") if show_progress else loader
    for batch_emb, batch_labels in iterator:
        proj = model(batch_emb.to(device))
        embeddings_list.append(proj.cpu())
        labels_list.append(batch_labels.cpu())
    
    embeddings = torch.cat(embeddings_list, dim=0).numpy().astype(np.float32)
    labels = torch.cat(labels_list, dim=0).numpy().astype(np.int64)
    
    if np.any(~np.isfinite(embeddings)):
        logger.warning("Found non-finite values in embeddings!")
    
    # Superfamily names
    if idx_to_sf is None:
        superfamily_names = {int(label): f"SF_{label}" for label in np.unique(labels)}
    else:
        superfamily_names = {int(label): idx_to_sf.get(int(label), f"SF_{label}") 
                            for label in np.unique(labels)}
    
    return DatasetSlice(embeddings, labels, domain_ids, superfamily_names)


def build_dataset(
    embedding_h5: Path,
    label_map: Dict[str, int],
    fasta_path: Path,
    cache_embeddings: bool = True,
) -> CathEmbeddingDataset:
    """Build dataset from FASTA file."""
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    h5_keys = load_h5_keys_from_fasta(fasta_path)
    if not h5_keys:
        raise ValueError(f"No valid sequences found in {fasta_path}")
    
    return CathEmbeddingDataset(
        h5_path=embedding_h5,
        h5_keys=h5_keys,
        labels=label_map,
        cache_embeddings=cache_embeddings,
    )


class ModelAndDataLoader:
    """Helper class to load models and datasets consistently across scripts."""
    
    def __init__(
        self,
        checkpoint: Path = None,
        embedding_h5: Path = None,
        labels_file: Path = None,
        batch_size: int = 256,
        num_workers: int = 0,
        cache_embeddings: bool = True,
    ):
        """Initialize loader with paths and configuration."""
        self.embedding_h5 = Path(embedding_h5).resolve() if embedding_h5 else None
        self.labels_file = Path(labels_file).resolve() if labels_file else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_embeddings = cache_embeddings
        
        # Load labels if provided
        if self.labels_file:
            logger.info("Loading labels...")
            self.id_to_idx, self.idx_to_sf = load_labels(self.labels_file)
        else:
            self.id_to_idx = None
            self.idx_to_sf = None
        
        # Load model if checkpoint provided
        if checkpoint:
            # Lazy import to avoid triggering huggingface_hub import issues during module load
            from contrasted.model import CathSupConModel
            import torch
            
            self.checkpoint = Path(checkpoint).resolve()
            logger.info("Loading model...")
            self.device = get_device()
            
            # Temporarily monkey-patch torch.load to disable weights_only for checkpoint loading
            # This is safe since checkpoints are from trusted sources
            # PyTorch 2.6+ defaults to weights_only=True which blocks omegaconf objects
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            
            try:
                self.model = CathSupConModel.load_from_checkpoint(
                    str(self.checkpoint),
                    map_location=self.device,
                    strict=False,
                )
            finally:
                # Restore original torch.load
                torch.load = original_load
            self.model.eval()
            logger.info(f"Model loaded on {self.device}")
        else:
            self.checkpoint = None
            self.model = None
            self.device = None
    
    def load_dataset(self, fasta_path: Path) -> CathEmbeddingDataset:
        """Load dataset from FASTA file."""
        fasta_path = fasta_path.resolve()
        return build_dataset(
            self.embedding_h5,
            self.id_to_idx,
            fasta_path,
            cache_embeddings=self.cache_embeddings,
        )
    
    def project_dataset(self, dataset: CathEmbeddingDataset, show_progress: bool = True) -> DatasetSlice:
        """Project dataset through model."""
        return project_dataset(
            self.model,
            dataset,
            self.batch_size,
            self.num_workers,
            idx_to_sf=self.idx_to_sf,
            show_progress=show_progress,
        )
    
    def load_baseline(self, dataset: CathEmbeddingDataset) -> DatasetSlice:
        """Load raw ProstT5 embeddings (baseline)."""
        return load_raw_embeddings(self.embedding_h5, dataset, self.idx_to_sf)


# --------------------------------------------------------------------------- #
# Geometric Utilities
# --------------------------------------------------------------------------- #


def compute_cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distances (1 - cosine similarity)."""
    similarities = embeddings @ embeddings.T
    distances = 1.0 - similarities
    distances = np.clip(distances, 0.0, 2.0)  # Ensure valid range
    return distances


def compute_superfamily_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Compute centroid for each superfamily."""
    centroids = {}
    for sf_idx in np.unique(labels):
        mask = labels == sf_idx
        if mask.sum() > 0:
            sf_embeddings = embeddings[mask]
            centroid = sf_embeddings.mean(axis=0)
            # Normalize centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[int(sf_idx)] = centroid
    return centroids


def nearest_centroid_distances(
    embeddings: np.ndarray,
    train_centroids: Dict[int, np.ndarray],
) -> np.ndarray:
    """Compute distance from each embedding to the nearest training centroid."""
    if len(embeddings) == 0 or not train_centroids:
        return np.full(len(embeddings), np.nan, dtype=np.float32)
    centroid_matrix = np.stack(list(train_centroids.values()), axis=0)
    sims = embeddings @ centroid_matrix.T
    dists = 1.0 - sims
    dists = np.clip(dists, 0.0, 2.0)
    min_dist = dists.min(axis=1)
    return min_dist.astype(np.float32)


def compute_intra_sf_radii(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Compute distances from each point to its superfamily centroid."""
    radii = {}
    for sf_idx in np.unique(labels):
        mask = labels == sf_idx
        if mask.sum() > 0 and sf_idx in centroids:
            sf_embeddings = embeddings[mask]
            centroid = centroids[sf_idx]
            # Cosine distance from centroid
            distances = 1.0 - (sf_embeddings @ centroid)
            distances = np.clip(distances, 0.0, 2.0)
            radii[int(sf_idx)] = distances
    return radii


# --------------------------------------------------------------------------- #
# Clustering Utilities
# --------------------------------------------------------------------------- #


def run_dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    return model.fit_predict(embeddings)


def compute_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    if len(true_labels) == 0:
        return float("nan")
    mask = cluster_labels != -1
    if not np.any(mask):
        return 0.0
    labels = true_labels[mask]
    clusters = cluster_labels[mask]
    total = len(labels)
    majority = 0
    for cid in np.unique(clusters):
        idx = clusters == cid
        if idx.sum() == 0:
            continue
        counts = Counter(labels[idx])
        majority += counts.most_common(1)[0][1]
    return float(majority) / float(total)


def clustering_report(true_labels: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
    if len(true_labels) == 0:
        return {
            "purity": 0.0, "ari": 0.0, "nmi": 0.0, "homogeneity": 0.0,
            "completeness": 0.0, "v_measure": 0.0, "n_clusters": 0,
            "noise_fraction": 1.0, "coverage": 0.0,
        }
    mask = cluster_labels != -1
    coverage = float(mask.sum()) / float(len(cluster_labels)) if len(cluster_labels) else 0.0
    purity = compute_purity(true_labels, cluster_labels)
    ari = float(adjusted_rand_score(true_labels[mask], cluster_labels[mask])) if np.any(mask) else 0.0
    nmi = float(normalized_mutual_info_score(true_labels[mask], cluster_labels[mask])) if np.any(mask) else 0.0
    
    homogeneity = float(homogeneity_score(true_labels[mask], cluster_labels[mask])) if np.any(mask) else 0.0
    completeness = float(completeness_score(true_labels[mask], cluster_labels[mask])) if np.any(mask) else 0.0
    v_measure = float(v_measure_score(true_labels[mask], cluster_labels[mask])) if np.any(mask) else 0.0
    
    n_clusters = int(len(set(cluster_labels[mask])) if np.any(mask) else 0)
    noise_fraction = float((cluster_labels == -1).mean()) if len(cluster_labels) else 1.0
    
    return {
        "purity": purity,
        "ari": ari,
        "nmi": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "coverage": coverage,
    }


def run_parameter_search(
    dataset: DatasetSlice,
    eps_range: Tuple[float, float],
    n_trials: int,
    min_samples: int,
    max_samples: Optional[int] = 10000,
    early_stop_purity: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Run parameter search with optional subsampling and early stopping.
    
    Note: Early stopping is disabled by default to ensure full exploration
    of the parameter space. If enabled, it stops when purity >= threshold
    and coverage >= 0.8, but this may not align with the selection metric.
    """
    if max_samples is None or len(dataset.embeddings) <= max_samples:
        search_embeddings = dataset.embeddings
        search_labels = dataset.labels
    else:
        indices = np.random.choice(len(dataset.embeddings), max_samples, replace=False)
        search_embeddings = dataset.embeddings[indices]
        search_labels = dataset.labels[indices]
    
    eps_values = np.linspace(eps_range[0], eps_range[1], n_trials)
    records = []
    logger.info(f"Running parameter search with {len(search_embeddings)} samples ({n_trials} eps values)...")
    
    for eps in tqdm(eps_values, desc="Parameter search"):
        clusters = run_dbscan(search_embeddings, eps, min_samples)
        report = clustering_report(search_labels, clusters)
        records.append({
            "eps": float(eps),
            **report
        })
        # Early stop only if explicitly enabled and conditions are met
        # Disabled by default to ensure full parameter space exploration
        if early_stop_purity is not None and report["purity"] >= early_stop_purity and report["coverage"] >= 0.8:
            logger.info(f"Early stopping at eps={eps:.4f} (purity={report['purity']:.3f})")
            logger.warning("Early stopping may prevent finding optimal epsilon for the selection metric!")
            break
            
    return records


def select_best_eps(
    records: Sequence[Dict[str, float]], 
    target_n_clusters: int = None, 
    method: str = "v_measure_coverage"
) -> float:
    """Select best eps using simple, interpretable methods."""
    if not records:
        raise ValueError("No parameter search records to select from.")
    
    if method == "geometric_mean":
        scored_records = [((r["ari"] * r["purity"] * r["coverage"]) ** (1/3), r) for r in records]
    elif method == "v_measure":
        scored_records = [(r.get("v_measure", 0.0), r) for r in records]
    elif method == "v_measure_coverage":
        scored_records = [((r.get("v_measure", 0.0) * r.get("coverage", 0.0)) ** 0.5, r) for r in records]
    elif method == "min_max":
        scored_records = [(min(r["ari"], r["purity"], r["coverage"]), r) for r in records]
    elif method == "pareto":
        # Simple implementation: score by geometric mean
        scored_records = [((r["ari"] * r["purity"] * r["coverage"]) ** (1/3), r) for r in records]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Sort by score (descending)
    sorted_records = sorted(scored_records, key=lambda x: (-x[0], x[1]["eps"]))
    best_record = sorted_records[0][1]
    best_score = sorted_records[0][0]
    
    if target_n_clusters is not None:
        # Tie-breaker
        candidates = [r for score, r in scored_records if abs(score - best_score) < 1e-6]
        if len(candidates) > 1:
            best_record = min(candidates, key=lambda r: abs(r["n_clusters"] - target_n_clusters))
            
    logger.info(f"Selected eps={best_record['eps']:.4f} (score={best_score:.4f}, method={method})")
    return float(best_record["eps"])


# --------------------------------------------------------------------------- #
# Large-scale Stats
# --------------------------------------------------------------------------- #


def compute_distance_statistics_chunked(
    embeddings: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 5000,
    sample_size: int = 1000000,
) -> Dict[str, float]:
    """Compute distance statistics incrementally without storing all distances."""
    n_samples = len(embeddings)
    
    intra_count = 0
    intra_sum = 0.0
    intra_sum_sq = 0.0
    
    inter_count = 0
    inter_sum = 0.0
    inter_sum_sq = 0.0
    
    all_distances_sample = []
    sample_idx = 0
    
    total_pairs = n_samples * (n_samples - 1) // 2
    
    with tqdm(total=total_pairs, desc="Computing pairwise distances", unit="pairs", unit_scale=True) as pbar:
        for i in range(0, n_samples, chunk_size):
            i_end = min(i + chunk_size, n_samples)
            chunk_emb = embeddings[i:i_end]
            remaining_emb = embeddings[i:]
            
            similarities = chunk_emb @ remaining_emb.T
            distances = 1.0 - similarities
            distances = np.clip(distances, 0.0, 2.0)
            
            chunk_labels_i = labels[i:i_end]
            remaining_labels = labels[i:]
            
            for chunk_idx, global_i in enumerate(range(i, i_end)):
                start_col = global_i - i + 1
                end_col = n_samples - i
                
                if start_col < end_col:
                    row_distances = distances[chunk_idx, start_col:end_col]
                    row_labels_j = remaining_labels[start_col:end_col]
                    
                    same_sf = chunk_labels_i[chunk_idx] == row_labels_j
                    intra_dists = row_distances[same_sf]
                    inter_dists = row_distances[~same_sf]
                    
                    if len(intra_dists) > 0:
                        intra_count += len(intra_dists)
                        intra_sum += intra_dists.sum()
                        intra_sum_sq += (intra_dists ** 2).sum()
                    
                    if len(inter_dists) > 0:
                        inter_count += len(inter_dists)
                        inter_sum += inter_dists.sum()
                        inter_sum_sq += (inter_dists ** 2).sum()
                    
                    # Reservoir sampling
                    for dist in row_distances:
                        if len(all_distances_sample) < sample_size:
                            all_distances_sample.append(float(dist))
                        elif np.random.randint(0, sample_idx + 1) < sample_size:
                            all_distances_sample[np.random.randint(0, sample_size)] = float(dist)
                        sample_idx += 1
                    
                    pbar.update(len(row_distances))
    
    intra_mean = intra_sum / intra_count if intra_count > 0 else np.nan
    intra_std = np.sqrt((intra_sum_sq / intra_count - intra_mean ** 2)) if intra_count > 0 else np.nan
    inter_mean = inter_sum / inter_count if inter_count > 0 else np.nan
    inter_std = np.sqrt((inter_sum_sq / inter_count - inter_mean ** 2)) if inter_count > 0 else np.nan
    
    all_distances_sample = np.array(all_distances_sample)
    if len(all_distances_sample) > 0:
        all_median = float(np.median(all_distances_sample))
        all_p1 = float(np.percentile(all_distances_sample, 1))
        all_p99 = float(np.percentile(all_distances_sample, 99))
    else:
        all_median = all_p1 = all_p99 = np.nan
        
    return {
        'intra_sf_mean': float(intra_mean), 'intra_sf_std': float(intra_std),
        'inter_sf_mean': float(inter_mean), 'inter_sf_std': float(inter_std),
        'median': all_median, 'p1': all_p1, 'p99': all_p99
    }


def extract_distances(
    embeddings: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pairwise distances for same and different superfamily pairs."""
    n_samples = len(embeddings)
    
    if n_samples > 50000:
        return extract_distances_chunked(embeddings, labels, chunk_size)
        
    distances = compute_cosine_distance(embeddings)
    label_matrix = labels[:, None] == labels[None, :]
    triu_idx = np.triu_indices_from(distances, k=1)
    all_distances = distances[triu_idx]
    same_sf = label_matrix[triu_idx]
    
    return all_distances[same_sf], all_distances[~same_sf]


def extract_distances_chunked(
    embeddings: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = len(embeddings)
    intra_dists_list = []
    inter_dists_list = []
    
    with tqdm(total=n_samples * (n_samples - 1) // 2, desc="Computing pairwise distances", unit="pairs", unit_scale=True) as pbar:
        for i in range(0, n_samples, chunk_size):
            i_end = min(i + chunk_size, n_samples)
            chunk_emb = embeddings[i:i_end]
            remaining_emb = embeddings[i:]
            
            similarities = chunk_emb @ remaining_emb.T
            distances = 1.0 - similarities
            distances = np.clip(distances, 0.0, 2.0)
            
            chunk_labels_i = labels[i:i_end]
            remaining_labels = labels[i:]
            
            for chunk_idx, global_i in enumerate(range(i, i_end)):
                start_col = global_i - i + 1
                end_col = n_samples - i
                if start_col < end_col:
                    row_distances = distances[chunk_idx, start_col:end_col]
                    row_labels_j = remaining_labels[start_col:end_col]
                    same_sf = chunk_labels_i[chunk_idx] == row_labels_j
                    intra_dists_list.append(row_distances[same_sf])
                    inter_dists_list.append(row_distances[~same_sf])
                    pbar.update(len(row_distances))
    
    intra = np.concatenate(intra_dists_list) if intra_dists_list else np.array([])
    inter = np.concatenate(inter_dists_list) if inter_dists_list else np.array([])
    return intra, inter


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    if len(data) < 2:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_bootstrap)]
    alpha = 1 - confidence
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def compute_unseen_superfamily_analysis(
    embeddings_holdout: np.ndarray,
    labels_holdout: np.ndarray,
    superfamily_names_holdout: Dict[int, str],
    embeddings_train: np.ndarray,
    labels_train: np.ndarray,
    superfamily_names_train: Dict[int, str],
    min_size: int = 2,
) -> pl.DataFrame:
    """Analyze unseen superfamilies compared to training superfamilies."""
    logger.info("Computing unseen superfamily analysis...")
    
    train_centroids = compute_superfamily_centroids(embeddings_train, labels_train)
    train_radii = compute_intra_sf_radii(embeddings_train, labels_train, train_centroids)
    
    trained_sf_stats = {}
    for sf_idx, centroid in train_centroids.items():
        radii = train_radii.get(sf_idx, np.array([]))
        if len(radii) >= min_size:
            trained_sf_stats[int(sf_idx)] = {
                'centroid': centroid,
                'mean_radius': float(radii.mean()),
                'size': len(radii),
            }
            
    if not trained_sf_stats:
        return pl.DataFrame()
        
    trained_mean_radii = [s['mean_radius'] for s in trained_sf_stats.values()]
    trained_median_radius = float(np.median(trained_mean_radii))
    
    holdout_centroids = compute_superfamily_centroids(embeddings_holdout, labels_holdout)
    holdout_radii = compute_intra_sf_radii(embeddings_holdout, labels_holdout, holdout_centroids)
    
    rows = []
    for sf_idx in np.unique(labels_holdout):
        sf_idx = int(sf_idx)
        radii = holdout_radii.get(sf_idx, np.array([]))
        if len(radii) < min_size:
            continue
            
        mean_radius = float(radii.mean())
        centroid = holdout_centroids[sf_idx]
        ci_low, ci_high = bootstrap_ci(radii)
        
        # Nearest trained SF
        min_dist = np.inf
        nearest_name = None
        nearest_idx = None
        
        for train_idx, train_stat in trained_sf_stats.items():
            dist = 1.0 - (centroid @ train_stat['centroid'])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = train_idx
                nearest_name = superfamily_names_train.get(train_idx, f"SF_{train_idx}")
        
        rows.append({
            'superfamily': superfamily_names_holdout.get(sf_idx, f"SF_{sf_idx}"),
            'superfamily_idx': sf_idx,
            'n_samples': len(radii),
            'mean_radius': mean_radius,
            'median_radius': float(np.median(radii)),
            'std_radius': float(radii.std()),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'nearest_trained_sf': nearest_name,
            'nearest_trained_sf_idx': nearest_idx,
            'distance_to_nearest_trained': float(min_dist),
            'radius_ratio': mean_radius / trained_median_radius if trained_median_radius > 0 else np.nan,
            'radius_percentile_in_trained': float(percentileofscore(trained_mean_radii, mean_radius)),
            'trained_median_radius': trained_median_radius,
        })
        
    return pl.DataFrame(rows)
