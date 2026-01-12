#!/usr/bin/env python
"""Benchmark sequence-based classification methods.

Compares sequence-based methods for CATH superfamily classification:
1. ProstT5 baseline (raw embeddings k-NN without contrastive training)
2. Contrasted model (projected embeddings k-NN with contrastive training)
3. ProtT5 baseline (raw ProtT5 embeddings k-NN, using v4.3 embeddings with domain mapping)
4. ProtTucker (ProtT5 embeddings projected through ProtTucker model)
5. MMseqs2 sequence search
6. HH-suite3 (profile-based search)

Outputs metrics in same format as structure benchmarks for unified plotting.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from contrasted.faiss_utils import build_faiss_index, search_faiss_index
from contrasted.model import CathSupConModel
from contrasted.utils import load_h5_keys_from_fasta

from benchmarking._utils import (
    extract_cath_levels, 
    compute_classification_metrics,
    load_labels_dict,
    get_device,
    l2_normalize,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Common Utilities
# =============================================================================
# Note: load_labels_dict, get_device, l2_normalize are imported from benchmarking._utils


def parse_fasta(fasta_path: Path, id_extractor: Callable[[str], str] = None) -> Dict[str, str]:
    """Parse FASTA file into {id: sequence} dict."""
    if id_extractor is None:
        id_extractor = extract_domain_id_from_header
    
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = id_extractor(line.strip())
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def extract_domain_id_from_header(header: str) -> str:
    """Extract domain ID from FASTA header (removes position range).
    
    Handles formats:
    - cath|4_4_0|1vhuA00/9-199 -> 1vhuA00
    - cath|4_4_0|12e8H01_1-113 -> 12e8H01
    - Plain: 1vhuA00 -> 1vhuA00
    """
    header = header.lstrip('>').split()[0]
    
    if '|' in header:
        domain_part = header.split('|')[-1]
        if '/' in domain_part:
            return domain_part.split('/')[0]
        elif '_' in domain_part:
            # Check if last part is position range (e.g., "1-113")
            parts = domain_part.split('_')
            if len(parts) >= 2 and '-' in parts[-1] and parts[-1].replace('-', '').isdigit():
                return '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
        return domain_part
    
    return header


def extract_domain_id_with_position(h5_key: str) -> str:
    """Extract domain ID WITH position range from H5 key (for v4.3 format).
    
    Example: 'cath|4_3_0|107lA00_1-162' -> '107lA00_1-162'
    """
    parts = h5_key.split('|')
    return parts[2] if len(parts) == 3 else h5_key


def strip_position_from_id(domain_id: str) -> str:
    """Strip position range from domain ID for label lookup.
    
    Example: '107lA00_1-162' -> '107lA00'
    """
    if '_' in domain_id:
        parts = domain_id.split('_')
        if len(parts) >= 2 and '-' in parts[-1] and parts[-1].replace('-', '').isdigit():
            return parts[0]
    return domain_id


def build_h5_key_mapping(h5_path: Path) -> Dict[str, str]:
    """Build mapping from domain_id (with position) to full H5 key.
    
    Used for v4.3 embeddings where keys include position ranges.
    """
    mapping = {}
    with h5py.File(h5_path, 'r') as f:
        for h5_key in f.keys():
            domain_id = extract_domain_id_with_position(h5_key)
            mapping[domain_id] = h5_key
    return mapping


# =============================================================================
# Metrics Computation
# =============================================================================

def load_seq_similarity_buckets(test_dir: Path) -> Dict[str, set]:
    """Load domain IDs from sequence similarity bucket FASTA files."""
    buckets = {}
    for fasta_file in sorted(test_dir.glob("s*.fasta")):
        bucket_name = fasta_file.stem
        domain_ids = set()
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    domain_ids.add(extract_domain_id_from_header(line.strip()))
        buckets[bucket_name] = domain_ids
        logger.debug(f"Loaded {len(domain_ids)} domains from {bucket_name}")
    return buckets


def compute_stratified_metrics(
    df_results: pl.DataFrame,
    method: str,
    buckets: Dict[str, set],
    id_to_sf: Dict[str, str],
) -> Optional[pl.DataFrame]:
    """Compute classification metrics stratified by sequence similarity buckets."""
    if df_results is None or len(df_results) == 0:
        return None
    
    results = []
    sorted_buckets = sorted(buckets.keys(), key=lambda x: int(x[1:]))
    
    # Build reverse mapping: domain_id (no position) -> query_ids
    domain_id_to_query_ids = {}
    for query_id in df_results['query_id'].unique().to_list():
        domain_id = strip_position_from_id(query_id)
        domain_id_to_query_ids.setdefault(domain_id, []).append(query_id)
    
    for bucket_name in sorted_buckets:
        bucket_ids = buckets[bucket_name]
        
        # Find matching query_ids
        matching_query_ids = []
        for bucket_domain_id in bucket_ids:
            if bucket_domain_id in domain_id_to_query_ids:
                matching_query_ids.extend(domain_id_to_query_ids[bucket_domain_id])
        
        if not matching_query_ids:
            continue
        
        df_bucket = df_results.filter(pl.col('query_id').is_in(matching_query_ids))
        if len(df_bucket) == 0:
            continue
        
        n_total_in_bucket = sum(
            1 for qid in bucket_ids 
            if id_to_sf.get(qid, '') or id_to_sf.get(qid.upper(), '')
        )
        
        metrics = compute_classification_metrics(df_bucket, method, n_total_in_bucket)
        if metrics is not None:
            metrics = metrics.with_columns([
                pl.lit(bucket_name).alias('bucket'),
                pl.lit(int(bucket_name[1:])).alias('seq_sim_threshold'),
            ])
            results.append(metrics)
    
    return pl.concat(results) if results else None


# =============================================================================
# Models
# =============================================================================

class ProtTuckerModel(nn.Module):
    """ProtTucker projection head: 1024 -> 256 -> 128 with L2 normalization."""
    
    def __init__(self):
        super().__init__()
        self.tucker = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tucker(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: torch.device):
        """Load ProtTucker model from checkpoint."""
        model = cls()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        model.to(device)
        return model


def load_contrasted_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load Contrasted model with PyTorch 2.6+ compatibility."""
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    
    try:
        model = CathSupConModel.load_from_checkpoint(str(checkpoint_path), strict=False)
    finally:
        torch.load = original_load
    
    model.eval()
    model.to(device)
    return model


# =============================================================================
# Generic Embedding k-NN Benchmark
# =============================================================================

@dataclass
class EmbeddingBenchmarkConfig:
    """Configuration for embedding-based k-NN benchmark."""
    name: str
    embedding_h5: Path
    labels_file: Path
    query_fasta: Path
    lookup_fasta: Path
    k: int = 1
    distance_cutoff: float = 2.0
    batch_size: int = 2048
    model: Optional[nn.Module] = None
    device: torch.device = None
    # For v4.3 format (ProtT5/ProtTucker) where IDs include position ranges
    use_position_mapping: bool = False
    h5_key_mapping: Optional[Dict[str, str]] = None


def load_embeddings_batched(
    h5_keys: List[str],
    h5_file: h5py.File,
    model: Optional[nn.Module],
    device: torch.device,
    batch_size: int,
    id_extractor: Callable[[str], str],
    h5_key_mapping: Optional[Dict[str, str]] = None,
    desc: str = "Loading",
) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings in batches with optional model projection."""
    embeddings = []
    domain_ids = []
    
    for i in tqdm(range(0, len(h5_keys), batch_size), desc=desc):
        batch_keys = h5_keys[i:i+batch_size]
        batch_embs = []
        batch_ids = []
        
        for key in batch_keys:
            # Handle v4.3 format with key mapping
            domain_id = id_extractor(key)
            actual_key = key
            if h5_key_mapping is not None:
                if domain_id not in h5_key_mapping:
                    continue
                actual_key = h5_key_mapping[domain_id]
            
            try:
                emb = h5_file[actual_key][:]
                if emb.dtype == np.float16:
                    emb = emb.astype(np.float32)
                batch_embs.append(emb)
                batch_ids.append(domain_id)
            except KeyError:
                continue
        
        if not batch_embs:
            continue
        
        batch_array = np.vstack(batch_embs).astype(np.float32)
        
        # Project through model if provided
        if model is not None:
            with torch.no_grad():
                tensor = torch.from_numpy(batch_array).float().to(device)
                batch_array = model(tensor).cpu().numpy()
        
        embeddings.append(batch_array)
        domain_ids.extend(batch_ids)
    
    if not embeddings:
        return np.array([]), []
    
    return np.vstack(embeddings).astype(np.float32), domain_ids


def run_embedding_knn_benchmark(config: EmbeddingBenchmarkConfig) -> Tuple[Optional[pl.DataFrame], float, int]:
    """Generic embedding-based k-NN classification benchmark.
    
    Works for: ProstT5, ProtT5, Contrasted, ProtTucker.
    """
    logger.info("=" * 70)
    logger.info(f"Running {config.name}")
    logger.info("=" * 70)
    
    start_time = time.time()
    device = config.device or get_device()
    logger.info(f"Using device: {device}")
    
    id_to_sf = load_labels_dict(config.labels_file)
    
    # Determine ID extractor based on format
    if config.use_position_mapping:
        id_extractor = extract_domain_id_with_position
    else:
        id_extractor = extract_domain_id_from_header
    
    # Load lookup embeddings
    logger.info("Loading lookup embeddings...")
    lookup_h5_keys = load_h5_keys_from_fasta(config.lookup_fasta)
    
    with h5py.File(config.embedding_h5, 'r') as h5f:
        lookup_embs, lookup_ids = load_embeddings_batched(
            lookup_h5_keys, h5f, config.model, device, config.batch_size,
            id_extractor, config.h5_key_mapping, "Loading lookup"
        )
    
    if len(lookup_embs) == 0:
        logger.warning(f"No lookup embeddings found. Skipping {config.name}.")
        return None, 0.0, 0
    
    lookup_embs = l2_normalize(lookup_embs)
    logger.info(f"Built lookup with {len(lookup_embs)} embeddings")
    
    # Build FAISS index
    index = build_faiss_index(lookup_embs)
    
    # Load query embeddings
    logger.info("Querying test embeddings...")
    query_h5_keys = load_h5_keys_from_fasta(config.query_fasta)
    n_total_queries = len(query_h5_keys)
    
    with h5py.File(config.embedding_h5, 'r') as h5f:
        query_embs, query_ids = load_embeddings_batched(
            query_h5_keys, h5f, config.model, device, config.batch_size,
            id_extractor, config.h5_key_mapping, "Querying"
        )
    
    if len(query_embs) == 0:
        logger.warning(f"No query embeddings found. Skipping {config.name}.")
        return None, 0.0, n_total_queries
    
    query_embs = l2_normalize(query_embs)
    
    # Search
    similarities, indices = search_faiss_index(index, query_embs, config.k)
    distances = 1.0 - similarities
    
    # Build results
    results = []
    for i, (query_id, dist_row, idx_row) in enumerate(zip(query_ids, distances, indices)):
        top_dist = dist_row[0]
        if top_dist > config.distance_cutoff:
            continue
        
        top_id = lookup_ids[idx_row[0]]
        
        # For v4.3 format, strip position for label lookup
        query_id_for_label = strip_position_from_id(query_id) if config.use_position_mapping else query_id
        top_id_for_label = strip_position_from_id(top_id) if config.use_position_mapping else top_id
        
        query_sf = id_to_sf.get(query_id_for_label, '') or id_to_sf.get(query_id_for_label.upper(), '')
        pred_sf = id_to_sf.get(top_id_for_label, '') or id_to_sf.get(top_id_for_label.upper(), '')
        
        if query_sf:
            results.append({
                'query_id': query_id,
                'target_id': top_id,
                'true_sf': query_sf,
                'predicted_sf': pred_sf if pred_sf else 'unknown',
                'distance': float(top_dist),
            })
    
    runtime = time.time() - start_time
    logger.info(f"{config.name} completed in {runtime:.2f}s")
    logger.info(f"Found {len(results)} predictions from {n_total_queries} queries")
    
    if not results:
        return None, runtime, n_total_queries
    
    return pl.DataFrame(results), runtime, n_total_queries


# =============================================================================
# MMseqs2 Benchmark
# =============================================================================

def run_mmseqs2_benchmark(
    query_fasta: Path,
    lookup_fasta: Path,
    labels_file: Path,
    sensitivity: float = 7.5,
    min_seq_id: float = 0.0,
    k: int = 1,
) -> Tuple[Optional[pl.DataFrame], float, int]:
    """Run MMseqs2 sequence search benchmark."""
    logger.info("=" * 70)
    logger.info("Running MMseqs2 Sequence Search")
    logger.info("=" * 70)
    
    if shutil.which('mmseqs') is None:
        logger.warning("MMseqs2 not found in PATH. Skipping.")
        return None, 0.0, 0
    
    start_time = time.time()
    id_to_sf = load_labels_dict(labels_file)
    
    # Get all query IDs
    all_query_ids = list(parse_fasta(query_fasta).keys())
    n_total_queries = len(all_query_ids)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        query_db = tmp_path / "queryDB"
        lookup_db = tmp_path / "lookupDB"
        result_db = tmp_path / "resultDB"
        
        logger.info("Creating MMseqs2 databases...")
        
        # Create databases
        for db, fasta in [(query_db, query_fasta), (lookup_db, lookup_fasta)]:
            result = subprocess.run(
                ["mmseqs", "createdb", str(fasta), str(db)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(f"MMseqs2 createdb failed: {result.stderr}")
                return None, 0.0, n_total_queries
        
        # Run search
        logger.info(f"Running MMseqs2 search (sensitivity={sensitivity})...")
        result = subprocess.run([
            "mmseqs", "search",
            str(query_db), str(lookup_db), str(result_db), str(tmp_path / "tmp"),
            "-s", str(sensitivity), "--max-seqs", str(k), "-a",
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"MMseqs2 search failed: {result.stderr}")
            return None, 0.0, n_total_queries
        
        # Convert to TSV
        alignments_file = tmp_path / "alignments.tsv"
        result = subprocess.run([
            "mmseqs", "convertalis",
            str(query_db), str(lookup_db), str(result_db), str(alignments_file),
            "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits",
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"MMseqs2 convertalis failed: {result.stderr}")
            return None, 0.0, n_total_queries
        
        # Parse results
        hits_by_query = {}
        if alignments_file.exists() and alignments_file.stat().st_size > 0:
            df_mmseqs = pl.read_csv(alignments_file, separator='\t', has_header=False)
            cols = ['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen',
                   'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits']
            df_mmseqs = df_mmseqs.rename({f'column_{i+1}': c for i, c in enumerate(cols)})
            
            for query_name in df_mmseqs['query'].unique().to_list():
                group = df_mmseqs.filter(pl.col('query') == query_name)
                top_hit = group.sort('bits', descending=True).head(1).row(0, named=True)
                
                if min_seq_id > 0 and top_hit['fident'] < min_seq_id:
                    continue
                
                query_id = extract_domain_id_from_header(str(query_name))
                target_id = extract_domain_id_from_header(str(top_hit['target']))
                target_sf = id_to_sf.get(target_id, '') or id_to_sf.get(target_id.upper(), '')
                
                hits_by_query[query_id] = {
                    'target_id': target_id,
                    'predicted_sf': target_sf if target_sf else 'unknown',
                    'evalue': top_hit['evalue'],
                    'bits': top_hit['bits'],
                    'fident': top_hit['fident'],
                }
        
        # Build results for all queries
        results = []
        n_hits = n_no_hits = 0
        
        for query_id in all_query_ids:
            query_sf = id_to_sf.get(query_id, '') or id_to_sf.get(query_id.upper(), '')
            if not query_sf:
                continue
            
            if query_id in hits_by_query:
                hit = hits_by_query[query_id]
                results.append({
                    'query_id': query_id,
                    'target_id': hit['target_id'],
                    'true_sf': query_sf,
                    'predicted_sf': hit['predicted_sf'],
                    'evalue': hit['evalue'],
                    'bits': hit['bits'],
                    'fident': hit['fident'],
                })
                n_hits += 1
            else:
                results.append({
                    'query_id': query_id,
                    'target_id': 'no_hit',
                    'true_sf': query_sf,
                    'predicted_sf': 'no_hit',
                    'evalue': float('inf'),
                    'bits': 0.0,
                    'fident': 0.0,
                })
                n_no_hits += 1
    
    runtime = time.time() - start_time
    logger.info(f"MMseqs2 completed in {runtime:.2f}s")
    logger.info(f"Found {n_hits} hits, {n_no_hits} no-hits from {n_total_queries} queries")
    
    return pl.DataFrame(results) if results else None, runtime, n_total_queries


# =============================================================================
# HH-suite3 Benchmark
# =============================================================================

def _run_single_hhsearch(
    query_id: str,
    query_seq: str,
    hhsuite_db: Path,
    tmpdir: Path,
    evalue_cutoff: float,
    hhsearch_threads: int = 1,
) -> Optional[Dict]:
    """Run hhsearch for a single query sequence."""
    query_file = tmpdir / f"query_{query_id}.a3m"
    result_file = tmpdir / f"result_{query_id}.hhr"
    
    try:
        with open(query_file, 'w') as f:
            f.write(f">{query_id}\n{query_seq}\n")
    except Exception:
        return None
    
    cmd = [
        "hhsearch",
        "-i", str(query_file),
        "-d", str(hhsuite_db),
        "-o", str(result_file),
        "-cpu", str(hhsearch_threads),
        "-e", str(evalue_cutoff),
        "-v", "0",
        "-maxfilt", "10000",
        "-realign_max", "1000",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
        if result.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, Exception):
        return None
    
    if result_file.exists():
        hit = _parse_hhr_top_hit(result_file)
        if hit is not None:
            hit['query_id'] = query_id
            return hit
    
    return None


def _parse_hhr_top_hit(hhr_file: Path) -> Optional[Dict]:
    """Parse HHsuite .hhr result file and return top hit info."""
    try:
        with open(hhr_file, 'r') as f:
            lines = f.readlines()
        
        in_hits = False
        for line in lines:
            if line.strip().startswith('No Hit'):
                in_hits = True
                continue
            
            if in_hits and line.strip():
                # Skip header line and empty lines
                if not line.lstrip()[0].isdigit():
                    continue
                
                # Parse fixed-width format:
                # "  1 1spvA00                         98.7 1.3E-11 3.1E-16  101.5   0.0  174   10-193     2-177 (184)"
                # Columns: hit_num (0-3), target_id (4-35), prob (36-40), evalue (41-49), pvalue (50-58), score (60-67)
                if len(line) < 67:
                    # Fallback to space-separated parsing
                    parts = line.split()
                    if len(parts) >= 6:
                        target_id = extract_domain_id_from_header(parts[1].replace('.hhm', ''))
                        try:
                            prob = float(parts[2]) if len(parts) > 2 else 0.0
                            evalue = float(parts[3]) if len(parts) > 3 else 1.0
                            score = float(parts[5]) if len(parts) > 5 else 0.0
                        except (ValueError, IndexError):
                            prob, evalue, score = 0.0, 1.0, 0.0
                        return {
                            'target_id': target_id,
                            'probability': prob,
                            'evalue': evalue,
                            'score': score,
                        }
                    break
                
                # Use space-separated parsing (more reliable than fixed-width)
                parts = line.split()
                if len(parts) < 6:
                    break
                
                try:
                    target_id = parts[1].replace('.hhm', '')
                    target_id = extract_domain_id_from_header(target_id)
                    
                    # Parse probability (column 2)
                    prob = float(parts[2]) if len(parts) > 2 else 0.0
                    # Parse evalue (column 3)
                    evalue = float(parts[3]) if len(parts) > 3 else 1.0
                    # Parse score (column 5, skipping pvalue at column 4)
                    score = float(parts[5]) if len(parts) > 5 else 0.0
                    
                    return {
                        'target_id': target_id,
                        'probability': prob,
                        'evalue': evalue,
                        'score': score,
                    }
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing HHR line: {line.strip()[:80]}... Error: {e}")
                    # Return at least the target ID with defaults
                    if len(parts) >= 2:
                        target_id = extract_domain_id_from_header(parts[1].replace('.hhm', ''))
                        return {
                            'target_id': target_id,
                            'probability': 0.0,
                            'evalue': 1.0,
                            'score': 0.0,
                        }
                break
        
        return None
    except Exception as e:
        logger.debug(f"Error parsing HHR file {hhr_file}: {e}")
        return None


def run_hhsuite_benchmark(
    query_fasta: Path,
    lookup_fasta: Path,
    labels_file: Path,
    hhsuite_db: Path = None,
    n_threads: int = 1,
    evalue_cutoff: float = 1e-3,
    max_workers: int = None,
) -> Tuple[Optional[pl.DataFrame], float, int]:
    """Run HH-suite3 sequence-to-HMM search benchmark."""
    logger.info("=" * 70)
    logger.info("Running HH-suite3 Sequence-to-HMM Search")
    logger.info("=" * 70)
    
    if shutil.which('hhsearch') is None:
        logger.warning("HH-suite3 (hhsearch) not found in PATH. Skipping.")
        logger.info("Install with: conda install -c conda-forge -c bioconda hhsuite")
        return None, 0.0, 0
    
    if hhsuite_db is None:
        logger.warning("--hhsuite_db not provided. Skipping HH-suite benchmark.")
        return None, 0.0, 0
    
    db_hhm = Path(f"{hhsuite_db}_hhm.ffdata")
    if not db_hhm.exists():
        logger.warning(f"HH-suite HMM database not found: {db_hhm}")
        return None, 0.0, 0
    
    start_time = time.time()
    id_to_sf = load_labels_dict(labels_file)
    
    # Parse query FASTA
    query_seqs = parse_fasta(query_fasta)
    all_query_ids = list(query_seqs.keys())
    n_total_queries = len(all_query_ids)
    
    logger.info(f"Loaded {n_total_queries} query sequences")
    logger.info(f"Using HH-suite database: {hhsuite_db}")
    
    # Parallelization
    hhsearch_threads = 1 if max_workers is None or max_workers > 1 else n_threads
    if max_workers is None:
        max_workers = max(1, cpu_count() // 2)
    
    logger.info(f"Running {n_total_queries} queries with {max_workers} parallel workers")
    
    hits_by_query = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        tasks = [
            (qid, query_seqs[qid], hhsuite_db, tmp_path, evalue_cutoff, hhsearch_threads)
            for qid in all_query_ids if qid in query_seqs
        ]
        
        logger.info(f"Submitting {len(tasks)} hhsearch tasks...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_hhsearch, *t): t[0] for t in tasks}
            
            pbar = tqdm(total=len(futures), desc="hhsearch", unit="query", miniters=1, mininterval=1.0)
            completed = 0
            
            for future in as_completed(futures):
                query_id = futures[future]
                completed += 1
                try:
                    hit = future.result(timeout=200)
                    if hit is not None:
                        hits_by_query[query_id] = hit
                except Exception as e:
                    logger.debug(f"Error processing query {query_id}: {e}")
                finally:
                    pbar.update(1)
                    if completed % 100 == 0 or completed == len(futures):
                        pbar.set_postfix({'hits': len(hits_by_query)})
            
            pbar.close()
    
    # Build results
    results = []
    n_hits = n_no_hits = 0
    
    for query_id in all_query_ids:
        query_sf = id_to_sf.get(query_id, '') or id_to_sf.get(query_id.upper(), '')
        if not query_sf:
            continue
        
        if query_id in hits_by_query:
            hit = hits_by_query[query_id]
            target_id = hit['target_id']
            target_sf = id_to_sf.get(target_id, '') or id_to_sf.get(target_id.upper(), '')
            
            results.append({
                'query_id': query_id,
                'target_id': target_id,
                'true_sf': query_sf,
                'predicted_sf': target_sf if target_sf else 'unknown',
                'evalue': hit['evalue'],
                'probability': hit['probability'],
                'score': hit['score'],
            })
            n_hits += 1
        else:
            results.append({
                'query_id': query_id,
                'target_id': 'no_hit',
                'true_sf': query_sf,
                'predicted_sf': 'no_hit',
                'evalue': float('inf'),
                'probability': 0.0,
                'score': 0.0,
            })
            n_no_hits += 1
    
    runtime = time.time() - start_time
    logger.info(f"HH-suite completed in {runtime:.2f}s")
    logger.info(f"Found {n_hits} hits, {n_no_hits} no-hits from {n_total_queries} queries")
    
    return pl.DataFrame(results) if results else None, runtime, n_total_queries


# =============================================================================
# Method Configuration & Registry
# =============================================================================

@dataclass
class MethodConfig:
    """Configuration for a benchmark method."""
    name: str
    output_file: str
    skip_flag: str
    runner: Callable = None
    # For embedding methods
    embedding_key: str = None  # 'prostt5' or 'prott5'
    model_loader: Callable = None
    model_checkpoint_arg: str = None
    distance_cutoff_arg: str = None
    use_position_mapping: bool = False


def create_method_registry() -> Dict[str, MethodConfig]:
    """Create registry of benchmark methods."""
    return {
        'prostt5': MethodConfig(
            name='ProstT5_Baseline',
            output_file='prostt5_baseline_results.tsv',
            skip_flag='skip_prostt5',
            embedding_key='prostt5',
            distance_cutoff_arg='prostt5_distance_cutoff',
        ),
        'prott5': MethodConfig(
            name='ProtT5_Baseline',
            output_file='prott5_baseline_results.tsv',
            skip_flag='skip_prott5_baseline',
            embedding_key='prott5',
            distance_cutoff_arg='prott5_distance_cutoff',
            use_position_mapping=True,
        ),
        'prottucker': MethodConfig(
            name='ProtTucker',
            output_file='prottucker_results.tsv',
            skip_flag='skip_prottucker',
            embedding_key='prott5',
            model_loader=ProtTuckerModel.from_checkpoint,
            model_checkpoint_arg='tucker_checkpoint',
            distance_cutoff_arg='prottucker_distance_cutoff',
            use_position_mapping=True,
        ),
        'contrasted': MethodConfig(
            name='Contrasted',
            output_file='contrasted_results.tsv',
            skip_flag='skip_contrasted',
            embedding_key='prostt5',
            model_loader=load_contrasted_model,
            model_checkpoint_arg='model_path',
            distance_cutoff_arg='contrasted_distance_cutoff',
        ),
        'mmseqs2': MethodConfig(
            name='MMseqs2',
            output_file='mmseqs2_results.tsv',
            skip_flag='skip_mmseqs2',
            runner=run_mmseqs2_benchmark,
        ),
        'hhsuite': MethodConfig(
            name='HHsuite3',
            output_file='hhsuite_results.tsv',
            skip_flag='skip_hhsuite',
            runner=run_hhsuite_benchmark,
        ),
    }


def run_embedding_method(method_config: MethodConfig, args) -> Tuple[Optional[pl.DataFrame], float, int]:
    """Run an embedding-based benchmark method."""
    # Determine embedding file
    if method_config.embedding_key == 'prott5':
        embedding_h5 = args.prott5_embedding_h5
        if not embedding_h5.exists():
            logger.warning(f"ProtT5 embedding file not found: {embedding_h5}. Skipping {method_config.name}.")
            return None, 0.0, 0
    else:
        embedding_h5 = args.embedding_h5
    
    device = get_device()
    
    # Load model if needed
    model = None
    if method_config.model_loader is not None:
        checkpoint = getattr(args, method_config.model_checkpoint_arg)
        if checkpoint is None or not checkpoint.exists():
            logger.warning(f"Checkpoint not found: {checkpoint}. Skipping {method_config.name}.")
            return None, 0.0, 0
        model = method_config.model_loader(checkpoint, device)
    
    # Build H5 key mapping for v4.3 format
    h5_key_mapping = None
    if method_config.use_position_mapping:
        logger.info("Building domain ID to H5 key mapping...")
        h5_key_mapping = build_h5_key_mapping(embedding_h5)
        logger.info(f"Mapped {len(h5_key_mapping)} domain IDs to H5 keys")
    
    # Get distance cutoff
    distance_cutoff = getattr(args, method_config.distance_cutoff_arg, 2.0) if method_config.distance_cutoff_arg else 2.0
    
    config = EmbeddingBenchmarkConfig(
        name=method_config.name,
        embedding_h5=embedding_h5,
        labels_file=args.labels_file,
        query_fasta=args.query_fasta,
        lookup_fasta=args.lookup_fasta,
        k=args.k,
        distance_cutoff=distance_cutoff,
        batch_size=args.batch_size,
        model=model,
        device=device,
        use_position_mapping=method_config.use_position_mapping,
        h5_key_mapping=h5_key_mapping,
    )
    
    return run_embedding_knn_benchmark(config)


def process_benchmark_results(
    df: pl.DataFrame,
    method_name: str,
    runtime: float,
    n_queries: int,
    output_dir: Path,
    output_file: str,
    dataset: str,
    buckets: Dict[str, set],
    id_to_sf: Dict[str, str],
) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """Process and save benchmark results."""
    if df is None or len(df) == 0:
        return None, None
    
    # Save raw results
    df.write_csv(output_dir / output_file, separator='\t')
    
    # Compute metrics
    metrics = compute_classification_metrics(df, method_name, n_queries, runtime)
    if metrics is not None:
        metrics = metrics.with_columns([pl.lit(dataset).alias('dataset')])
    
    # Compute stratified metrics
    stratified = None
    if buckets:
        stratified = compute_stratified_metrics(df, method_name, buckets, id_to_sf)
        if stratified is not None:
            stratified = stratified.with_columns([pl.lit(dataset).alias('dataset')])
    
    return metrics, stratified


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark sequence-based classification methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument("--query_fasta", type=Path,
                        default=Path("data/clustered-datasets/test.fasta"),
                        help="Query FASTA file (test set)")
    parser.add_argument("--lookup_fasta", type=Path,
                        default=Path("data/clustered-datasets/train_excl_holdouts66.fasta"),
                        help="Lookup FASTA file (training set)")
    parser.add_argument("--embedding_h5", type=Path,
                        default=Path("data/cath-domain-seqs-S100.h5"),
                        help="HDF5 file with ProstT5 embeddings (v4.4)")
    parser.add_argument("--prott5_embedding_h5", type=Path,
                        default=Path("data/tucker/cath_v430_dom_seqs_S100_161121.h5"),
                        help="HDF5 file with ProtT5 embeddings (v4.3, for ProtT5 baseline and ProtTucker)")
    parser.add_argument("--tucker_checkpoint", type=Path,
                        default=Path("data/tucker/ProtTucker_ProtT5.pt"),
                        help="Path to ProtTucker model checkpoint")
    parser.add_argument("--labels_file", type=Path,
                        default=Path("data/cath-domain-sf-list.txt"),
                        help="CATH superfamily labels file")
    parser.add_argument("--model_path", type=Path,
                        default=Path("checkpoints/holdouts66.ckpt"),
                        help="Path to trained Contrasted model checkpoint")
    parser.add_argument("--output_dir", type=Path,
                        default=Path(f"outputs/{Path(__file__).parent.name}"),
                        help="Output directory")
    parser.add_argument("--test_buckets_dir", type=Path,
                        default=Path("data/clustered-datasets/test"),
                        help="Directory containing sequence similarity bucket FASTAs")
    
    # Method selection
    parser.add_argument("--skip_prostt5", action="store_true", help="Skip ProstT5 baseline")
    parser.add_argument("--skip_prott5_baseline", action="store_true", help="Skip ProtT5 baseline")
    parser.add_argument("--skip_prottucker", action="store_true", help="Skip ProtTucker")
    parser.add_argument("--skip_mmseqs2", action="store_true", help="Skip MMseqs2")
    parser.add_argument("--skip_contrasted", action="store_true", help="Skip Contrasted")
    parser.add_argument("--skip_hhsuite", action="store_true", help="Skip HH-suite3")
    parser.add_argument("--hhsuite_db", type=Path, default=None,
                        help="Pre-built HH-suite database path (without extension)")
    
    # Parameters
    parser.add_argument("--k", type=int, default=1, help="Number of neighbors")
    parser.add_argument("--prostt5_distance_cutoff", type=float, default=2.0)
    parser.add_argument("--prott5_distance_cutoff", type=float, default=2.0)
    parser.add_argument("--prottucker_distance_cutoff", type=float, default=1.0)
    parser.add_argument("--contrasted_distance_cutoff", type=float, default=1.0)
    parser.add_argument("--mmseqs2_sensitivity", type=float, default=7.5)
    parser.add_argument("--mmseqs2_min_seq_id", type=float, default=0.0)
    parser.add_argument("--hhsuite_evalue", type=float, default=1e-3)
    parser.add_argument("--hhsuite_threads", type=int, default=1)
    parser.add_argument("--hhsuite_max_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--dataset", type=str, default="test")
    
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load common data
    id_to_sf = load_labels_dict(args.labels_file)
    buckets = load_seq_similarity_buckets(args.test_buckets_dir) if args.test_buckets_dir.exists() else {}
    if buckets:
        logger.info(f"Loaded {len(buckets)} sequence similarity buckets: {sorted(buckets.keys(), key=lambda x: int(x[1:]))}")
    
    all_results = []
    all_stratified = []
    
    # Method registry
    methods = create_method_registry()
    
    # Run each method
    for method_key, method_config in methods.items():
        # Check skip flag
        if getattr(args, method_config.skip_flag, False):
            continue
        
        # Run benchmark
        if method_config.embedding_key is not None:
            # Embedding-based method
            df, runtime, n_queries = run_embedding_method(method_config, args)
        elif method_key == 'mmseqs2':
            df, runtime, n_queries = run_mmseqs2_benchmark(
                args.query_fasta, args.lookup_fasta, args.labels_file,
                sensitivity=args.mmseqs2_sensitivity,
                min_seq_id=args.mmseqs2_min_seq_id,
                k=args.k,
            )
        elif method_key == 'hhsuite':
            df, runtime, n_queries = run_hhsuite_benchmark(
                args.query_fasta, args.lookup_fasta, args.labels_file,
                hhsuite_db=args.hhsuite_db,
                n_threads=args.hhsuite_threads,
                evalue_cutoff=args.hhsuite_evalue,
                max_workers=args.hhsuite_max_workers,
            )
        else:
            continue
        
        # Process results
        metrics, stratified = process_benchmark_results(
            df, method_config.name, runtime, n_queries,
            args.output_dir, method_config.output_file, args.dataset,
            buckets, id_to_sf,
        )
        
        if metrics is not None:
            all_results.append(metrics)
        if stratified is not None:
            all_stratified.append(stratified)
    
    # Save summary
    if all_results:
        df_summary = pl.concat(all_results)
        summary_file = args.output_dir / "summary.tsv"
        df_summary.write_csv(summary_file, separator='\t')
        
        logger.info("\n" + "=" * 70)
        logger.info("Sequence Classification Benchmark Summary")
        logger.info("=" * 70)
        logger.info(f"\n{df_summary}")
        logger.info(f"\nResults saved to: {summary_file}")
    else:
        logger.warning("No results to save")
    
    # Save stratified results
    if all_stratified:
        df_stratified = pl.concat(all_stratified)
        stratified_file = args.output_dir / "summary_by_seqsim.tsv"
        df_stratified.write_csv(stratified_file, separator='\t')
        
        logger.info("\n" + "=" * 70)
        logger.info("Stratified Results by Sequence Similarity")
        logger.info("=" * 70)
        
        for method in df_stratified['method'].unique().to_list():
            method_df = df_stratified.filter(pl.col('method') == method)
            logger.info(f"\n{method}:")
            logger.info(f"{method_df.select(['bucket', 'accuracy', 'n_queries', 'n_total_queries'])}")
        
        logger.info(f"\nStratified results saved to: {stratified_file}")


if __name__ == "__main__":
    main()
