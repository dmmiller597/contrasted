"""Create a FAISS vector database from trained model embeddings."""

import hydra
from omegaconf import DictConfig
import torch
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from contrasted.utils import load_h5_keys_from_fasta, extract_domain_id
from contrasted.model import CathSupConModel
from contrasted.faiss_utils import build_faiss_index

logger = logging.getLogger(__name__)


@torch.no_grad()
def embed_sequences(
    model: CathSupConModel,
    h5_file: h5py.File,
    h5_keys: list[str],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[str]]:
    """Project embeddings through trained model.
    
    Args:
        model: Trained CathSupConModel
        h5_file: Open HDF5 file with embeddings
        h5_keys: List of HDF5 keys to process
        device: Device for inference
        batch_size: Batch size for processing
        
    Returns:
        embeddings: (N, D) normalized projected embeddings (float32)
        domain_ids: List of domain IDs
    """
    model.eval()
    all_embeddings = []
    domain_ids = []
    
    for i in tqdm(range(0, len(h5_keys), batch_size), desc="Projecting embeddings"):
        batch_keys = h5_keys[i : i + batch_size]
        batch_embs = []
        
        for h5_key in batch_keys:
            try:
                embedding = torch.from_numpy(h5_file[h5_key][:]).float()
                batch_embs.append(embedding)
                domain_ids.append(extract_domain_id(h5_key))
            except KeyError:
                logger.warning(f"Missing embedding for key: {h5_key}")
                continue
        
        if not batch_embs:
            continue
        
        batch_tensor = torch.stack(batch_embs).to(device)
        projected = model(batch_tensor)
        all_embeddings.append(projected.cpu().numpy())
    
    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    return embeddings_matrix, domain_ids


@hydra.main(version_base=None, config_path="configs", config_name="make_db")
def main(cfg: DictConfig):
    """Create FAISS index from trained model embeddings."""
    
    # Validate inputs
    input_path = Path(cfg.input)
    model_path = Path(cfg.model_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Setup device
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = CathSupConModel.load_from_checkpoint(str(model_path), strict=False)
    model.eval()
    model.to(device)
    
    # Load FASTA and get HDF5 keys
    logger.info(f"Loading sequences from: {input_path}")
    h5_keys = load_h5_keys_from_fasta(input_path)
    
    # Filter by IDs if specified
    if cfg.ids:
        ids_set = set(cfg.ids)
        h5_keys = [k for k in h5_keys if extract_domain_id(k) in ids_set]
        logger.info(f"Filtered to {len(h5_keys)} sequences matching provided IDs")
    else:
        logger.info(f"Processing {len(h5_keys)} sequences")
    
    # Load embeddings and project through model
    embedding_file = Path(cfg.get("embedding_file", "data/cath-domain-seqs-S100.h5"))
    logger.info(f"Loading embeddings from: {embedding_file}")
    
    with h5py.File(embedding_file, "r") as h5f:
        embeddings_matrix, domain_ids = embed_sequences(
            model, h5f, h5_keys, device, batch_size=256
        )
    
    # Validate embeddings
    if embeddings_matrix.size == 0:
        raise ValueError("No valid embeddings generated")
    
    logger.info(
        f"Generated {embeddings_matrix.shape[0]} embeddings "
        f"of dimension {embeddings_matrix.shape[1]}"
    )
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    index = build_faiss_index(embeddings_matrix)
    
    # Save index
    index_path = Path(cfg.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    import faiss
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to: {index_path}")
    
    # Save domain IDs
    ids_path = index_path.with_suffix(".npy")
    np.save(ids_path, np.array(domain_ids))
    logger.info(f"Saved domain IDs to: {ids_path}")
    
    logger.info("✓ Database creation complete!")


if __name__ == "__main__":
    main()

