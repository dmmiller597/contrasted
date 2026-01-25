#!/usr/bin/env python3
"""
Embed CATH sequences using ProstT5 and save to .pt file.

Creates a .pt file containing:
    - embeddings: Tensor[N, 1024] float16 - per-protein embeddings
    - labels: Tensor[N] int64 - superfamily label indices
    - ids: List[str] - domain IDs
    - idx_to_label: Dict[int, str] - label index to superfamily code

Usage:
    python embed_prostt5.py \
        --fasta data/cath-c123-S100.fasta \
        --labels data/cath-domain-sf-list.txt \
        --output data/cath-c123-S100.pt \
        --half
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(model_dir: str, device: torch.device, half_precision: bool):
    """Load ProstT5 model and tokenizer."""
    logger.info(f"Loading ProstT5 from: {model_dir}")
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    if half_precision:
        model = model.half()
        logger.info("Using half precision model")
    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, tokenizer


def parse_fasta_header(header: str) -> str:
    """Extract domain_id from CATH FASTA header.

    Format: >cath|{cath_release}|{domain_id}/{start}-{end}
    Returns: domain_id (e.g., '12e8H01')
    """
    parts = header.strip().lstrip('>').split('|')
    if len(parts) >= 3:
        return parts[2].split('/')[0]
    raise ValueError(f"Invalid FASTA header: {header}")


def read_fasta(fasta_path: Path) -> dict[str, str]:
    """Read FASTA file and return dict of domain_id -> sequence."""
    sequences = {}
    current_id = None

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                try:
                    current_id = parse_fasta_header(line)
                    sequences[current_id] = ''
                except ValueError as e:
                    logger.warning(f"Skipping invalid header: {e}")
                    current_id = None
            elif current_id is not None:
                # Remove gaps and join multi-line sequences
                sequences[current_id] += line.replace('-', '')

    return sequences


def load_labels(label_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load CATH superfamily labels.

    File format: {domain_id}\t{superfamily_code}
    Returns: (domain_id -> sf_idx, sf_idx -> sf_code)
    """
    id_to_sf_idx: dict[str, int] = {}
    sf_to_idx: dict[str, int] = {}

    with open(label_path) as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    domain_id, superfamily = parts[0], parts[1]
                    if superfamily not in sf_to_idx:
                        sf_to_idx[superfamily] = len(sf_to_idx)
                    id_to_sf_idx[domain_id] = sf_to_idx[superfamily]

    idx_to_sf = {v: k for k, v in sf_to_idx.items()}
    logger.info(f"Loaded {len(id_to_sf_idx)} domain labels, {len(idx_to_sf)} superfamilies")
    return id_to_sf_idx, idx_to_sf


def embed_sequences(
    sequences: dict[str, str],
    model,
    tokenizer,
    device: torch.device,
    max_residues: int = 4000,
    max_seq_len: int = 1000,
    max_batch: int = 100,
) -> dict[str, torch.Tensor]:
    """Embed sequences using ProstT5.

    Args:
        sequences: Dict of domain_id -> sequence
        model: ProstT5 encoder model
        tokenizer: T5 tokenizer
        device: Device to run on
        max_residues: Max residues per batch
        max_seq_len: Max sequence length (longer processed individually)
        max_batch: Max sequences per batch

    Returns:
        Dict of domain_id -> embedding tensor (1024 dims)
    """
    embeddings = {}
    prefix = "<AA2fold>"  # AA sequence to structure embedding

    # Sort by length (descending) to trigger OOM early
    sorted_seqs = sorted(sequences.items(), key=lambda x: len(x[1]), reverse=True)

    logger.info(f"Embedding {len(sorted_seqs)} sequences")
    logger.info(f"Average length: {sum(len(s) for _, s in sorted_seqs) / len(sorted_seqs):.1f}")

    batch: list[tuple[str, str, int]] = []
    start_time = time.time()

    for seq_idx, (domain_id, seq) in enumerate(tqdm(sorted_seqs, desc="Embedding"), 1):
        # Replace non-standard amino acids
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)

        # Format for ProstT5: prefix + space-separated residues
        formatted_seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((domain_id, formatted_seq, seq_len))

        # Check if batch should be processed
        n_res_batch = sum(s_len for _, _, s_len in batch)
        should_process = (
            len(batch) >= max_batch or
            n_res_batch >= max_residues or
            seq_idx == len(sorted_seqs) or
            seq_len > max_seq_len
        )

        if should_process:
            domain_ids, seqs, seq_lens = zip(*batch)
            batch = []

            # Tokenize
            token_encoding = tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding="longest",
                return_tensors='pt'
            ).to(device)

            try:
                with torch.no_grad():
                    output = model(
                        token_encoding.input_ids,
                        attention_mask=token_encoding.attention_mask
                    )

                # Extract per-protein embeddings (mean pooling)
                for batch_idx, did in enumerate(domain_ids):
                    s_len = seq_lens[batch_idx]
                    # Account for prefix token in offset (+1)
                    emb = output.last_hidden_state[batch_idx, 1:s_len + 1]
                    # Mean pool to get per-protein embedding
                    emb = emb.mean(dim=0)
                    embeddings[did] = emb.cpu()

            except RuntimeError as e:
                logger.error(f"Error embedding batch starting with {domain_ids[0]}: {e}")
                continue

    elapsed = time.time() - start_time
    logger.info(f"Embedded {len(embeddings)} sequences in {elapsed:.1f}s "
                f"({elapsed / len(embeddings):.3f}s per sequence)")

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Embed CATH sequences with ProstT5 and save to .pt file"
    )
    parser.add_argument(
        '--fasta', '-f',
        type=Path,
        required=True,
        help='Path to CATH FASTA file'
    )
    parser.add_argument(
        '--labels', '-l',
        type=Path,
        required=True,
        help='Path to cath-domain-sf-list.txt label file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output .pt file path'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Rostlab/ProstT5',
        help='HuggingFace model name or local path (default: Rostlab/ProstT5)'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use half precision for model inference'
    )
    parser.add_argument(
        '--max-residues',
        type=int,
        default=4000,
        help='Max residues per batch (default: 4000)'
    )
    parser.add_argument(
        '--max-batch',
        type=int,
        default=100,
        help='Max sequences per batch (default: 100)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label file not found: {args.labels}")

    # Setup
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Reading FASTA: {args.fasta}")
    sequences = read_fasta(args.fasta)
    logger.info(f"Read {len(sequences)} sequences")

    logger.info(f"Loading labels: {args.labels}")
    id_to_label_idx, idx_to_label = load_labels(args.labels)

    # Filter sequences to those with labels
    sequences = {k: v for k, v in sequences.items() if k in id_to_label_idx}
    logger.info(f"Filtered to {len(sequences)} sequences with labels")

    # Load model
    model, tokenizer = load_model(args.model, device, args.half)

    # Embed
    embeddings_dict = embed_sequences(
        sequences,
        model,
        tokenizer,
        device,
        max_residues=args.max_residues,
        max_batch=args.max_batch,
    )

    # Build tensors in consistent order
    domain_ids = list(embeddings_dict.keys())
    embeddings_list = [embeddings_dict[did] for did in domain_ids]
    labels_list = [id_to_label_idx[did] for did in domain_ids]

    embeddings_tensor = torch.stack(embeddings_list).half()  # Save as float16
    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'embeddings': embeddings_tensor,
            'labels': labels_tensor,
            'ids': domain_ids,
            'idx_to_label': idx_to_label,
        },
        args.output
    )

    # Summary
    file_size_mb = args.output.stat().st_size / (1024 * 1024)
    logger.info(f"Saved to: {args.output}")
    logger.info(f"  - {len(domain_ids)} sequences")
    logger.info(f"  - {embeddings_tensor.shape[1]} embedding dims")
    logger.info(f"  - {len(idx_to_label)} classes")
    logger.info(f"  - {file_size_mb:.1f} MB")


if __name__ == '__main__':
    main()
