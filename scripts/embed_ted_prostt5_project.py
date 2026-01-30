#!/usr/bin/env python3
"""
Stream a large FASTA, embed with ProstT5, and project to 128 dims via Contrastive.

This script is designed for very large FASTA files (tens of millions of sequences).
It performs two passes:
  1) Count labeled sequences to pre-allocate a single output embedding directory.
  2) Stream, embed, project, and write results sequentially to disk.

Output directory files:
  - embeddings.npy: float16, shape (N, 128)
  - labels.npy: int64, shape (N,)
  - ids.txt: one ID per line
  - metadata.json: dims, count, dtype, source, idx_to_label
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from contrasted.data import parse_fasta_header
from contrasted.model import ContrastiveModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ted_labels(label_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load TED labels from two-column TSV: domain_id<TAB>label_code."""
    id_to_label_idx: dict[str, int] = {}
    label_to_idx: dict[str, int] = {}

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            domain_id, label = parts[0], parts[1]
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
            id_to_label_idx[domain_id] = label_to_idx[label]

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    logger.info(
        "Loaded %s labels across %s classes",
        len(id_to_label_idx),
        len(idx_to_label),
    )
    return id_to_label_idx, idx_to_label


def iter_fasta(fasta_path: Path):
    """Yield (domain_id, sequence) from FASTA in a streaming manner."""
    current_id = None
    seq_chunks: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, "".join(seq_chunks).replace("-", "")
                try:
                    current_id = parse_fasta_header(line)
                except ValueError as exc:
                    logger.warning("Skipping invalid header: %s", exc)
                    current_id = None
                seq_chunks = []
            elif current_id is not None:
                seq_chunks.append(line)

    if current_id is not None:
        yield current_id, "".join(seq_chunks).replace("-", "")


def count_labeled_sequences(
    fasta_path: Path, id_to_label_idx: dict[str, int]
) -> int:
    """Count FASTA entries that have labels."""
    total = 0
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            try:
                domain_id = parse_fasta_header(line)
            except ValueError:
                continue
            if domain_id in id_to_label_idx:
                total += 1
    return total


def load_prostt5(model_dir: str, device: torch.device, half: bool):
    logger.info("Loading ProstT5 from: %s", model_dir)
    model = T5EncoderModel.from_pretrained(model_dir).to(device).eval()
    if half:
        model = model.half()
        logger.info("Using half precision for ProstT5")
    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, tokenizer


def load_projection_head(checkpoint: Path, device: torch.device, half: bool):
    logger.info("Loading ContrastiveModel from: %s", checkpoint)
    model = ContrastiveModel.load_from_checkpoint(
        str(checkpoint), strict=False, weights_only=False
    )
    model = model.eval().to(device)
    if half:
        model = model.half()
        logger.info("Using half precision for projection head")
    return model


def format_sequence(seq: str) -> str:
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
    return "<AA2fold> " + " ".join(list(seq))


def process_batch(
    batch,
    model,
    projector,
    tokenizer,
    device: torch.device,
):
    domain_ids, seqs, seq_lens, label_idxs = zip(*batch, strict=False)

    token_encoding = tokenizer.batch_encode_plus(
        seqs, add_special_tokens=True, padding="longest", return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model(
            token_encoding.input_ids,
            attention_mask=token_encoding.attention_mask,
        )

        pooled = []
        for batch_idx, s_len in enumerate(seq_lens):
            emb = output.last_hidden_state[batch_idx, 1 : s_len + 1]
            pooled.append(emb.mean(dim=0))
        pooled_tensor = torch.stack(pooled, dim=0)

        projected = projector(pooled_tensor)

    return domain_ids, projected, label_idxs


def main():
    parser = argparse.ArgumentParser(
        description="Embed TED FASTA with ProstT5 and project to 128 dims"
    )
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Rostlab/ProstT5")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-residues", type=int, default=4000)
    parser.add_argument("--max-batch", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--flush-every", type=int, default=10000)
    args = parser.parse_args()

    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label file not found: {args.labels}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = get_device()
    logger.info("Using device: %s", device)

    logger.info("Loading labels from: %s", args.labels)
    id_to_label_idx, idx_to_label = load_ted_labels(args.labels)

    logger.info("Counting labeled sequences in FASTA (first pass)...")
    start_count = time.time()
    total = count_labeled_sequences(args.fasta, id_to_label_idx)
    logger.info(
        "Found %s labeled sequences in %.1fs",
        total,
        time.time() - start_count,
    )
    if total == 0:
        raise ValueError("No labeled sequences found in FASTA.")

    args.output.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output / "embeddings.npy"
    labels_path = args.output / "labels.npy"
    ids_path = args.output / "ids.txt"
    metadata_path = args.output / "metadata.json"

    embeddings_mm = np.lib.format.open_memmap(
        embeddings_path, mode="w+", dtype=np.float16 if args.half else np.float32, shape=(total, 128)
    )
    labels_mm = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int64, shape=(total,)
    )

    model, tokenizer = load_prostt5(args.model, device, args.half)
    projector = load_projection_head(args.checkpoint, device, args.half)

    batch = []
    write_idx = 0
    last_flush = 0
    progress = tqdm(total=total, desc="Embedding", unit="seq")

    try:
        with open(ids_path, "w") as ids_file:
            for domain_id, seq in iter_fasta(args.fasta):
                label_idx = id_to_label_idx.get(domain_id)
                if label_idx is None:
                    continue

                seq_len = len(seq)
                formatted = format_sequence(seq)
                batch.append((domain_id, formatted, seq_len, label_idx))

                n_res_batch = sum(s_len for _, _, s_len, _ in batch)
                should_process = (
                    len(batch) >= args.max_batch
                    or n_res_batch >= args.max_residues
                    or seq_len > args.max_seq_len
                )

                if should_process:
                    domain_ids, projected, label_idxs = process_batch(
                        batch, model, projector, tokenizer, device
                    )
                    batch = []

                    projected_np = projected.detach().cpu().numpy()
                    if args.half:
                        projected_np = projected_np.astype(np.float16, copy=False)
                    else:
                        projected_np = projected_np.astype(np.float32, copy=False)

                    end_idx = write_idx + projected_np.shape[0]
                    if end_idx > total:
                        raise RuntimeError(
                            f"Write index exceeded preallocated size ({end_idx} > {total})"
                        )
                    embeddings_mm[write_idx:end_idx] = projected_np
                    labels_mm[write_idx:end_idx] = np.asarray(
                        label_idxs, dtype=np.int64
                    )
                    ids_file.write("\n".join(domain_ids) + "\n")

                    write_idx = end_idx
                    progress.update(len(domain_ids))

                    if write_idx - last_flush >= args.flush_every:
                        embeddings_mm.flush()
                        labels_mm.flush()
                        last_flush = write_idx

            if batch:
                domain_ids, projected, label_idxs = process_batch(
                    batch, model, projector, tokenizer, device
                )
                projected_np = projected.detach().cpu().numpy()
                if args.half:
                    projected_np = projected_np.astype(np.float16, copy=False)
                else:
                    projected_np = projected_np.astype(np.float32, copy=False)

                end_idx = write_idx + projected_np.shape[0]
                if end_idx > total:
                    raise RuntimeError(
                        f"Write index exceeded preallocated size ({end_idx} > {total})"
                    )
                embeddings_mm[write_idx:end_idx] = projected_np
                labels_mm[write_idx:end_idx] = np.asarray(label_idxs, dtype=np.int64)
                ids_file.write("\n".join(domain_ids) + "\n")
                write_idx = end_idx
                progress.update(len(domain_ids))

    finally:
        progress.close()
        embeddings_mm.flush()
        labels_mm.flush()

    if write_idx != total:
        raise RuntimeError(
            f"Wrote {write_idx} embeddings but expected {total}. "
            "Ensure FASTA headers match the label file."
        )

    metadata = {
        "dims": 128,
        "count": int(total),
        "dtype": str(embeddings_mm.dtype),
        "source": str(args.fasta),
        "idx_to_label": {int(k): v for k, v in idx_to_label.items()},
    }
    metadata_path.write_text(json.dumps(metadata))

    file_size_gb = embeddings_path.stat().st_size / (1024**3)
    logger.info("Saved embeddings to: %s", args.output)
    logger.info("  - %s sequences", total)
    logger.info("  - %s dims", embeddings_mm.shape[1])
    logger.info("  - %.2f GB embeddings.npy", file_size_gb)


if __name__ == "__main__":
    main()
