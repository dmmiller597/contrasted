#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Incremental version that saves embeddings after each batch rather than at the end.
Useful for long-running jobs where you want to preserve progress.

LMDB version for handling large-scale datasets (13M+ sequences).
"""

import argparse
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path
import torch
import lmdb
import numpy as np
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))

# Constants for LMDB embedding storage
EMBEDDING_DIM = 1024
EMBEDDING_DTYPE = np.float16  # half precision


def get_T5_model(model_dir):
    print("Loading T5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False )
    return model, vocab


def fasta_stream(fasta_path: Path, split_char: str, id_field: int, is_3Di: bool):
    """Yield (identifier, sequence) pairs from a FASTA file."""
    current_id = None
    seq_chunks = []
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if line.startswith('>'):
                if current_id is not None:
                    seq = ''.join(seq_chunks)
                    yield current_id, seq
                header = line[1:].strip()
                parts = header.split(split_char)
                if id_field >= len(parts):
                    current_id = header
                else:
                    current_id = parts[id_field]
                current_id = current_id.replace("/", "_").replace(".", "_")
                seq_chunks = []
            else:
                seq_line = ''.join(line.split()).replace("-", "")
                if is_3Di:
                    seq_line = seq_line.lower()
                seq_chunks.append(seq_line)
        if current_id is not None:
            seq = ''.join(seq_chunks)
            yield current_id, seq


def embedding_to_bytes(emb: np.ndarray) -> bytes:
    """Convert numpy embedding to bytes for LMDB storage."""
    return emb.astype(EMBEDDING_DTYPE).tobytes()


def bytes_to_embedding(data: bytes, per_protein: bool = True) -> np.ndarray:
    """Convert bytes back to numpy embedding."""
    arr = np.frombuffer(data, dtype=EMBEDDING_DTYPE)
    if per_protein:
        return arr  # 1D array of shape (EMBEDDING_DIM,)
    else:
        # Per-residue: reshape to (seq_len, EMBEDDING_DIM)
        return arr.reshape(-1, EMBEDDING_DIM)


def get_embeddings(seq_path, emb_path, model_dir, split_char, id_field,
                   per_protein, half_precision, is_3Di,
                   max_residues=12000, max_seq_len=1000, max_batch=256,
                   bucket_size=10000, commit_every=20000, map_size_gb=120,
                   sync_every_secs=600):
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    prefix_token_len = None

    model, vocab = get_T5_model(model_dir)
    if half_precision:
        model = model.half()
        print("Using model in half-precision!")
    prefix_token_len = len(vocab.encode(prefix, add_special_tokens=False))

    # Create/open LMDB environment
    emb_path.mkdir(parents=True, exist_ok=True)
    map_size = int(map_size_gb * 1024 * 1024 * 1024)
    env = lmdb.open(
        str(emb_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    # Write metadata if missing
    meta_key = b"__meta__"
    with env.begin(write=True) as txn:
        if txn.get(meta_key) is None:
            meta = {
                "model": model_dir,
                "embedding_dim": EMBEDDING_DIM,
                "dtype": str(EMBEDDING_DTYPE),
                "per_protein": bool(per_protein),
                "half_precision": bool(half_precision),
                "is_3Di": bool(is_3Di),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            txn.put(meta_key, json.dumps(meta).encode("utf-8"))

    start = time.time()
    last_sync = start
    example_shown = False
    total_embedded = 0
    total_seen = 0
    skipped_existing = 0
    skipped_long = 0
    skipped_empty = 0
    failed_runtime = 0
    total_len = 0

    pending_writes = []

    def flush_writes(force_sync=False):
        nonlocal pending_writes, last_sync
        if not pending_writes:
            return
        with env.begin(write=True) as txn:
            for identifier, emb_numpy in pending_writes:
                key = identifier.encode("utf-8")
                value = embedding_to_bytes(emb_numpy)
                txn.put(key, value, overwrite=False)
        pending_writes = []
        now = time.time()
        if force_sync or (now - last_sync) >= sync_every_secs:
            env.sync()
            last_sync = now

    def embed_batch(batch_items):
        """Return (embedded_count, embedded_len_sum, failed_count)."""
        if not batch_items:
            return 0, 0, 0
        pdb_ids, seqs, seq_lens = zip(*batch_items)
        token_encoding = vocab(
            list(seqs),
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(device)
        try:
            with torch.inference_mode():
                embedding_repr = model(
                    token_encoding.input_ids,
                    attention_mask=token_encoding.attention_mask
                )
        except RuntimeError as exc:
            if len(batch_items) == 1:
                print(f"RuntimeError during embedding for {pdb_ids[0]} (L={seq_lens[0]}): {exc}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return 0, 0, 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            mid = len(batch_items) // 2
            left_count, left_len, left_failed = embed_batch(batch_items[:mid])
            right_count, right_len, right_failed = embed_batch(batch_items[mid:])
            return (
                left_count + right_count,
                left_len + right_len,
                left_failed + right_failed,
            )

        embedded_count = 0
        embedded_len_sum = 0
        for batch_idx, identifier in enumerate(pdb_ids):
            s_len = seq_lens[batch_idx]
            start_idx = prefix_token_len
            end_idx = start_idx + s_len
            emb = embedding_repr.last_hidden_state[batch_idx, start_idx:end_idx]
            if per_protein:
                emb = emb.mean(dim=0)
            emb_numpy = emb.detach().cpu().numpy().squeeze()
            pending_writes.append((identifier, emb_numpy))
            embedded_count += 1
            embedded_len_sum += s_len
        if len(pending_writes) >= commit_every:
            flush_writes()
        return embedded_count, embedded_len_sum, 0

    def process_bucket(bucket_items):
        nonlocal skipped_existing, total_embedded, example_shown, total_len, failed_runtime
        if not bucket_items:
            return
        # Filter already-embedded keys
        with env.begin(write=False) as txn:
            remaining = []
            for identifier, raw_seq, seq_len in bucket_items:
                key = identifier.encode("utf-8")
                if txn.get(key) is not None:
                    skipped_existing += 1
                    continue
                remaining.append((identifier, raw_seq, seq_len))
        if not remaining:
            return
        remaining.sort(key=lambda x: x[2], reverse=True)

        batch = []
        n_res_batch = 0
        for identifier, raw_seq, seq_len in remaining:
            token_seq = prefix + ' ' + ' '.join(list(raw_seq))
            batch.append((identifier, token_seq, seq_len))
            n_res_batch += seq_len
            if len(batch) >= max_batch or n_res_batch >= max_residues:
                embedded_count, embedded_len_sum, failed_count = embed_batch(batch)
                total_embedded += embedded_count
                total_len += embedded_len_sum
                failed_runtime += failed_count
                if not example_shown:
                    print(f"Example: embedded protein {batch[0][0]} with length {batch[0][2]}")
                    example_shown = True
                batch = []
                n_res_batch = 0

        if batch:
            embedded_count, embedded_len_sum, failed_count = embed_batch(batch)
            total_embedded += embedded_count
            total_len += embedded_len_sum
            failed_runtime += failed_count
            if not example_shown:
                print(f"Example: embedded protein {batch[0][0]} with length {batch[0][2]}")
                example_shown = True

    buffer = []
    stream = fasta_stream(seq_path, split_char, id_field, is_3Di)
    for identifier, seq in tqdm(stream, desc="Embedding", unit="seq"):
        total_seen += 1
        if not is_3Di:
            seq = re.sub(r"[UZOB]", "X", seq)
        seq_len = len(seq)
        if seq_len == 0:
            skipped_empty += 1
            continue
        if seq_len > max_seq_len:
            skipped_long += 1
            continue
        buffer.append((identifier, seq, seq_len))
        if len(buffer) >= bucket_size:
            process_bucket(buffer)
            buffer = []

    if buffer:
        process_bucket(buffer)
    flush_writes(force_sync=True)
    env.sync()
    env.close()

    end = time.time()
    avg_len = (total_len / max(total_embedded, 1)) if total_embedded else 0.0
    print('\n############# STATS #############')
    print(f"Total sequences seen: {total_seen}")
    print(f"Total embeddings written: {total_embedded}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped long (> {max_seq_len}): {skipped_long}")
    print(f"Skipped empty: {skipped_empty}")
    print(f"Failed (runtime errors): {failed_runtime}")
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(
        end-start, (end-start)/max(total_embedded, 1), avg_len))
    return True


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'embed_incremental.py creates ProstT5-Encoder embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format. ' +
            'Embeddings are saved incrementally after each batch to LMDB, allowing for ' +
            'resume capability if the script is interrupted. ' +
            'Example: python embed_incremental.py --input /path/to/some_sequences.fasta --output /path/to/embeddings_lmdb --half 1 --is_3Di 0 --per_protein 1' ) )
    
    # Required positional argument
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='A path for saving the created embeddings as LMDB directory.')

    
    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                    default="Rostlab/ProstT5",
                    help='Either a path to a directory holding the checkpoint for a pre-trained model or a huggingface repository link.' )

    # Optional argument
    parser.add_argument('--split_char', type=str, 
                    default='!',
                    help='The character for splitting the FASTA header in order to retrieve ' +
                        "the protein identifier. Should be used in conjunction with --id." +
                        "Default: '!' ")
    
    # Optional argument
    parser.add_argument('--id', type=int, 
                    default=0,
                    help='The index for the uniprot identifier field after splitting the ' +
                        "FASTA header after each symbole in ['|', '#', ':', ' ']." +
                        'Default: 0')
    # Optional argument
    parser.add_argument('--per_protein', type=int,
                    default=1,
                    help="Whether to return per-residue embeddings (0) or the mean-pooled per-protein representation (1: default).")
        
    parser.add_argument('--half', type=int, 
                    default=1,
                    help="Whether to use half_precision or not. Default: 1 (half-precision)")
    
    parser.add_argument('--is_3Di', type=int, 
                    default=0,
                    help="Whether to create embeddings for 3Di or AA file. Default: 0 (generate AA-embeddings)")

    parser.add_argument('--max_seq_len', type=int,
                    default=1280,
                    help="Maximum sequence length to embed. Longer sequences are skipped.")

    parser.add_argument('--max_residues', type=int,
                    default=12000,
                    help="Maximum total residues per batch.")

    parser.add_argument('--max_batch', type=int,
                    default=256,
                    help="Maximum sequences per batch.")

    parser.add_argument('--bucket_size', type=int,
                    default=10000,
                    help="Number of sequences to buffer and length-sort per bucket.")

    parser.add_argument('--commit_every', type=int,
                    default=20000,
                    help="Number of embeddings to accumulate before LMDB write.")

    parser.add_argument('--map_size_gb', type=int,
                    default=120,
                    help="LMDB map size in GB.")

    parser.add_argument('--sync_every_secs', type=int,
                    default=600,
                    help="Seconds between LMDB sync calls.")
    
    return parser

def main():
    parser     = create_arg_parser()
    args       = parser.parse_args()
    
    seq_path   = Path( args.input ) # path to input FASTAS
    emb_path   = Path( args.output) # path where embeddings should be stored (LMDB dir)
    model_dir  = args.model # path/repo_link to checkpoint
    
    split_char = args.split_char
    id_field   = args.id

    per_protein    = False if int(args.per_protein) == 0 else True
    half_precision = False if int(args.half)        == 0 else True
    is_3Di         = False if int(args.is_3Di)      == 0 else True


    get_embeddings(
        seq_path, 
        emb_path, 
        model_dir, 
        split_char, 
        id_field, 
        per_protein=per_protein,
        half_precision=half_precision, 
        is_3Di=is_3Di,
        max_seq_len=args.max_seq_len,
        max_residues=args.max_residues,
        max_batch=args.max_batch,
        bucket_size=args.bucket_size,
        commit_every=args.commit_every,
        map_size_gb=args.map_size_gb,
        sync_every_secs=args.sync_every_secs
        )


if __name__ == '__main__':
    main()
