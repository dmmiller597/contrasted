#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Incremental version that saves embeddings after each batch rather than at the end.
Useful for long-running jobs where you want to preserve progress.

LMDB version for handling large-scale datasets (13M+ sequences).
"""

import argparse
import time
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


def read_fasta( fasta_path, split_char, id_field, is_3Di ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                # repl. all white-space chars and join seqs spanning multiple lines
                if is_3Di:
                    sequences[ uniprot_id ] += ''.join( line.split() ).replace("-","").lower() # drop gaps and cast to upper-case
                else:
                    sequences[ uniprot_id ] += ''.join( line.split() ).replace("-","")
                    
    return sequences


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


def get_existing_keys(env: lmdb.Environment) -> set:
    """Get all existing keys from LMDB database."""
    existing_keys = set()
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            existing_keys.add(key.decode('utf-8'))
    return existing_keys


def get_embeddings( seq_path, emb_path, model_dir, split_char, id_field, 
                       per_protein, half_precision, is_3Di,
                       max_residues=4000, max_seq_len=1000, max_batch=100 ):
    
    seq_dict = dict()

    # Read in fasta
    seq_dict = read_fasta( seq_path, split_char, id_field, is_3Di )
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    
    model, vocab = get_T5_model(model_dir)
    if half_precision:
        model = model.half()
        print("Using model in half-precision!")

    print('########################################')
    print(f"Input is 3Di: {is_3Di}")
    print('Example sequence: {}\n{}'.format( next(iter(
            seq_dict.keys())), next(iter(seq_dict.values()))) )
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long     = sum([ 1 for _, seq in seq_dict.items() if len(seq)>max_seq_len])
    # sort sequences by length to trigger OOM at the beginning
    seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))
    
    # Calculate LMDB map size
    # 13M sequences * 1024 dims * 2 bytes (float16) = ~26GB for embeddings
    # Add overhead for keys and LMDB structure (~50% extra to be safe)
    estimated_size = len(seq_dict) * EMBEDDING_DIM * 2 * 2  # 2x safety factor
    map_size = max(estimated_size, 50 * 1024 * 1024 * 1024)  # At least 50GB
    
    # Create/open LMDB environment
    emb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(emb_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )
    
    # Check which embeddings already exist
    print("Scanning existing embeddings...")
    existing_keys = get_existing_keys(env)
    n_skipped = len(existing_keys)
    
    if n_skipped > 0:
        print(f"Found {n_skipped} existing embeddings. Skipping already processed sequences.")
    
    # Filter out already processed sequences
    seq_dict = [(pdb_id, seq) for pdb_id, seq in seq_dict if pdb_id not in existing_keys]
    print(f"Processing {len(seq_dict)} remaining sequences.")
    
    # Early return if all sequences already processed
    if len(seq_dict) == 0:
        print("All sequences have already been processed. Exiting.")
        env.close()
        return True
    
    start = time.time()
    batch = list()
    total_embedded = n_skipped
    example_shown = False
    batch_embeddings = []  # Accumulate batch results for single transaction
    
    for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict, total=len(seq_dict), desc="Embedding"),1):
        # replace non-standard AAs
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch (each sequence length is already included in batch)
        # this ensures we don't double-count the last sequence and can pack batches up to max_residues
        n_res_batch = sum([ s_len for  _, _, s_len in batch ])
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, 
                                                     add_special_tokens=True, 
                                                     padding="longest", 
                                                     return_tensors='pt' 
                                                     ).to(device)
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids, 
                                           attention_mask=token_encoding.attention_mask
                                           )
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(
                    pdb_id, seq_len)
                    )
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx,1:s_len+1]
                
                if per_protein:
                    emb = emb.mean(dim=0)
                emb_numpy = emb.detach().cpu().numpy().squeeze()
                
                # Accumulate for batch write
                batch_embeddings.append((identifier, emb_numpy))
                total_embedded += 1
                
                if not example_shown:
                    print("Example: embedded protein {} with length {} to emb. of shape: {}".format(
                                identifier, s_len, emb_numpy.shape))
                    example_shown = True
            
            # Write batch to LMDB
            with env.begin(write=True) as txn:
                for identifier, emb_numpy in batch_embeddings:
                    key = identifier.encode('utf-8')
                    value = embedding_to_bytes(emb_numpy)
                    txn.put(key, value)
            batch_embeddings = []
    
    # Sync and close
    env.sync()
    env.close()

    end = time.time()
    
    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(total_embedded))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format( 
            end-start, (end-start)/max(total_embedded - n_skipped, 1), avg_length))
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
                    default=0,
                    help="Whether to return per-residue embeddings (0: default) or the mean-pooled per-protein representation (1).")
        
    parser.add_argument('--half', type=int, 
                    default=0,
                    help="Whether to use half_precision or not. Default: 0 (full-precision)")
    
    parser.add_argument('--is_3Di', type=int, 
                    default=0,
                    help="Whether to create embeddings for 3Di or AA file. Default: 0 (generate AA-embeddings)")
    
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
        is_3Di=is_3Di 
        )


if __name__ == '__main__':
    main()
