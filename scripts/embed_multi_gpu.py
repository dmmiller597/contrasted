#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 2025

@author: mheinzinger
@author: dmiller (multi-gpu support)
"""

import argparse
import time
from pathlib import Path
import torch
import h5py
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
import torch.multiprocessing as mp
import shutil

def get_T5_model(model_dir, device):
    print(f"Loading T5 from: {model_dir} on {device}")
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, vocab

def read_fasta(fasta_path, split_char, id_field, is_3Di):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[uniprot_id] = ''
            else:
                # repl. all white-space chars and join seqs spanning multiple lines
                if is_3Di:
                    sequences[uniprot_id] += ''.join(line.split()).replace("-","").lower() # drop gaps and cast to upper-case
                else:
                    sequences[uniprot_id] += ''.join(line.split()).replace("-","")
                    
    return sequences

def worker_process(rank, gpu_id, seq_list, args, temp_output_path):
    """
    Worker process for embedding generation.
    
    Args:
        rank: Process rank (index in the list of processes)
        gpu_id: Physical GPU ID to use
        seq_list: List of (pdb_id, sequence) tuples to process
        args: Parsed command line arguments
        temp_output_path: Path to write temporary H5 file
    """
    if isinstance(gpu_id, str) and gpu_id == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
    print(f"Process {rank} using device: {device} processing {len(seq_list)} sequences")
    
    # Model loading
    model, vocab = get_T5_model(args.model, device)
    if args.half:
        model = model.half()
        print(f"Process {rank}: Using model in half-precision!")

    prefix = "<fold2AA>" if args.is_3Di else "<AA2fold>"
    max_residues = 4000
    max_seq_len = 1000
    max_batch = 100
    
    emb_dict = dict()
    batch = list()
    
    # Process sequences
    # They are already sorted by length from the main process if we did round-robin correctly
    
    for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_list, desc=f"Proc {rank}", position=rank)):
        # replace non-standard AAs
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch (batch already includes current sequence)
        n_res_batch = sum([s_len for _, _, s_len in batch]) 
        
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_list)-1 or seq_len > max_seq_len:
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
                print(f"RuntimeError during embedding for {pdb_id} (L={seq_len})")
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx, 1:s_len+1]
                
                if args.per_protein:
                    emb = emb.mean(dim=0)
                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()
                
    # Write to temp file
    print(f"Process {rank}: Writing {len(emb_dict)} embeddings to {temp_output_path}")
    with h5py.File(str(temp_output_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)
    
    return True

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""
    parser = argparse.ArgumentParser(description='Multi-GPU embed.py')
    
    parser.add_argument('-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')
    parser.add_argument('-o', '--output', required=True, type=str, 
                    help='A path for saving the created embeddings as NumPy npz file.')
    parser.add_argument('--model', required=False, type=str,
                    default="Rostlab/ProstT5",
                    help='Either a path to a directory holding the checkpoint or huggingface repo link.')
    parser.add_argument('--split_char', type=str, default='!',
                    help="The character for splitting the FASTA header.")
    parser.add_argument('--id', type=int, default=0,
                    help='The index for the uniprot identifier field.')
    parser.add_argument('--per_protein', type=int, default=0,
                    help="Whether to return per-residue embeddings (0) or mean-pooled per-protein (1).")
    parser.add_argument('--half', type=int, default=0,
                    help="Whether to use half_precision (1) or not (0).")
    parser.add_argument('--is_3Di', type=int, default=0,
                    help="Whether to create embeddings for 3Di (1) or AA (0).")
    parser.add_argument('--gpus', type=str, default=None,
                    help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). Default: all available.")
    
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Process boolean args
    args.per_protein = bool(args.per_protein)
    args.half = bool(args.half)
    args.is_3Di = bool(args.is_3Di)

    # Setup devices
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        print("No CUDA devices available. Falling back to CPU (single process).")
        gpu_ids = []

    if not gpu_ids:
        print("No CUDA devices available/specified. Falling back to CPU (single process).")
        gpu_ids = ['cpu']

    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs: {gpu_ids}")

    # Read Sequences
    seq_path = Path(args.input)
    print(f"Reading sequences from {seq_path}...")
    seq_dict = read_fasta(seq_path, args.split_char, args.id, args.is_3Di)
    
    # Sort by length (descending) to optimize batching
    # We will distribute round-robin so each GPU gets a mix of lengths but still ordered locally
    print("Sorting sequences by length...")
    sorted_items = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    # Partition
    partitions = [[] for _ in range(num_gpus)]
    for i, item in enumerate(sorted_items):
        partitions[i % num_gpus].append(item)
        
    # Check distribution and handle empty partitions
    for i, p in enumerate(partitions):
        if len(p) == 0:
            print(f"Warning: GPU {gpu_ids[i]} has no sequences assigned. Consider using fewer GPUs.")
        else:
            print(f"GPU {gpu_ids[i]}: {len(p)} sequences")

    # Prepare output paths
    output_path = Path(args.output)
    temp_dir = output_path.parent / f"temp_embed_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_files = []
    processes = []
    
    try:
        # Spawn processes
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, which is fine
            pass
        
        for rank, gpu_id in enumerate(gpu_ids):
            if len(partitions[rank]) == 0:
                print(f"Skipping GPU {gpu_id} (rank {rank}) - no sequences assigned")
                continue
            temp_file = temp_dir / f"part_{rank}.h5"
            temp_files.append(temp_file)
            
            p = mp.Process(
                target=worker_process,
                args=(rank, gpu_id, partitions[rank], args, temp_file)
            )
            p.start()
            processes.append(p)
            
        # Wait for completion
        for p in processes:
            p.join()
            
        # Check exit codes
        failed_processes = [(i, p.exitcode) for i, p in enumerate(processes) if p.exitcode != 0]
        if failed_processes:
            failed_info = ", ".join([f"rank {i} (exit code {code})" for i, code in failed_processes])
            raise RuntimeError(f"One or more worker processes failed: {failed_info}")
            
        # Merge files
        print("Merging temporary files...")
        start_merge = time.time()
        with h5py.File(str(output_path), "w") as combined_h5:
            for temp_file in temp_files:
                print(f"Merging {temp_file}...")
                with h5py.File(str(temp_file), "r") as part_h5:
                    for key in part_h5.keys():
                        part_h5.copy(key, combined_h5)
                        
        print(f"Merged file saved to {output_path}")
        print(f"Merge time: {time.time() - start_merge:.2f}s")
        
    finally:
        # Cleanup
        if temp_dir.exists():
            print(f"Cleaning up {temp_dir}...")
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()

