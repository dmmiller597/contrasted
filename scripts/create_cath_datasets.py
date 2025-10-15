import os
import subprocess
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_fasta(file_path: Path) -> dict[str, str]:
    """Parses a FASTA file into a dictionary of headers to sequences."""
    sequences = {}
    current_header = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_header = line[1:]
                sequences[current_header] = ''
            elif current_header:
                sequences[current_header] += line
    return sequences

def write_fasta(file_path: Path, sequences: dict[str, str]):
    """Writes a dictionary of sequences to a FASTA file."""
    with open(file_path, 'w') as f:
        for header, seq in sequences.items():
            f.write(f'>{header}\n{seq}\n')

def get_cath_label(header: str) -> str:
    """Extracts the CATH superfamily label from a FASTA header."""
    
    # Load the domain to superfamily mapping if not already loaded
    if not hasattr(get_cath_label, '_sf_mapping'):
        mapping = {}
        try:
            # The label file maps CATH domain IDs to superfamily IDs.
            script_dir = Path(__file__).parent.parent  # Go up to project root
            mapping_file = script_dir / 'data' / 'cath-domain-sf-list.txt'
            with open(mapping_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        domain_id, sf_label = parts
                        mapping[domain_id] = sf_label
            get_cath_label._sf_mapping = mapping
        except FileNotFoundError:
            logging.error("Could not find data/cath-domain-sf-list.txt file. This file is required.")
            get_cath_label._sf_mapping = {}
            # Exit or raise might be better if the script can't proceed. For now, it will fail later.

    # Extract domain ID from header. Example: >cath|4_4_0|107lA00/1-162 -> 107lA00
    try:
        domain_id = header.split('|')[2].split('/')[0]
    except IndexError:
        logging.warning(f"Could not parse CATH domain ID from header: {header}")
        return ""

    # Look up the superfamily label for this domain ID
    sf_label = get_cath_label._sf_mapping.get(domain_id, "")
    if not sf_label:
        logging.warning(f"Could not find CATH superfamily label for domain ID: '{domain_id}' (from header: '{header}')")
    
    return sf_label

def run_mmseqs2(
    input_file: Path, 
    output_dir: Path, 
    identity: float, 
    coverage: float = 0.8,
    sensitivity: Optional[float] = None
) -> Path:
    """Runs MMSeqs2 easy-cluster to cluster sequences."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define simple, relative paths for MMSeqs2 to use inside its working directory
    cluster_file_base_name = f"cluster_{int(identity*100)}"
    tmp_dir_name = "tmp"

    # Resolve input_file to an absolute path to ensure it's found by the subprocess.
    # The output and tmp paths are relative, as they are used within the `cwd`.
    cmd = [
        "mmseqs", "easy-cluster", str(input_file.resolve()), 
        cluster_file_base_name, 
        tmp_dir_name,
        "--min-seq-id", str(identity),
        "-c", str(coverage),
        "--cov-mode", "0",
        "--cluster-mode", "2",
        "--threads", str(os.cpu_count() or 1)
    ]
    if sensitivity is not None:
        cmd.extend(["-s", str(sensitivity)])
    
    logging.info(f"Running MMSeqs2 for {int(identity*100)}% identity...")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command from within the specified output directory.
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=output_dir)
        logging.info("MMSeqs2 completed successfully.")
        logging.debug(f"MMSeqs2 stdout:\n{result.stdout}")
        logging.debug(f"MMSeqs2 stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"MMSeqs2 failed with exit code {e.returncode}")
        logging.error(f"Stderr: {e.stderr}")
        raise
    
    # The output cluster file will be inside the output_dir.
    cluster_tsv = output_dir / f"{cluster_file_base_name}_cluster.tsv"
    if not cluster_tsv.is_file():
        raise FileNotFoundError(f"MMSeqs2 did not produce the expected cluster file: {cluster_tsv}")
        
    return cluster_tsv

def parse_clusters(cluster_file: Path) -> dict[str, list[str]]:
    """Parses the MMSeqs2 cluster file."""
    clusters = defaultdict(list)
    with open(cluster_file, 'r') as f:
        for line in f:
            rep, member = line.strip().split('\t')
            clusters[rep].append(member)
    return dict(clusters)

def generate_summary(
    final_training_set: dict[str, str],
    final_validation_set: dict[str, str],
    final_test_sets: dict[int, dict[str, str]]
) -> str:
    """Generates a formatted text summary of the created datasets."""
    
    final_master_train_labels = {get_cath_label(h) for h in final_training_set.keys()}
    final_master_val_labels = {get_cath_label(h) for h in final_validation_set.keys()}

    summary_data = []
    for identity, test_seqs in sorted(final_test_sets.items()):
        final_test_labels = {get_cath_label(h) for h in test_seqs.keys()}
        summary_data.append({
            "Identity": f"S{identity}",
            "Test Seqs": len(test_seqs),
            "Test SFs": len(final_test_labels),
        })

    lines = []
    lines.append("="*90)
    lines.append(" " * 30 + "Final Dataset Creation Summary")
    lines.append("="*90)
    lines.append(f"Final Training Set:     {len(final_training_set):>7} sequences, {len(final_master_train_labels):>5} SFs")
    lines.append(f"Final Validation Set:   {len(final_validation_set):>7} sequences, {len(final_master_val_labels):>5} SFs")
    lines.append("-" * 90)

    if summary_data:
        headers = summary_data[0].keys()
        col_widths = {h: max(len(h), max((len(str(r[h])) for r in summary_data), default=0)) for h in headers}
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        for result in summary_data:
            row_line = " | ".join(str(result[h]).ljust(col_widths[h]) for h in headers)
            lines.append(row_line)
    else:
        lines.append("No test sets were generated.")
    lines.append("="*90)
    
    return "\n".join(lines)

def main(args):
    input_fasta = Path(args.input_fasta)
    if not input_fasta.is_file():
        logging.error(f"Input FASTA file not found at: {input_fasta}")
        return

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logging.info("--- Script Configuration ---")
    for key, value in vars(args).items():
        logging.info(f"{key:<25}: {value}")
    logging.info("----------------------------")

    logging.info("Parsing all sequences from input FASTA...")
    all_sequences = parse_fasta(input_fasta)
    logging.info(f"Found {len(all_sequences)} total sequences.")

    if not args.include_class_4_and_6:
        logging.info("Excluding sequences from CATH Class 4 and 6 (default behavior)...")
        original_count = len(all_sequences)
        sequences_to_keep = {
            h: s for h, s in all_sequences.items() 
            if not (get_cath_label(h).startswith('4.') or get_cath_label(h).startswith('6.'))
        }
        all_sequences = sequences_to_keep
        logging.info(f"Removed {original_count - len(all_sequences)} sequences from Class 4 and 6. Pool now contains {len(all_sequences)} sequences.")

    # --- Strategy ---
    # 1. FIRST, cluster at validation identity threshold and carve out validation set.
    #    This ensures validation sequences have <=val_identity% similarity to training.
    # 2. Iteratively build test sets from highest identity (90) to lowest (10).
    #    In each iteration, cluster the remaining pool, identify test clusters, and add their
    #    representatives to the corresponding test set. Then, remove ALL MEMBERS of these
    #    test clusters from the pool to prevent data leakage.
    # 3. After validation and test sets are carved out, the remaining pool becomes training.
    # 4. Perform a final label consistency check across all sets.

    pool_for_clustering = all_sequences.copy()
    logging.info(f"Initial pool for clustering: {len(pool_for_clustering)} sequences.")

    # --- Step 1: Create Validation Set from Validation Identity Clustering ---
    logging.info(f"\n--- Step 1: Creating Validation Set at {args.val_identity_threshold}% Identity ---")
    final_validation_set = {}
    
    if args.val_ratio > 0 and args.val_identity_threshold > 0:
        # Create temporary fasta for validation clustering
        val_pool_fasta = output_base_dir / f"temp_pool_val_{args.val_identity_threshold}.fasta"
        write_fasta(val_pool_fasta, pool_for_clustering)
        
        val_identity = args.val_identity_threshold / 100.0
        val_cluster_dir = output_base_dir / f"validation_clustering_s{args.val_identity_threshold}"
        
        # Use high sensitivity if needed
        val_sensitivity = args.high_sensitivity if val_identity < args.sensitivity_threshold else None
        
        try:
            val_cluster_file = run_mmseqs2(
                val_pool_fasta,
                val_cluster_dir,
                val_identity,
                args.coverage,
                sensitivity=val_sensitivity
            )
            
            val_clusters = parse_clusters(val_cluster_file)
            all_val_reps = list(val_clusters.keys())
            
            # Stratified sampling for validation
            val_rep_to_label = {rep: get_cath_label(rep) for rep in all_val_reps if get_cath_label(rep)}
            val_label_counts = Counter(val_rep_to_label.values())
            val_splittable_reps = [
                rep for rep in all_val_reps 
                if rep in val_rep_to_label and val_label_counts[val_rep_to_label[rep]] >= args.min_label_count_for_split
            ]
            
            val_selected_reps = set()
            if val_splittable_reps:
                val_splittable_labels = [val_rep_to_label[rep] for rep in val_splittable_reps]
                try:
                    _, selected_val_reps = train_test_split(
                        val_splittable_reps,
                        test_size=args.val_ratio,
                        stratify=val_splittable_labels,
                        random_state=args.random_state
                    )
                    val_selected_reps = set(selected_val_reps)
                except ValueError:
                    logging.warning("Stratified validation sampling failed. Falling back to random sampling.")
                    _, selected_val_reps = train_test_split(
                        val_splittable_reps,
                        test_size=args.val_ratio,
                        random_state=args.random_state
                    )
                    val_selected_reps = set(selected_val_reps)
            
            # Create validation set from selected representatives
            final_validation_set = {
                rep: all_sequences[rep] for rep in val_selected_reps if rep in all_sequences
            }
            
            # Remove ALL members of validation clusters from pool
            val_members_to_remove = set()
            for rep in val_selected_reps:
                val_members_to_remove.update(val_clusters.get(rep, []))
                val_members_to_remove.add(rep)
            
            pool_for_clustering = {
                h: s for h, s in pool_for_clustering.items() if h not in val_members_to_remove
            }
            
            logging.info(f"Sampled {len(val_selected_reps)} clusters for validation set ({args.val_ratio:.1%} of splittable clusters).")
            logging.info(f"Validation set contains {len(final_validation_set)} representatives.")
            logging.info(f"Removed {len(val_members_to_remove)} members of validation clusters from pool.")
            logging.info(f"Remaining pool for test/train: {len(pool_for_clustering)} sequences.")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"Failed to create validation set. Falling back to old method. Error: {e}")
            final_validation_set = {}
        
        # Clean up temp file
        if val_pool_fasta.exists():
            val_pool_fasta.unlink()
    else:
        logging.info("Validation ratio is 0 or validation identity threshold is 0. Skipping validation clustering.")

    # --- Step 2: Create Test Sets by Iterative Clustering ---
    logging.info(f"\n--- Step 2: Creating Test Sets by Iterative Clustering ---")
    
    # This set will accumulate ALL members of ANY cluster that gets chosen for a test set.
    all_test_cluster_members = set()
    final_test_sets = {}
    identities = sorted([int(i) for i in args.identity_thresholds.split(',')], reverse=True)
    
    for identity_percent in identities:
        logging.info(f"\n--- Processing {identity_percent}% Identity Test Set ---")
        
        if not pool_for_clustering:
            logging.warning(f"Clustering pool is empty. Stopping at {identity_percent}%.")
            break

        # Create a temporary fasta file for the current (shrinking) pool
        pool_fasta_path = output_base_dir / f"temp_pool_{identity_percent}.fasta"
        write_fasta(pool_fasta_path, pool_for_clustering)

        identity_threshold = identity_percent / 100.0
        identity_dir = output_base_dir / f"s{identity_percent}"
        
        # Use high sensitivity only for thresholds below the specified value.
        sensitivity_to_use = args.high_sensitivity if identity_threshold < args.sensitivity_threshold else None

        try:
            cluster_file = run_mmseqs2(
                pool_fasta_path,
                identity_dir,
                identity_threshold,
                args.coverage,
                sensitivity=sensitivity_to_use
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"Failed to generate clusters for {identity_percent}%. Skipping. Error: {e}")
            if pool_fasta_path.exists():
                pool_fasta_path.unlink()
            continue
        
        clusters_from_current_pool = parse_clusters(cluster_file)
        
        # --- Sample a fixed number of representatives for the test set using stratified random sampling ---
        all_reps_in_clustering = list(clusters_from_current_pool.keys())
        
        # Ensure we have labels for stratification and filter out reps from rare classes
        rep_to_label = {rep: get_cath_label(rep) for rep in all_reps_in_clustering if get_cath_label(rep)}
        label_counts = Counter(rep_to_label.values())
        splittable_reps = [
            rep for rep in all_reps_in_clustering 
            if rep in rep_to_label and label_counts[rep_to_label[rep]] >= args.min_label_count_for_split
        ]

        new_test_reps = set()

        if splittable_reps and args.test_set_ratio > 0:
            splittable_labels = [rep_to_label[rep] for rep in splittable_reps]
            # Use train_test_split to get a stratified random sample based on the ratio
            try:
                _, selected_reps = train_test_split(
                    splittable_reps,
                    test_size=args.test_set_ratio,
                    stratify=splittable_labels,
                    random_state=args.random_state
                )
                new_test_reps = set(selected_reps)
            except ValueError:
                logging.warning(f"Stratified sampling failed for S{identity_percent}. Falling back to random sampling.")
                # Fallback to non-stratified sampling if stratification is not possible
                _, selected_reps = train_test_split(
                    splittable_reps,
                    test_size=args.test_set_ratio,
                    random_state=args.random_state
                )
                new_test_reps = set(selected_reps)

        logging.info(f"Sampled {len(new_test_reps)} representatives for the S{identity_percent} test set ({args.test_set_ratio:.1%} of splittable clusters).")
        
        # The test set for this identity level consists ONLY of these new representatives.
        final_test_sets[identity_percent] = {
            rep: all_sequences[rep] for rep in new_test_reps if rep in all_sequences
        }
        
        # Accumulate all members of the new test clusters. These will be excluded from the pool for subsequent clustering.
        members_to_remove = set()
        for rep in new_test_reps:
            # Remove all cluster members associated with the selected representative
            members_to_remove.update(clusters_from_current_pool.get(rep, []))
            # ALSO remove the representative itself to avoid it leaking into the training set
            members_to_remove.add(rep)
        
        all_test_cluster_members.update(members_to_remove)

        # The pool for the NEXT iteration is the current pool, minus all members of the clusters just selected for the test set.
        pool_for_clustering = {
            h: s for h, s in pool_for_clustering.items() if h not in members_to_remove
        }
        
        logging.info(f"Removed {len(members_to_remove)} members of new test clusters. Total quarantined: {len(all_test_cluster_members)}.")
        logging.info(f"Next clustering pool contains {len(pool_for_clustering)} sequences.")

        # Clean up the temporary file for this iteration
        if pool_fasta_path.exists():
            pool_fasta_path.unlink()

    # The final training pool is what remains in the pool_for_clustering after validation and test members have been removed.
    final_training_set = pool_for_clustering

    # --- Step 3: Final Label Consistency Check ---
    logging.info("\n--- Step 3: Final Label Consistency Check ---")
    logging.info(f"Final training set contains {len(final_training_set)} sequences.")
    final_train_labels = {get_cath_label(h) for h in final_training_set.keys()}
    logging.info(f"Final training set contains {len(final_train_labels)} unique labels.")

    # Filter validation set
    original_val_len = len(final_validation_set)
    final_validation_set = {h: s for h, s in final_validation_set.items() if get_cath_label(h) in final_train_labels}
    logging.info(f"Filtered validation set from {original_val_len} to {len(final_validation_set)} sequences to match training labels.")

    # Filter all test sets
    for identity in final_test_sets:
        original_test_len = len(final_test_sets[identity])
        final_test_sets[identity] = {
            h: s for h, s in final_test_sets[identity].items() if get_cath_label(h) in final_train_labels
        }
        logging.info(f"Filtered S{identity} test set from {original_test_len} to {len(final_test_sets[identity])} sequences.")

    # --- Save aggregated and filtered datasets ---
    # Save the single, final training set
    write_fasta(output_base_dir / 'train.fasta', final_training_set)
    write_fasta(output_base_dir / 'val.fasta', final_validation_set)
    logging.info(f"\nSaved final training set to {output_base_dir / 'train.fasta'}")
    logging.info(f"Saved final validation set to {output_base_dir / 'val.fasta'}")

    # Save each test set in its identity-specific directory
    for identity, test_seqs in final_test_sets.items():
        identity_dir = output_base_dir / f"s{identity}"
        identity_dir.mkdir(parents=True, exist_ok=True)
        write_fasta(identity_dir / 'test.fasta', test_seqs)
        logging.info(f"Saved S{identity} test set to {identity_dir / 'test.fasta'}")

    # --- Update and Print Final Summary Table ---
    summary_text = generate_summary(final_training_set, final_validation_set, final_test_sets)

    # Print to console
    print("\n" + summary_text)

    # Save to file
    if args.summary_file:
        summary_file_path = output_base_dir / args.summary_file
        try:
            with open(summary_file_path, 'w') as f:
                f.write(summary_text)
            logging.info(f"Summary table saved to {summary_file_path}")
        except IOError as e:
            logging.error(f"Failed to write summary file to {summary_file_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create CATH train/validation/test splits by iteratively clustering with MMSeqs2. "
                    "First, validation set is carved out at a specified identity threshold (default S50) to ensure "
                    "validation sequences have <=50%% similarity to training (good for early stopping). "
                    "Then, at each test identity level, a fraction of clusters is sampled for the test set, "
                    "and its members are removed from the pool for subsequent, lower-identity clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_fasta", 
        type=str, 
        default="data/sequence-data/cath-domain-seqs-S100.fa",
        help="Path to the input FASTA file with CATH domain IDs in the header."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/clustered_datasets",
        help="Base directory to save the clustered dataset files."
    )
    parser.add_argument(
        "--identity_thresholds",
        type=str,
        default="10,20,30,40,50,60,70,80,90",
        help="Comma-separated list of identity percentages to process (e.g., '30,50,70')."
    )
    parser.add_argument(
        "--test_set_ratio",
        type=float,
        default=0.05,
        help="Fraction of clusters at each identity step to hold out for the test set."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of clusters at validation identity level to use for the validation set."
    )
    parser.add_argument(
        "--val_identity_threshold",
        type=int,
        default=50,
        help="Identity percentage for validation set clustering (e.g., 50 for S50). "
             "Ensures validation sequences are <=X%% similar to training for better generalization testing. "
             "Set to 0 to disable dedicated validation clustering (will use old method)."
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Minimum coverage for MMSeqs2 clustering ('-c' parameter)."
    )
    parser.add_argument(
        "--sensitivity_threshold",
        type=float,
        default=0.5,
        help="Sequence identity threshold below which high sensitivity MMSeqs2 settings are used."
    )
    parser.add_argument(
        "--high_sensitivity",
        type=float,
        default=7.5,
        help="Sensitivity for identities below the sensitivity threshold."
    )
    parser.add_argument(
        "--min_label_count_for_split",
        type=int,
        default=3,
        help="Minimum number of representatives for a CATH label to be included in test/validation splits."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility of splits."
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="summary.txt",
        help="Name of the file in the output directory to save the final summary table. If empty, not saved."
    )
    parser.add_argument(
        "--include_class_4_and_6",
        action="store_true",
        help="If set, include CATH superfamilies belonging to Class 4 and 6. By default, they are excluded."
    )
    
    args = parser.parse_args()
    main(args) 