<p align="center">
  <img src="contrasted-logo.png" alt="contrasted logo" style="max-width: 100%; height: auto;"/>
</p>

Supervised contrastive learning for CATH protein superfamily classification.

contrasted encodes domains with ProstT5 AA∥3Di (2048-d), projects them with a 128-d head, and annotates by nearest superfamily centroid with a distance cutoff.

## Installation

```bash
git clone https://github.com/dmmiller597/contrasted
cd contrasted
uv sync
```

Or `pip install -e .` from the clone. Python 3.11 or 3.12.

## Get the pretrained models and reference index

Weights are not in git. Copy these files into `$CONTRASTED_DATA_DIR` (default `~/.cache/contrasted`):

| File | Role |
|------|------|
| `aa3di_s20_seed40_head.pt` | Projection head (AA∥3Di) |
| `cath_s20_centroids.pt` | CATH superfamily centroid index |

Filenames, sha256 hashes, and the calibrated distance cutoff (`0.24048`) are in `src/contrasted/registry.json`. `contrasted-annotate` looks in `$CONTRASTED_DATA_DIR` for those names if the path you pass does not exist.

```bash
export CONTRASTED_DATA_DIR="${CONTRASTED_DATA_DIR:-$HOME/.cache/contrasted}"
mkdir -p "$CONTRASTED_DATA_DIR"
# copy aa3di_s20_seed40_head.pt and cath_s20_centroids.pt into that directory
```

## Quick start

You pass a prebuilt query `EmbeddingStore` (`embedding_dir`). Embed FASTA first, then annotate against the CATH centroid index. You do not run `make-db` unless you want a custom reference set.

The query FASTA is an ID list. Sequences are looked up in the store, not re-encoded.

### 1. Embed queries

AA-only (1024-d):

```bash
contrasted-embed \
  input=queries.fasta \
  output_dir=data/embeddings/queries-aa
```

The published head is 2048-d. Build an AA∥3Di store from that AA store plus 3Di. Give 3Di as a `.npy` cache, a 3Di FASTA (`--di-fasta`), or PDBs plus Foldseek on `PATH` (`--pdb-dir`). Foldseek is an external binary, not a pip dependency. Without `--domain-ids` or `--split-fastas`, concat uses every id in the AA store.

```bash
contrasted-build-concat-store \
  --aa-store data/embeddings/queries-aa \
  --di-cache-npy path/to/3di_embeddings.npy \
  --di-cache-ids path/to/3di_embedding_ids.txt \
  --output-dir data/embeddings/queries-aa3di
```

`embedding_dir` is required. Direct FASTA or PDB embedding inside `contrasted-annotate` is not supported yet.

### 2. Annotate

```bash
contrasted-annotate \
  input=queries.fasta \
  embedding_dir=data/embeddings/queries-aa3di
```

Defaults: centroid search, the projection head and CATH centroid index from `$CONTRASTED_DATA_DIR`, cutoff `0.24048`. Override `model_path` / `index` with a path or a filename that lives in the cache directory.

Output is `{fasta_stem}_annotations.tsv` under `output_dir` (default `outputs/annotations`). Columns are `query_id`, `predicted_annotation`, `distance`. Neighbours beyond `distance_cutoff` become `unknown`. Queries missing from the store become `missing_embedding`.

Hydra recipes load from the installed `configs` package. To use a different recipe directory, set `CONTRASTED_CONFIG_DIR` before launching the process.

### Custom reference set

```bash
contrasted-make-db \
  input=reference.fasta \
  embedding_dir=data/embeddings/reference-aa3di \
  label_file=reference-labels.txt \
  index_path=my_index.pt
```

Then annotate with `index=my_index.pt`. Default `index_path` is `index.pt` in the current directory. Do not name a custom index `cath_s20_centroids.pt`. That basename is hash-checked against the published file.

### Train (optional)

```bash
contrasted-train --config-name=train/cath_s20_aa3di
```

This uses the reported center-contrastive recipe and `input_dim: 2048`. Override `datamodule.embedding_dir` and the split FASTAs to match local stores. Those FASTAs are not in git.

## Data format

Inputs:

- FASTA header: `>cath|{version}|{domain_id}/{start}-{end}` (CATH) or plain ids.
- Embedding directory (`EmbeddingStore`):
  - `embeddings.npy` `(N, D)` float16/float32
  - `labels.npy` `(N,)` int64 (optional for inference)
  - `ids.txt` one domain id per line
  - `metadata.json` with at least `dims` / `count` / `dtype`
  - AA∥3Di stores set `modality: aa_3di_concat` with `aa_dim` / `di_dim`

A head with `input_dim=2048` requires an AA∥3Di store. AA-only ProstT5 stores are 1024-d.

## Console scripts

| Script | Role |
|--------|------|
| `contrasted-embed` | Encode FASTA with ProstT5 (AA) |
| `contrasted-build-concat-store` | Build a 2048-d AA∥3Di `EmbeddingStore` |
| `contrasted-annotate` | Centroid or k-NN annotate against an index |
| `contrasted-make-db` | Project a reference set into a vector index |
| `contrasted-train` | Train the projection head (Hydra) |

`embed`, `annotate`, `make-db`, and `train` take Hydra `key=value` overrides. `contrasted-build-concat-store` takes argparse `--flags`.

## Citation

Paper and archive links will be added here when they are public.
