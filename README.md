<p align="center">
  <img src="contrasted-logo.png" alt="Contrasted Logo" style="max-width: 100%; height: auto;"/>
</p>

Supervised contrastive learning for CATH protein superfamily classification.

Headline ContrasTED uses ProstT5 AA∥3Di (2048-d) inputs and a 128-d CCL projection head. Annotation is nearest-centroid in that space, with a distance cutoff.

## Installation

```bash
pip install contrasted
```

From a clone:

```bash
git clone https://github.com/dmmiller597/contrasted
cd contrasted
uv sync
```

## Get the production head and CATH index

Weights are not in git. Put them in `$CONTRASTED_DATA_DIR` (default `~/.cache/contrasted`):

| File | Role |
|------|------|
| `aa3di_s40_seed40_head.pt` | Canonical S40 AA∥3Di head |
| `cath_s40_centroids.pt` | CATH S40 centroid index |
| `aa3di_s20_seed40_head.pt` | Optional S20 head |

The Zenodo DOI is not minted yet. Filenames, sha256 hashes, and the S40 cutoff (`0.24699`) are in `src/contrasted/assets.json`. `contrasted-annotate` looks in the cache directory for those names if the path you pass does not exist.

```bash
export CONTRASTED_DATA_DIR="${CONTRASTED_DATA_DIR:-$HOME/.cache/contrasted}"
mkdir -p "$CONTRASTED_DATA_DIR"
# wget the user bundle into $CONTRASTED_DATA_DIR once the DOI is live
```

## Quick start

You still pass a prebuilt query `EmbeddingStore` (`embedding_dir`). Embed FASTA first, then annotate against the downloaded CATH index. You do not run `make-db` unless you want a custom reference set.

### 1. Embed queries

AA-only (1024-d):

```bash
uv run contrasted-embed \
  input=queries.fasta \
  output_dir=data/embeddings/queries-aa
```

Headline AA∥3Di (2048-d) needs an AA store plus 3Di vectors, then:

```bash
uv run contrasted-build-concat-store \
  --aa-store data/embeddings/queries-aa \
  --di-cache-npy path/to/3di_embeddings.npy \
  --di-cache-ids path/to/3di_embedding_ids.txt \
  --output-dir data/embeddings/queries-aa3di
```

Direct FASTA or PDB embedding inside `contrasted-annotate` is planned. Until then, `embedding_dir` is required.

### 2. Annotate

```bash
uv run contrasted-annotate \
  input=queries.fasta \
  embedding_dir=data/embeddings/queries-aa3di
```

Defaults: centroid search, the S40 head and CATH centroid index from `$CONTRASTED_DATA_DIR`, cutoff `0.24699`. Override `model_path` / `index` with a path or a filename that lives in the cache directory.

Output is `annotations.tsv` (`query_id`, `predicted_annotation`, `distance`, …). Neighbours beyond `distance_cutoff` become `unknown`. Queries missing from the store become `missing_embedding`.

### Custom reference set

```bash
uv run contrasted-make-db \
  input=reference.fasta \
  embedding_dir=data/embeddings/reference-aa3di \
  label_file=reference-labels.txt \
  index_path=my_index.pt
```

Then annotate with `index=my_index.pt`. Use `--config-name=make_db_s20` for the S20 head.

### Train (optional)

```bash
uv run contrasted-train --config-name=train/cath_s40_aa3di
uv run contrasted-train --config-name=train/cath_s20_aa3di
```

Both use the reported center-contrastive recipe and `input_dim: 2048`. Override `datamodule.embedding_dir` and the split FASTAs to match local stores.

## Data format

Inputs:

- FASTA header: `>cath|{version}|{domain_id}/{start}-{end}` (CATH) or plain ids.
- Embedding directory (`EmbeddingStore`):
  - `embeddings.npy` `(N, D)` float16/float32
  - `labels.npy` `(N,)` int64 (optional for inference)
  - `ids.txt` one domain id per line
  - `metadata.json` with at least `dims` / `count` / `dtype`
  - Headline stores set `modality: aa_3di_concat` with `aa_dim` / `di_dim`

A head with `input_dim=2048` requires an AA∥3Di store. AA-only ProstT5 stores are 1024-d.

## Console scripts

| Script | Role |
|--------|------|
| `contrasted-embed` | Encode FASTA with ProstT5 (AA) |
| `contrasted-build-concat-store` | Build a 2048-d AA∥3Di `EmbeddingStore` |
| `contrasted-annotate` | Centroid or k-NN annotate against an index |
| `contrasted-make-db` | Project a reference set into a vector index |
| `contrasted-train` | Train the projection head (Hydra) |

## Citation

Paper, Zenodo archive, and the analysis/reproduction tree will be linked here once they are public. This repository is the installable method.
