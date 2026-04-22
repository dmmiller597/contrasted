<p align="center">
  <img src="contrasted-logo.png" alt="Contrasted Logo" style="max-width: 100%; height: auto;"/>
</p>

Supervised contrastive learning for CATH protein superfamily classification.


## Installation

```bash
git clone https://github.com/dmmiller597/contrasted
cd contrasted
uv sync
```

`uv sync` installs everything needed for training, embedding, make-db, and annotation, plus the `dev` group (`pytest`, `ruff`, `ty`). Optional extras:

- `uv sync --extra cloud` -- Modal dependency for the cloud embedding scripts under `scripts/`.
- `uv sync --extra analysis` -- libraries used by the exploratory scripts under `scripts/` (polars, matplotlib, scipy, umap, etc.).

<details>
<summary><strong>Without uv</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

</details>

## Quick Start

Four console scripts are installed with the package:

| Command | Purpose |
|---|---|
| `contrasted-train` | Train a model. |
| `contrasted-embed` | Encode a FASTA into a reusable ProstT5 embedding directory. |
| `contrasted-make-db` | Build a vector database from a checkpoint (accepts FASTA directly). |
| `contrasted-annotate` | Annotate queries against a vector database (accepts FASTA directly; pass `compute_metrics=true` to also write a selective-classification curve when truth labels are available). |

All scripts are Hydra entry points; override any config key on the command line.

### Annotate a FASTA (one step)

```bash
uv run contrasted-annotate \
    input=queries.fasta \
    model_path=outputs/checkpoints/best.ckpt \
    index=data/vector_db/train.pt
```

`contrasted-annotate` encodes `queries.fasta` with ProstT5 in-process, projects the embeddings through the checkpoint, and writes the TSV. No preprocessing step required.

For repeat runs, set `embedding_dir=path/to/cache` -- on-the-fly embeddings are written there once and loaded on subsequent runs:

```bash
uv run contrasted-annotate \
    input=queries.fasta \
    model_path=outputs/checkpoints/best.ckpt \
    index=data/vector_db/train.pt \
    embedding_dir=cache/queries-prostt5
```

Or precompute explicitly:

```bash
uv run contrasted-embed input=queries.fasta output_dir=cache/queries-prostt5
```

### Build a vector database

```bash
uv run contrasted-make-db \
    input=reference.fasta \
    model_path=outputs/checkpoints/best.ckpt \
    index_path=data/vector_db/train.pt
```

`embedding_dir=<path>` may optionally be set to reuse or cache ProstT5 embeddings for the reference set.

### Train

```bash
uv run contrasted-train
uv run contrasted-train +experiment=supcon datamodule.batch_size=512
```

## Data Format

Inputs:
- FASTA header: `>cath|{version}|{domain_id}/{start}-{end}` (CATH) or `>AF-..._TED03`, `>plain_id` (generic).
- Embedding directory:
  - `embeddings.npy` -- `(N, D)` float16/float32
  - `labels.npy` -- `(N,)` int64 (optional for inference, required for training/eval)
  - `ids.txt` -- one domain ID per line
  - `metadata.json` -- at minimum `dims`/`count`/`dtype`; may also include `idx_to_label`
  - `id_to_row.npy` -- optional precomputed `id -> row` mapping

Outputs:
- `annotations.tsv`: `query_id`, `predicted_annotation`, `distance`, `confidence`
- `metrics.json`, `selective_curve.tsv` (when `compute_metrics=true` and truth labels are available)

## Development

```bash
uv sync
uv run ruff check --fix . && uv run ruff format .
uv run pytest
uvx ty check src/
```

<details>
<summary><strong>Without uv</strong></summary>

```bash
python -m pip install -e . --group dev
ruff check --fix . && ruff format .
pytest
```

</details>
