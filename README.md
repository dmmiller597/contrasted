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

`uv sync` installs the runtime dependencies plus the `dev` group (`pytest`, `ruff`, `ty`). Optional extras:

- `uv sync --extra embed` -- ProstT5/transformers dependencies for generating embeddings.
- `uv sync --extra analysis` -- libraries used by the exploratory scripts under `scripts/` (polars, matplotlib, scipy, umap, etc.).

<details>
<summary><strong>Without uv</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[embed]"
```

</details>

## Quick Start

Three console scripts are installed with the package:

| Command | Purpose |
|---|---|
| `contrasted-train` | Train a model. |
| `contrasted-make-db` | Build a vector database from a checkpoint. |
| `contrasted-annotate` | Annotate queries against a vector database (pass `compute_metrics=true` to also write a selective-classification curve when truth labels are available). |

All scripts are Hydra entry points; override any config key on the command line.

### Train

```bash
uv run contrasted-train
uv run contrasted-train +experiment=supcon datamodule.batch_size=512
```

### Build vector database

```bash
uv run contrasted-make-db \
    model_path=outputs/checkpoints/best.ckpt \
    embedding_dir=data/cath-c123-S100-prostt5 \
    index_path=data/vector_db/train.pt
```

### Annotate sequences

```bash
uv run contrasted-annotate \
    model_path=outputs/checkpoints/best.ckpt \
    index=data/vector_db/train.pt \
    embedding_dir=data/cath-c123-S100-prostt5 \
    input=data/clustered_datasets/test/s30.fasta \
    compute_metrics=true   # writes metrics.json + selective_curve.tsv when labels are set
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
