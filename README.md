# Contrasted


Supervised contrastive learning for CATH protein superfamily classification.

Train a projection head on ProstT5 embeddings using contrastive learning for fast k-NN annotation.

## Installation

```bash
git clone https://github.com/dmmiller597/contrasted
cd contrasted
uv sync
```

Optional embeddings dependencies:
```bash
uv sync --extra embed
```

<details>
<summary><strong>Without uv</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

With optional embeddings extras:
```bash
python -m pip install -e ".[embed]"
```

</details>

## Quick Start

### Train
```bash
uv run python train.py
```

Configuration example:
```bash
uv run python train.py model.loss_type=supcon data.batch_size=512
```

<details>
<summary><strong>Without uv</strong></summary>

```bash
python train.py
python train.py model.loss_type=supcon data.batch_size=512
```

</details>

### Build vector database
```bash
uv run python make_db.py \
    model_path=outputs/checkpoints/epoch=108.ckpt \
    index_path=data/vector_db/train.index
```

<details>
<summary><strong>Without uv</strong></summary>

```bash
python make_db.py \
    model_path=outputs/checkpoints/epoch=108.ckpt \
    index_path=data/vector_db/train.index
```

</details>

### Annotate sequences
```bash
uv run python annotate.py \
    model_path=outputs/checkpoints/epoch=108.ckpt \
    index=data/vector_db/train.index \
    input=data/clustered_datasets/test/s30.fasta
```

<details>
<summary><strong>Without uv</strong></summary>

```bash
python annotate.py \
    model_path=outputs/checkpoints/epoch=108.ckpt \
    index=data/vector_db/train.index \
    input=data/clustered_datasets/test/s30.fasta
```

</details>

## Data Format

Inputs:
- FASTA: `>cath|{version}|{domain_id}/{start}-{end}`
- Embeddings: `.pt` with keys `embeddings`, `labels`, `ids`, `idx_to_label`

Outputs:
- Annotations TSV: `query_id`, `predicted_annotation`, `distance`, `confidence`

## Development

```bash
uv sync --extra dev
uv run ruff check --fix . && uv run ruff format .
uv run pytest
```

<details>
<summary><strong>Without uv</strong></summary>

```bash
python -m pip install -e ".[dev]"
ruff check --fix . && ruff format .
pytest
```

</details>
