# Contrasted
![Uploading Generated Image January 22, 2026 - 3_54PM.png…]()

**Supervised contrastive learning for CATH protein superfamily classification**

This project trains a projection head using contrastive learning to create high-quality protein embeddings for superfamily classification. The learned embeddings can be used for fast k-NN annotation of new sequences.

## Overview

The pipeline consists of three main stages:

1. **Training** (`train.py`): Train a projection head using supervised contrastive learning
2. **Database Creation** (`make_db.py`): Create a FAISS vector database from trained embeddings
3. **Annotation** (`annotate.py`): Annotate new sequences using k-NN search

## Installation

```bash
# Clone the repository
git clone https://github.com/dmmiller597/contrasted
cd contrasted

# Install dependencies
pip install -e .

```

## Quick Start

### 1. Train a Model

Train a contrastive learning model on CATH protein embeddings:

```bash
python train.py
```

This will:
- Load pre-computed ProstT5 embeddings (1024-dim)
- Train a projection head to 128-dim using supervised contrastive loss
- Evaluate using k-NN classification

**Configuration**: Edit `configs/train.yaml` or override via command line:

```bash
# Use proxy-anchor loss instead of supcon
python train.py experiment=proxy_anchor

# Override specific parameters
python train.py data.batch_size=512 trainer.max_epochs=50
```

### 2. Create Vector Database

After training, create a FAISS index for fast similarity search:

```bash
python make_db.py \
    model_path=outputs/2025-10-15/21-26-35/checkpoints/best.ckpt \
    index_path=data/vector_db/train.index
```

This will:
- Load the trained model
- Project training embeddings through the model
- Build a FAISS index (inner product for cosine similarity)
- Save index and domain IDs to `data/vector_db/`

**Configuration**: Edit `configs/make_db.yaml` or override via command line.

### 3. Annotate New Sequences

Annotate test sequences using k-NN search:

```bash
python annotate.py \
    model_path=outputs/2025-10-15/21-26-35/checkpoints/best.ckpt \
    index=data/vector_db/train.index \
    output_path=outputs/annotations/test_annotations.tsv
```

This will:
- Load the trained model and FAISS index
- Project query embeddings through the model
- Find k-nearest neighbors for each query
- Transfer annotations via majority vote
- Save results with confidence scores

**Configuration**: Edit `configs/annotate.yaml` or override via command line.

## Data Format

### Input Files

- **FASTA files**: Protein sequences in FASTA format
  ```
  >cath|4_4_0|12e8H01/1-113
  MKKYTCTVCGYIYNPEDGDPDNGVNPGTDFKDIPDDWVCPLCGVGKDQFEEVEE
  ```

- **HDF5 embeddings**: Pre-computed ProstT5 embeddings (1024-dim)
  - Keys: `cath|4_4_0|12e8H01_1-113` (note: `/` replaced with `_`)
  - Values: numpy arrays of shape `(1024,)`

- **Label file**: Domain ID to superfamily mapping (TSV)
  ```
  12e8H01    1.10.8.10
  12gsA00    3.40.50.720
  ```

### Output Files

- **Annotations** (`annotate.py`): TSV file with columns:
  - `query_id`: Query domain ID
  - `predicted_annotation`: Predicted superfamily
  - `distance`: Cosine distance to nearest neighbor (optional)
  - `confidence`: Confidence score [0, 1] (optional)

## Project Structure

```
contrasted/
├── contrasted/           # Core package
│   ├── data.py          # Data loading and preprocessing
│   ├── model.py         # Model architecture
│   ├── losses.py        # Contrastive loss functions
│   ├── callbacks.py     # Training callbacks (k-NN evaluation)
│   └── utils.py         # Utility functions
├── configs/             # Hydra configuration files
│   ├── train.yaml       # Training configuration
│   ├── make_db.yaml     # Database creation configuration
│   ├── annotate.yaml    # Annotation configuration
│   └── experiment/      # Experiment-specific configs
├── scripts/             # Helper scripts
│   └── embed.py         # Generate ProstT5 embeddings
├── train.py             # Training script
├── make_db.py           # Database creation script
└── annotate.py          # Annotation script
```
