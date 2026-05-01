# Hidden State Genomics

Graph-based analysis of sparse autoencoder (SAE) feature activity in genomic language models (gLMs).

This repository contains the codebase, analysis workflows, and manuscript assets for Hidden State Genomics, a mechanistic interpretability study of the InstaDeep Nucleotide Transformer.

## Study Overview

The computational pipeline used in this study is:

`genomic sequences -> transformer hidden states -> SAE latents -> graph + intervention analysis -> biological hypotheses`

Core methodological components implemented in this repository include:

- Training SAEs over NTv2 hidden states across encoder layers.
- Constructing typed token-to-feature and sequence-to-feature multigraphs.
- Comparing cisplatin-binding and non-binding sequence communities.
- Performing latent interventions with downstream CNN-based evaluation.

## Repository Structure

- `hsg/sae/`: SAE architecture, training loop, and training protocol code.
- `hsg/stattools/`: wrappers for loading parent model + SAE for inference.
- `hsg/pipelines/`: hidden state extraction, variant mapping, and data ETL utilities.
- `hsg/featureanalysis/`: knowledge graph construction, correlation analysis, and interventions.
- `hsg/cisplatinRNA/`: downstream CNN classifier heads and training scripts.
- `hsg/depend/`: dependency/sensitivity mapping utilities.
- `hsg/tests/`: environment, pipeline, and SAE unit tests.
- `publication/`: manuscript and figures.
- `data/`: local datasets, annotations, and generated analysis outputs.
- `checkpoints/`: SAE checkpoints and metadata.

## Software Requirements

- Python `3.12.8` (pinned in `pyproject.toml`)
- PyTorch `2.5.1`
- transformers `4.47.1`
- biocommons-seqrepo `0.6.9`
- hgvs `1.5.4`
- gffutils

Pinned dependency versions are declared in `pyproject.toml`.

## Installation

1. Clone the repository and move to the project root.

```bash
git clone <your-fork-or-origin-url>
cd Hidden-State-Genomics
```

2. Create and activate a Python 3.12.8 environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

3. Install the project in editable mode.

```bash
pip install -e .
```

4. Pull the SeqRepo database (if not already available locally).

```bash
seqrepo --root-dir ./data/2024-12-20 pull
```

## Environment Configuration

Create a `.env` file in the repository root.

```bash
NT_MODEL="InstaDeepAI/nucleotide-transformer-500m-human-ref"
SEQREPO_PATH="./data/2024-12-20"
REFSEQ_CACHE="./data/refseq_cache.txt"
REFSEQ_GTF="./data/Annotation Data/hg38.ncbiRefSeq.gtf"
GCLOUD_BUCKET="gs://hidden-state-genomics"
CLIN_GEN_CSV="./data/erepo.tabbed.txt"
CLIN_VAR_CSV="./data/variant_summary.txt"
```

Notes:
- Most scripts call `load_dotenv()` and expect these variables to be present.
- Paths may be absolute or relative to the repository root.

## Reproducible Workflows

### 1) SAE Training Across Layers

`hsg/sae/train.py` exposes a command-line interface via `tapify(train_all_layers)`.

```bash
python -m hsg.sae.train \
   --parent_model "$NT_MODEL" \
   --SAE_directory "./checkpoints/hidden-state-genomics/ef8/sae" \
   --log_dir "./log_dir" \
   --epochs 100 \
   --dict_expansion_factor 8 \
   --learning_rate 1e-3 \
   --l1_penalty 1e-3 \
   --l1_annealing_steps 100
```

### 2) Token-Feature Knowledge Graph Construction

```bash
python -m hsg.featureanalysis.featureKG \
   --input data/cisplatin_pos.fa \
   --output data/cisplatin_pos_kg.json \
   --exp_factor 8 \
   --sae_dir "./checkpoints/hidden-state-genomics/ef8/sae" \
   --layer_idx 23
```

### 3) Feature Intervention Analysis

```bash
python -m hsg.featureanalysis.intervention \
   --feature 3378 \
   --min_act 0.1 \
   --act_factor 10.0 \
   --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/features.pt \
   --sae gs://hidden-state-genomics/ef8/sae/layer_23.pt \
   --cisplatin_positive data/A2780_Cisplatin_Binding/cisplatin_pos.bed \
   --cisplatin_negative data/A2780_Cisplatin_Binding/cisplatin_neg_45k.bed
```

### 4) CNN Head Training for Downstream Classification

```bash
python -m hsg.cisplatinRNA.CNNtrain \
   --cisplatin_positive data/A2780_Cisplatin_Binding/cisplatin_pos.bed \
   --cisplatin_negative data/A2780_Cisplatin_Binding/cisplatin_neg_45k.bed \
   --layer_idx 23 \
   --exp_factor 8 \
   --condition all
```

## Reproducibility Notes

- SAE checkpoints are organized by expansion factor and layer in `checkpoints/`.
- Generated intervention outputs are stored in `data/intervention_reports/`.
- Knowledge graphs are serialized to JSON (`*_kg.json`) in a NetworkX-compatible format.
- Large artifacts can be staged in Google Cloud Storage (`gs://hidden-state-genomics/`).
- If CUDA memory is insufficient, key model-loading paths attempt CPU fallback.

## Validation and Tests

Run the full test suite:

```bash
python -m unittest
```

Run specific test modules:

```bash
python -m unittest -v hsg.tests.test_env_vars
python -m unittest -v hsg.tests.test_pipelines
python -m unittest -v hsg.tests.test_sae_objects
```

## Data and Code Access

- Source code: https://github.com/ek775/Hidden-State-Genomics
- Sequence inputs and generated outputs are organized under `data/` (some large files are not tracked in git history).
- Reference genome and annotation assets are derived from GRCh38/hg38 and NCBI RefSeq/UCSC resources.

## Citation

If you use this repository, please cite the Hidden State Genomics manuscript (`publication/manuscript.md`) and the relevant upstream model and data sources (including Nucleotide Transformer and SeqRepo).

## License

This project is released under CC0-1.0 (see `LICENSE`).

## Contact

- Eliot Kmiec - ek990@georgetown.edu
- Samuel O'Brien - sto31@georgetown.edu
- Matthew McCoy - mdm299@georgetown.edu

