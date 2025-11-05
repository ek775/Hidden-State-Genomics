# Hidden-State-Genomics AI Coding Instructions

## Project Overview
This is a mechanistic interpretability project that uses Sparse Auto-Encoders (SAEs) to extract interpretable features from genomic language model embeddings, then constructs knowledge graphs to analyze cisplatin-RNA binding patterns. The pipeline goes: genomic sequences → nucleotide transformer embeddings → SAE features → knowledge graphs → biological analysis.

## Architecture & Key Components

### Core Pipeline (hsg/ directory)
- **`hsg/sae/`**: SAE training and feature extraction
  - `train.py` - Main SAE training with command-line interface via `tapify()`
  - `dictionary.py` - AutoEncoder implementation for sparse feature learning
  - Uses expansion factors (ef8, ef16, ef32) to control dictionary size
- **`hsg/pipelines/`**: Data processing and model integration
  - `hidden_state.py` - Extracts embeddings from nucleotide transformer models
  - `variantmap.py` - DNA variant processing using HGVS and SeqRepo
- **`hsg/featureanalysis/`**: Knowledge graph construction and analysis
  - `featureKG.py` - Main KG construction from SAE features
  - `intervention.py` - Feature intervention experiments
  - Shell scripts for batch processing large datasets

### Command-Line Interface Pattern
All main scripts use `tap` (typed-argument-parser) with `tapify()` for CLI:
```python
if __name__ == "__main__":
    tapify(main_function)
```
Run modules as: `python -m hsg.module.script --arguments`

### Environment Configuration
**Critical**: Always load environment variables first:
```python
from dotenv import load_dotenv
load_dotenv()
```
Required `.env` variables (see test_env_vars.py):
- `NT_MODEL`: Nucleotide transformer model path
- `SEQREPO_PATH`: Local sequence database path
- `REFSEQ_CACHE`: Reference sequence cache
- `GCLOUD_BUCKET`: Cloud storage bucket for models/data

### Data Organization Patterns
- **`data/`**: Large datasets not tracked in git
  - Cisplatin binding data: `cisplatin_pos.fa`, `cisplatin_neg45k.fa`
  - Knowledge graphs: `*_kg.json` files
  - Reference annotations: `Annotation Data/` with GTF files
- **`checkpoints/`**: Trained SAE models organized by expansion factor (ef8/, ef16/, ef32/)

## Development Workflows

### SAE Training
```bash
python -m hsg.sae.train --model_name $NT_MODEL --layer_idx 23 --expansion_factor 8
```
Models saved to `checkpoints/` with tensorboard logging.

### Knowledge Graph Construction
```bash
python -m hsg.featureanalysis.featureKG --input data/sequences.fa --output data/output_kg.json
```
Requires SAE checkpoint and creates NetworkX graphs with token-feature relationships.

### Testing & Validation
Run full test suite: `python -m unittest` from root
Key test categories:
- Environment variables (`test_env_vars.py`)
- Pipeline integration (`test_pipelines.py`)
- SAE objects (`test_sae_objects.py`)

### Batch Processing
Shell scripts in `hsg/featureanalysis/` handle large-scale analysis:
- `largeKGconstruction.sh` - Process multiple sequence files
- `intervention_battery.sh` - Feature intervention experiments across parameter grids

## Project-Specific Conventions

### Model Architecture
- Uses InstaDeepAI nucleotide transformer (500M parameters)
- SAE layers target specific transformer layers (typically layer 23)
- Features represent strongest activations per token position

### Data Flow
1. Genomic sequences (FASTA) → embeddings via nucleotide transformer
2. Embeddings → sparse features via trained SAE
3. Features + sequence metadata → NetworkX knowledge graphs
4. Knowledge graphs → centrality analysis and gene set enrichment

### File Naming
- SAE checkpoints: `checkpoints/{expansion_factor}/layer_{idx}/`
- Knowledge graphs: `{dataset}_kg.json`
- Intervention results: `intervention_reports/feature_{id}/`

### Memory Management
Code handles GPU/CPU fallback automatically:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model.to(device)
except torch.OutOfMemoryError:
    device = torch.device("cpu")
    model.to(device)
```

## Integration Points
- **SeqRepo**: Local genomic sequence database (requires `seqrepo pull`)
- **Google Cloud**: Model storage and large dataset handling
- **MAFFT**: External tool for multiple sequence alignment
- **NetworkX**: Graph analysis and centrality computations
- **HGVS**: Genomic variant notation parsing

## Critical Dependencies
- PyTorch ecosystem for model training
- BioPython + biocommons.seqrepo for sequence handling
- NetworkX for knowledge graph analysis
- Transformers library for nucleotide transformer models
- Environment variables for external data/model paths