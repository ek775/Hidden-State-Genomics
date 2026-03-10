# Hidden-State-Genomics AI Coding Instructions

## Project Overview
This is a mechanistic interpretability project studying genomic language models. The core pipeline: genomic sequences → nucleotide transformer embeddings → SAE sparse features → knowledge graphs → biological insights. Specifically analyzes cisplatin-RNA binding patterns through SAE feature extraction and graph-theoretic analysis.

## Architecture & Key Components

### Core Pipeline (hsg/ directory)
- **`hsg/sae/`**: Sparse Auto-Encoder training and feature extraction
  - `train.py` - Main SAE training via `tapify()` CLI with TensorBoard logging
  - `dictionary.py` - `AutoEncoder` class: one-layer autoencoder with ReLU activation, unit-norm decoder
  - `protocol/` - Training protocols: `etl.py` (hidden state extraction), `epoch.py` (train/validate), `checkpoint.py` (early stopping)
  - Expansion factors (ef8/ef16/ef32) control latent dimension: `dict_size = activation_dim × expansion_factor`
- **`hsg/stattools/`**: Model wrappers for feature extraction
  - `features.py` - **Critical**: `LatentModel` and `FullLatentModel` classes wrap parent + SAE for inference
  - `get_latent_model()` factory function loads combined parent+SAE model from checkpoint
- **`hsg/pipelines/`**: Data processing and genomic model integration
  - `hidden_state.py` - Extract embeddings via `load_model()` (handles GPU/CPU fallback automatically)
  - `variantmap.py` - `DNAVariantProcessor` for HGVS variant notation and SeqRepo sequence retrieval
  - `feature_sensitivity.py` - Dependency map analysis for feature-position relationships
- **`hsg/featureanalysis/`**: Knowledge graph construction and causal analysis
  - `featureKG.py` - Constructs NetworkX MultiDiGraph: tokens→features with genomic metadata edges
  - `intervention.py` - Feature intervention: amplify target feature by `act_factor`, suppress others by `1/act_factor`
  - `*.sh` scripts - Batch processing with parameter grids (see [Batch Processing](#batch-processing))

### Additional Components
- **`hsg/cisplatinRNA/`**: Downstream classification tasks
  - `CNNhead.py` - `CNNHead` model: 2-layer CNN classifier on SAE features/embeddings
  - `CNNtrain.py` - Training loop with `prepare_data()` for cisplatin binding classification
  - Classifies sequences based on SAE-derived features (binary: binding vs non-binding)
- **`hsg/depend/`**: Feature dependency analysis
  - `featurewise.py` - `calculate_position_dependency_map()` for feature-position relationships
  - `reduce.py` - Dimensionality reduction: `spectral_embedding()`, `minmax_normalize()`
- **`hsg/sequence.py`**: Core sequence utilities
  - `transcribe(seq)` - DNA→RNA with reverse complement
  - `revcomp(seq)` - DNA reverse complement

### Command-Line Interface Pattern
All main scripts use `tap` (typed-argument-parser) with `tapify()` for automatic CLI generation:
```python
from tap import tapify

def main(model_name: str, layer_idx: int = 23, expansion_factor: int = 8):
    """Docstring becomes --help text. Type hints define argument types."""
    pass

if __name__ == "__main__":
    tapify(main)  # Generates CLI from function signature
```
Run modules as: `python -m hsg.module.script --arg1 value --arg2 value`

### Environment Configuration
**Critical**: Always load environment variables at the top of scripts:
```python
from dotenv import load_dotenv
load_dotenv()  # Must call before accessing os.environ
```
Required `.env` variables (documented in `hsg/tests/test_env_vars.py`):
- `NT_MODEL`: HuggingFace model path (default: `InstaDeepAI/nucleotide-transformer-500m-human-ref`)
- `SEQREPO_PATH`: Local SeqRepo database (e.g., `./data/2024-12-20`)
- `REFSEQ_CACHE`: Cache file for RefSeq lookups
- `REFSEQ_GTF`: NCBI RefSeq GTF annotation file (`data/Annotation Data/hg38.ncbiRefSeq.gtf`)
- `GCLOUD_BUCKET`: Google Cloud Storage bucket (e.g., `gs://hidden-state-genomics`)
- `CLIN_GEN_CSV`, `CLIN_VAR_CSV`: Clinical variant datasets

### Data Organization & File Naming
- **`data/`**: Large datasets (not tracked in git, loaded from Google Cloud)
  - Cisplatin sequences: `cisplatin_pos.fa`, `cisplatin_neg45k.fa` (FASTA format)
  - Knowledge graphs: `{dataset}_kg.json` (NetworkX JSON serialization)
  - Gene sets: `gene_sets/enrich_{feature_id}/` (GO enrichment results)
  - Annotations: `Annotation Data/hg38.ncbiRefSeq.gtf` (UCSC genome browser format)
  - SeqRepo databases: `2024-05-23/`, `2024-12-20/` (versioned reference genomes)
- **`checkpoints/`**: Trained models organized hierarchically
  - `checkpoints/hidden-state-genomics/ef{8,16,32}/sae/layer_{idx}.pt` - SAE weights
  - `checkpoints/best_model23.pt` - Best performing SAE (layer 23)
  - CNN heads stored on GCS: `gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/`
- **`intervention_reports/`**: Generated intervention analysis results
  - `feature_{id}/min{min_act}_factor{act_factor}/` - Markdown reports + figures
- **`notebook_stash/`**: Exploratory analysis notebooks (not production code)

## Development Workflows

### SAE Training
Train sparse auto-encoders on specific transformer layers:
```bash
python -m hsg.sae.train \
    --model_name $NT_MODEL \
    --layer_idx 23 \
    --expansion_factor 8 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --l1_penalty 1e-5
```
- Saves to `checkpoints/hidden-state-genomics/ef{expansion_factor}/sae/layer_{layer_idx}.pt`
- TensorBoard logs: `log_dir/layer_{layer_idx}/` with train/val loss curves
- Early stopping via `History` tracker (patience=100 by default)

### Loading Models for Inference
Use `LatentModel` wrapper to combine parent model + SAE:
```python
from hsg.stattools.features import get_latent_model

model = get_latent_model(
    parent_model_path=os.environ["NT_MODEL"],
    layer_idx=23,
    sae_path="checkpoints/hidden-state-genomics/ef8/sae/layer_23.pt"
)

# Get SAE features + optional outputs
latents, hidden_states, tokens = model.forward(
    "ATCGATCG",
    return_hidden_states=True,
    return_tokens=True
)
```

### Knowledge Graph Construction
Generate NetworkX MultiDiGraph from genomic sequences:
```bash
python -m hsg.featureanalysis.featureKG \
    --input data/cisplatin_pos.fa \
    --output data/cisplatin_pos_kg.json \
    --exp_factor 8 \
    --layer_idx 23
```
**Graph structure** (see `docs/knowledge_graph_theory.md`):
- Vertices: tokens (strings) + features (integers 0 to dict_size-1)
- Edges: directed token→feature, typed by token value
- Edge attributes: `{sequence_id, chrom, start, end, strand, annotations}`
- Annotations enriched via `gffutils` from NCBI RefSeq GTF

### Feature Intervention Analysis
Test causal effect of specific features on downstream tasks:
```bash
python -m hsg.featureanalysis.intervention \
    --feature 3378 \
    --min_act 0.1 \
    --act_factor 10.0 \
    --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt
```
**Intervention math** (see `docs/intervention_operation.md`):
- Clamps target feature: `latent[:, feature] = max(latent[:, feature], min_act)`
- Amplifies target: `latent[:, feature] *= act_factor`
- Suppresses others: `latent[:, j≠feature] *= 1/act_factor`
- Generates markdown report with ROC curves, confusion matrices

### Testing & Validation
Run full test suite from project root:
```bash
python -m unittest          # All tests
python -m unittest -v       # Verbose output
python -m unittest hsg.tests.test_pipelines  # Specific module
```
Key test files:
- `test_env_vars.py` - Validates `.env` configuration and required variables
- `test_pipelines.py` - HGVS parsing, SeqRepo retrieval, variant mapping (1000 sample test)
- `test_sae_objects.py` - AutoEncoder forward/backward passes

### Batch Processing
Shell scripts use parameter grids for systematic analysis:
```bash
# hsg/featureanalysis/intervention_battery.sh
for feature in 407 3378 4793; do
  for min_act in 0.1 10.0; do
    for act_factor in 0.0 10.0; do
      python -m hsg.featureanalysis.intervention \
        --feature $feature --min_act $min_act --act_factor $act_factor
    done
  done
done
```
Common pattern: nested loops over features × hyperparameters, results uploaded to GCS

## Project-Specific Conventions

### Model Architecture Details
- **Parent model**: InstaDeepAI nucleotide-transformer-500m-human-ref (500M parameters, ESM-based)
  - 33 transformer layers, default analysis on layer 23 (late semantic layer)
  - Layer access: `parent_model.esm.encoder.layer[layer_idx]`
  - Tokenizes DNA sequences with 6-mer BPE tokens
- **SAE architecture** (`hsg/sae/dictionary.py`):
  - Single-layer: encoder (linear + ReLU), decoder (linear, unit-norm weights)
  - Bias vector learned separately, subtracted before encoding
  - Standard expansion factors: ef8 (10,240 features), ef16 (20,480), ef32 (40,960)
  - Training loss: MSE reconstruction + L1 sparsity penalty (annealed over first N steps)
- **CNN classification head** (`hsg/cisplatinRNA/CNNhead.py`):
  - 2 conv layers (adaptive kernel size), 2 max pools, dropout, final FC layer
  - Input: SAE features OR decoder reconstructions (shape: `[seq_length, feature_dim]`)
  - Output: binary classification (cisplatin binding probability)
  - Includes `pad_sequence()` method for variable-length inputs

### Complete Data Flow Pipeline
```
1. Raw sequences (FASTA/BED)
   ↓ (SeqRepo retrieval if BED, BioPython if FASTA)
2. Tokenized sequences
   ↓ (nucleotide transformer forward pass)
3. Hidden states from layer N (shape: [seq_len, 1280])
   ↓ (SAE encoder)
4. Sparse features (shape: [seq_len, dict_size])
   ↓ (argmax per position OR full tensor)
5a. Knowledge Graph: token→feature_id edges with genomic metadata
5b. CNN Classification: SAE features → binary prediction
5c. Intervention: modified features → causal analysis
```

### GPU Memory Management Pattern
Universal pattern across codebase:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model.to(device)
except torch.OutOfMemoryError:
    logging.error("GPU insufficient, falling back to CPU")
    device = torch.device("cpu")
    model.to(device)
# Also: torch.cuda.empty_cache() after inference in LatentModel
```

### Google Cloud Storage Integration
Files loaded transparently from GCS using `gcsfs`:
```python
# Direct loading from GCS paths
model = CNNHead.from_pretrained("gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt")
# Upload results after batch processing
os.system("gsutil -m cp -r ./intervention_reports/ gs://hidden-state-genomics/featureanalysis/")
```

### Sequence Format Handling
- **FASTA**: Use BioPython `SeqIO.parse(input, "fasta")`
  - `record.id` → sequence identifier (used in KG)
  - `str(record.seq)` → sequence string
- **BED**: Use custom `read_bed_file()` + SeqRepo
  - Columns: `chrom, chromStart, chromEnd, name, score, strand`
  - Description format: `{chrom}:{start}-{end}({strand})`
  - Requires SeqReposequence database for coordinate→sequence mapping

### Genomic Annotation System
Uses `gffutils` for GTF→SQLite database:
```python
# First run creates .db file from GTF (slow, one-time)
annotdb = gffutils.create_db(
    os.environ["REFSEQ_GTF"],
    dbfn="path/to/output.db",
    merge_strategy='merge'
)
# Subsequent runs connect to existing DB (fast)
annotdb = gffutils.FeatureDB("path/to/output.db")
# Query by genomic coordinates
features = annotdb.region(seqid="chr1", start=1000, end=2000)
```

## Integration Points & External Systems

### SeqRepo - Local Genomic Reference Database
- **Setup**: `seqrepo --root-dir ./data pull` (downloads ~10GB hg38 reference)
- **Usage**: `SeqRepo(os.environ["SEQREPO_PATH"])` gives coordinate→sequence API
- **Versioning**: Multiple versions in `data/2024-05-23/`, `data/2024-12-20/`
- **Purpose**: Convert genomic coordinates (BED format) to actual nucleotide sequences

### Google Cloud Storage
- **Access pattern**: Direct path URLs work transparently with PyTorch/gcsfs
  - `torch.load("gs://bucket/path/model.pt")` - automatic download & caching
  - `gsutil -m cp -r` - bulk uploads after batch processing
- **Usage**: Large model storage (CNN heads), intervention results, training checkpoints
- **Bucket**: `gs://hidden-state-genomics/` (defined in `GCLOUD_BUCKET` env var)

### HGVS - Variant Notation System
- **Library**: `hgvs` package + `biocommons.seqrepo`
- **Parser**: `DNAVariantProcessor.parse_variant(hgvs_string)` returns `SequenceVariant` object
- **Projections**: Genomic→coding→protein coordinate transformations
- **Test coverage**: `test_pipelines.py` validates 1000 random ClinGen/ClinVar variants

### NetworkX - Knowledge Graph Engine
- **Graph type**: `nx.MultiDiGraph` (directed, supports parallel edges)
- **Serialization**: `json_graph.node_link_data(G)` → JSON file
- **Analysis**: PageRank, betweenness centrality, Louvain community detection
- **Scale**: Typical KGs have ~100k nodes (tokens + features), ~500k edges

### GFFUtils - Genomic Annotation Database
- **Input**: NCBI RefSeq GTF file (`hg38.ncbiRefSeq.gtf`)
- **Creates**: SQLite index (`.db` file) for fast coordinate-based queries
- **Queries**: `annotdb.region(seqid="chr1", start=X, end=Y)` → genes/transcripts/exons
- **One-time setup**: Initial DB creation takes ~30 minutes, subsequent loads are instant

### MAFFT - Multiple Sequence Alignment
- **Usage**: External command-line tool for sequence alignments (called via `os.system()`)
- **Purpose**: Align multiple genomic sequences for motif discovery
- **Location**: Expected in system PATH

## Critical Dependencies & Version Constraints

### Core Stack (fixed versions in `pyproject.toml`)
- **Python**: `==3.12.8` (strict - compatibility issues with newer versions)
- **PyTorch**: `2.5.1` with `torchvision==0.20.1` (CUDA 11.8/12.1 compatible)
- **Transformers**: `4.47.1` (HuggingFace) - nucleotide transformer models
- **nnsight**: `0.3.7` - for extracting intermediate activations from transformers

### Bioinformatics Stack
- **BioPython**: `1.84` - FASTA/GenBank parsing, sequence operations
- **biocommons.seqrepo**: `0.6.9` - reference genome database
- **hgvs**: `1.5.4` - variant notation parsing/validation
- **gffutils**: GTF/GFF annotation processing

### Analysis & Visualization
- **NetworkX**: Knowledge graph construction and analysis
- **pandas**: `2.2.3`, **numpy**: `2.0.0` - data manipulation
- **scikit-learn**: `1.6.0`, **umap-learn**: `0.5.7` - dimensionality reduction
- **plotly**: `5.24.1`, **seaborn**: `0.13.2` - visualization
- **tensorboard**: `2.19.0` - training metrics logging

### CLI & Utilities
- **typed-argument-parser**: `1.10.1` - `tap`/`tapify()` for automatic CLI generation
- **python-dotenv**: `1.0.1` - `.env` file loading
- **tqdm**: `4.67.1` - progress bars
- **gcsfs**: `2024.12.0` - Google Cloud Storage filesystem interface

### Installation Note
Use editable install from project root: `pip install -e .`  
This installs the `HiddenStateGenomics` package and all dependencies from `pyproject.toml`

## Common Gotchas & Troubleshooting

1. **Missing `.env` file**: Many scripts silently fail without proper environment variables. Run `python -m unittest hsg.tests.test_env_vars` to validate.

2. **SeqRepo database not initialized**: Convert BED to sequences requires SeqRepo. Error: `KeyError` when accessing sequences. Fix: `seqrepo --root-dir ./data/2024-12-20 pull`

3. **CUDA out of memory**: Models auto-fallback to CPU, but check logs. May need to reduce batch size or use smaller expansion factor.

4. **GFFUtils database missing**: First run of `featureKG.py` creates `.db` file (30 min). Subsequent runs fail if GTF path changes but old `.db` exists - delete stale `.db` files.

5. **Google Cloud authentication**: GCS access requires `gcloud auth login` or service account credentials. Set `GOOGLE_APPLICATION_CREDENTIALS` env var if using service accounts.

6. **Module import errors**: Ensure package is installed in editable mode (`pip install -e .`) so `from hsg.module import ...` works from any directory.

7. **Test failures on variant parsing**: ~10-15% of ClinGen/ClinVar variants use unsupported HGVS notation (e.g., uncertain positions, complex rearrangements). This is expected.