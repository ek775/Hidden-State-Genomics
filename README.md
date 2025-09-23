# Hidden-State-Genomics

## Introduction

Advances in Mechanistic Interpretability have made it possible to decompose neural network activations into interpretable features via sparse auto-encoders. These features represent concepts that are learned by the model, and can be used to understand how a neural network makes its predictions. Early mechanistic interpretability studies on protein language models have led to speculation that studying the internals of these models may reveal novel biology, however, investigating this theory poses a difficult technical challenge. We attempt to answer this question by constructing and exploring knowledge-graph relationships between features extracted from genomic language model embeddings and predicted RNA structures for novel cisplatin-RNA complexes. 

**Contents**
- [Example Graph Analysis](#multi-edge-sae-knowledge-graphs)
- [Notes](#notes)
- [Setup/Installation](#setup-instructions)

### Multi-Edge SAE Knowledge Graphs

*Multi-Layer Directed SAE knowledge graph on a random selection of cisplatin binding motifs. The basic knowledge graph is constructed from per-token strongest feature activations with associations drawn according to the triplet (subject, object, predicate) where (token, feature, sequenceID) includes metadata containing NCBI refseq annotations as found in the UCSC genome broswer. Red nodes indicate a feature, blue nodes indicate a token, and nodes are connected based on the presence of a token causing feature activation. Edge metadata is used for gene set enrichment analysis.*

![Figure 2](selected_cisplatinbinding_seqs.png) 

### Gene Set Enrichment Analysis of SAE Features (Feature 3378)

*High-centrality feature-derived gene set enrichment for SAE feature 3378 (layer 23, ef8), derived from PageRank centrality score in a knowledge graph generated from putative cisplatin-binding transcripts in the human genome. In the enriched gene set graph for feature 3378, we can see enriched GO term DAG structure and general biological processes identifiable from DNA sequences alone using genomic language model features extracted via SAE.*

![Figure 3](data/feat_3378/enrichr_GO_BPnet.png)

## Notes

The repository is generally organized with input and output `data` from various command line tools in the data folder, and our code under the `hsg` folder. Most of the scripts are intended to be run as command line tools from the root directory (i.e. `python -m hsg.some_folder.some_script --options arguments`). 

Although we mainly rely on open-source datasets for this project, some of the data we used in this analysis -- and many of the models we trained -- are too large for storage in a github repository. In addition, for some elements of our analysis, we rely on other external bioinformatics tools such as MAFFT for large-scale multiple sequence alignment. Documentation on data, processing tools, and data sourcing can be found in the data folder of this repository. If you have further questions, or would like access to the models we trained, feel free to reach out to us using the contact information in `pyproject.toml`.

## Setup Instructions
- Clone the repository

- Download the ClinGen and ClinVar datasets from the google drive shared folder

    - *I placed these inside the data directory, but you can place them anywhere that's convenient for you*

- Install the dependencies in an environment of your choice

    - *The code below executes a pip install in the "editable" mode using the pyproject.toml specifications, and will ensure you have all the dependencies in an environment agnostic way (i.e. python venv vs conda)*

```
pip install -e .
# OR
pip install -e [this repository]
```

- Install a local biocommons.seqrepo database

    - *full documentation here: https://hgvs.readthedocs.io/en/stable/installation.html#installing-seqrepo-optional*

```
seqrepo --root-dir ./data pull
```

- Add a ".env" file in the root directory to specify directories, etc.

    - This file is not tracked by git, but python-dotenv uses it to load environment variables to reduce hardcoding of things like data directories since the files are too large to be tracked as part of the repo.
    - My example is below:

```
# core environment variables
CLIN_GEN_CSV="~/Hidden-State-Genomics/data/erepo.tabbed.txt"
CLIN_VAR_CSV="~/Hidden-State-Genomics/data/variant_summary.txt"
NT_MODEL="InstaDeepAI/nucleotide-transformer-500m-human-ref"
GCLOUD_BUCKET="gs://hidden-state-genomics"
SEQREPO_PATH="./data/2024-12-20"
REFSEQ_CACHE="./data/refseq_cache.txt"
REFSEQ_GTF="data/Annotation Data/hg38.ncbiRefSeq.gtf"
```

- Run unit tests to ensure everything is running smooth

```
python -m unittest [-v]
```

