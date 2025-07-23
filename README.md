# Hidden-State-Genomics
Advances in Mechanistic Interpretability have made it possible to decompose neural network activations into interpretable features via sparse auto-encoders. These features represent concepts that are learned by the model, and can be used to understand how a neural network makes its predictions. Early mechanistic interpretability studies on protein language models have led to speculation that studying the internals of these models may reveal novel biology, however, investigating this theory poses a difficult technical challenge. We attempt to answer this question by exploring relationships between features extracted from genomic language model embeddings and predicted RNA structures for novel cisplatin-RNA complexes. 

***Setup Instructions***
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
SEQREPO_PATH="./data/2024-05-23"
REFSEQ_CACHE="./data/refseq_cache.txt"
```

- Run unit tests to ensure everything is running smooth

```
python -m unittest [-v]
```

