# Hidden-State-Genomics
ClinGen is starting to explore the utility of machine learning tools for their curation efforts, and the aim here is to predict HGVS annotations based on nucleotide sequence information alone, while providing interpretable feature activations to explain model behavior. This provides two key benefits over tools like AlphaMissense: (1) predictions can be made on variants that occur in non-coding regions, and (2) feature activations can be used to explain model behavior and identify potential gaps in literature being considered.

***Collaborator Setup Instructions***
- Clone the repository

- Download the ClinGen and ClinVar datasets from the google drive shared folder

    - *I placed these inside the genome_databases directory, but you can place them anywhere that's convenient for you*

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
seqrepo --root-dir ./genome_databases pull
```

- Add a ".env" file in the root directory to specify directories, etc.

    - This file is not tracked by git, but python-dotenv uses it to load environment variables to reduce hardcoding of things like data directories since the files are too large to be tracked as part of the repo.
    - My example is below:

```
# core environment variables
CLIN_GEN_CSV="~/Hidden-State-Genomics/genome_databases/erepo.tabbed.txt"
CLIN_VAR_CSV="~/Hidden-State-Genomics/genome_databases/variant_summary.txt"
NT_MODEL="InstaDeepAI/nucleotide-transformer-500m-human-ref"
```

- Run unit tests to ensure everything is running smooth

```
python -m unittest [-v]
```

