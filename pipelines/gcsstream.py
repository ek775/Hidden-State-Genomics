import google.auth
from google.cloud import storage
import os
import numpy as np
import asyncio
import tensorflow_datasets as tfds

def get_client() -> storage.Client:

    """Authenticate to Google Cloud Storage and return a client"""

    print("Authenticating to Google Cloud Storage...")
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    print(f"Authenticated to project {project_id}")
    return storage_client

def data_stream(client:storage.Client, embeddings_path, metadata_path, batch_size=100):

    """Stream data from Google Cloud Storage"""

    return None

embeddings = "gs://ek990/swiss-prot-esm-1280/embeddings/embeddings"
metadata = "gs://ek990/swiss-prot-esm-1280/sequence_metadata.csv"
