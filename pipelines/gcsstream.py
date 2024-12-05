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


