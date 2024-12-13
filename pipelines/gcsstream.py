import google.auth
from google.cloud import storage
import tensorflow as tf
import random

def get_client() -> storage.Client:

    """Authenticate to Google Cloud Storage and return a client"""

    print("Authenticating to Google Cloud Storage...")
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    print(f"Authenticated to project {project_id}")
    return storage_client

def train_datastream(bucket, tfrecord_path_pattern) -> list[str]:

    """Obtain filenames for training data from a GCS bucket"""

    print("Gathering TFRecords from GCS...")
    filenames = bucket.list_blobs(match_glob=tfrecord_path_pattern)
    filenames = [f"gs://{bucket.name}/{f.name}" for f in filenames]
    print(f"Found {len(filenames)} TFRecords.")
    print("Shuffling entries for training...")
    random.shuffle(filenames)
    print("--- Done ---")

    return filenames, len(filenames)

@tf.function
def parse_tf_record(record) -> tf.Tensor:

    """Parse a TFRecord into a tensor"""

    description = {
        "features": tf.io.FixedLenFeature([1280], tf.float32),
        "labels": tf.io.FixedLenFeature([1280], tf.float32)
    }

    parsed = tf.io.parse_single_example(record, description)

    return (parsed["features"], parsed["labels"])