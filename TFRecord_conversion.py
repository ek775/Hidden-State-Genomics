import google.auth
from google.cloud import storage
import pandas as pd
import tensorflow as tf
from pipelines.gcsstream import get_client
from io import StringIO
from tqdm import tqdm

# connect to gcs bucket
print("Authenticating to Google Cloud Storage...")
gcs_client = get_client()
print(f"Authenticated to project {gcs_client.project}")

print("Locating files for processing...")
bucket = gcs_client.get_bucket("ek990")
data_loc = bucket.list_blobs(match_glob="sp-per-residue-embeddings/*")

# read csv into memory
print("Processing ESM-2 output into TFRecords...")
for csv in tqdm(data_loc):
    # read csv into memory
    csv_file = bucket.get_blob(csv.name)
    csv_file = csv_file.download_as_string()
    csv_file = csv_file.decode("utf-8")
    csv_file = StringIO(csv_file)
    data = pd.read_csv(csv_file)
    id = csv.name.split("/")[-1]

    # convert to tfrecords, write to bucket
    
    with tf.io.TFRecordWriter(f"gs://ek990/sp-embed-tfrecords/{id}.tfrecord") as writer:
        for row in data.itertuples():
            metadata, embeddings = row[:4], row[4:]
            record = tf.train.Example()
            # autoencoder reconstructs inputs
            record.features.feature["features"].float_list.value.extend(embeddings)
            record.features.feature["labels"].float_list.value.extend(embeddings)
            writer.write(record.SerializeToString())

print("TFRecord conversion complete.")