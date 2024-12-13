import google.auth
from google.cloud import storage
import pandas as pd
import tensorflow as tf
from pipelines.gcsstream import get_client
from io import StringIO
from tqdm import tqdm
import sys

restart = sys.argv[1]
# connect to gcs bucket
gcs_client = get_client()

print("Locating files for processing...")
bucket = gcs_client.get_bucket("ek990")
data_loc = bucket.list_blobs(match_glob="sp-per-residue-embeddings/*")

# read csv into memory
print("Processing ESM-2 output into TFRecords...")
for i, csv in enumerate(tqdm(data_loc)):
    if i < int(restart):
        continue
    # read csv into memory
    csv_file = bucket.get_blob(csv.name)
    csv_file = csv_file.download_as_string()
    csv_file = csv_file.decode("utf-8")
    csv_file = StringIO(csv_file)
    data = pd.read_csv(csv_file)
    id = csv.name.split("/")[-1]

    # convert to tfrecords, write to bucket
    for row in data.itertuples():
        with tf.io.TFRecordWriter(f"gs://ek990/sp-embed-tfrecords/{id}|{row[0]}.tfrecord") as writer:
            metadata, embeddings = row[:4], row[4:]
            record = tf.train.Example()
            # autoencoder reconstructs inputs
            record.features.feature["features"].float_list.value.extend(embeddings)
            record.features.feature["labels"].float_list.value.extend(embeddings)
            writer.write(record.SerializeToString())
        
print("TFRecord conversion complete.")