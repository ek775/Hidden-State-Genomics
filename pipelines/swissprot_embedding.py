# import dependencies
import os, sys
from Bio import SeqIO
import numpy as np
from transformers import AutoTokenizer, TFEsmModel
from csv import writer
from tqdm import tqdm
import asyncio
import tensorflow as tf
import tensorflow_datasets as tfds

import google.auth
from google.cloud import storage


### Configuration ###


# tpu config
if len(sys.argv) > 1 and sys.argv[1] == "tpu":
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))


### Core Functions ###


# connect to gcs bucket
def get_client() -> storage.Client:

    """Authenticate to Google Cloud Storage and return a client"""

    print("Authenticating to Google Cloud Storage...")
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    print(f"Authenticated to project {project_id}")
    return storage_client

# loop for embedding
def embed(SeqIO_entry, tokenizer, esm_model):
    # extract from fasta
    sequence = str(SeqIO_entry.seq)
    id = str(SeqIO_entry.id)
    description = (SeqIO_entry.description)
    # generate embeddings
    input = tokenizer(sequence, return_tensors='tf')
    output = esm_model(input)

    # format as per-residue embeddings in csv
    embeddings = output.last_hidden_state.numpy().squeeze()
    return sequence, id, description, embeddings

# format data for csv
def preprocess(sequence, id, description, embeddings):
    data = []
    # start token
    data.append([id, description, "<start>", *embeddings[0]])
    # per-residue embeddings
    for n, i in enumerate(sequence, start=1):
        data.append([id, description, i, *embeddings[n]])
    # end token
    data.append([id, description, "<end>", *embeddings[-1]])

    # check enumerate starting at 0 and counting from 1
    if len(data) != len(embeddings):
        raise ValueError("Data length does not match embeddings length")

    return data

# write csv lines to bucket
def gcs_write_csv(bucket, path, data):
    blob = bucket.blob(path)
    with blob.open('a') as f:
        csv_writer = writer(f)
        csv_writer.writerows(data)


### Main ###    


# load resources
gcs_client = get_client()
hug_face_id = "facebook/esm2_t33_650M_UR50D"
print(f"Loading model and tokenizer: {hug_face_id}")
tokenizer = AutoTokenizer.from_pretrained(hug_face_id)
esm_model = TFEsmModel.from_pretrained(hug_face_id)

# make output csv

# generae embeddings
with open('./data/swissprot/uniprot_sprot.fasta') as f:
    for entry in tqdm(SeqIO.parse(f, 'fasta'), total=572214):
        sequence, id, description, embeddings = embed(entry, tokenizer, esm_model)
        data = preprocess(sequence, id, description, embeddings)
        gcs_write_csv(gcs_client.get_bucket("ek990"), "sp-per-residue-embeddings.csv", data)
        