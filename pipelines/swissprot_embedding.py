import os, sys
from Bio import SeqIO
import numpy as np
from transformers import AutoTokenizer, TFEsmModel
from csv import writer
from tqdm import tqdm
import asyncio


# load swissprot data and generate embeddings
num_workers = sys.argv[0]

# file locations and ids
embed_directory = './data/swissprot/embeddings/'
meta_file = './data/swissprot/sequence_metadata.csv'
hug_face_id = "facebook/esm2_t33_650M_UR50D"

# make csv file for metadata
os.system(f'touch {meta_file}')
os.makedirs(embed_directory, exist_ok=True)

# load model
hug_face_id = "facebook/esm2_t33_650M_UR50D"
workers = [(AutoTokenizer.from_pretrained(hug_face_id), TFEsmModel.from_pretrained(hug_face_id)) for i in range(num_workers)]

# core function
async def embed(SeqIO_entry, tokenizer, esm_model):
    sequence = str(SeqIO_entry.seq)
    id = str(SeqIO_entry.id)
    description = (SeqIO_entry.description)

    # generate embeddings
    input = tokenizer(sequence, return_tensors='tf')
    output = esm_model(input)
    # save embeddings as numpy binaries
    temp_file = f"{embed_directory}{id}.npy"
    with open(temp_file, "wb") as f:
        np.save(f, np.array(output.last_hidden_state))
    os.system(f"gcloud storage cp {temp_file} gs://ek990/swiss-prot-esm-1280/embeddings/")
    os.remove(temp_file)
    # append metadata to csv
    with open(meta_file, 'a') as f:
        csv_writer = writer(f)
        csv_writer.writerow([id, description, sequence])

# main coroutine generator
async def main(SeqIO_entry, worker_list):
    await asyncio.gather(
        *[embed(SeqIO_entry, w[0], w[1]) for w in worker_list]
    )

### Main Loop ###    
with open('./data/swissprot/uniprot_sprot.fasta') as f:
    for entry in tqdm(SeqIO.parse(f, 'fasta'), total=572214):
        asyncio.run(main(entry, worker_list=workers))
        

# upload metadata to cloud storage
os.system(f"gcloud storage cp {meta_file} gs://ek990/swiss-prot-esm-1280/sequece_metadata.csv")