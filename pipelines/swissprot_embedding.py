import os
from Bio import SeqIO
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
from csv import writer
from tqdm import tqdm


# load swissprot data and generate embeddings

# file locations and ids
embed_directory = '../data/swissprot/embeddings/'
meta_file = '../data/swissprot/sequence_metadata.csv'
hug_face_id = "facebook/esm2_t33_650M_UR50D"

# make csv file for metadata
if os.path.exists(meta_file):
    raise Exception('metadata file already exists')
else:
    os.system(f'touch {meta_file}')

os.makedirs(embed_directory, exist_ok=True)

# load model
hug_face_id = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(hug_face_id)
ESM_1280 = TFAutoModel.from_pretrained(hug_face_id)

with open('../data/swissprot/uniprot_sprot.fasta') as f:
    for entry in tqdm(SeqIO.parse(f, 'fasta'), total=572214):
        
        # get metadata
        sequence = str(entry.seq)
        id = str(entry.id)
        description = (entry.description)

        # generate embeddings
        input = tokenizer(sequence, return_tensors='tf')
        output = ESM_1280(input)
        # save embeddings as numpy binaries
        with open(f"{embed_directory}{id}.npy", "wb") as f:
            np.save(f, np.array(output.last_hidden_state))
        os.system(f"gcloud storage cp {embed_directory}{id}.npy gs://ek990/swiss-prot-esm-1280/embeddings/{id}.npy")

        # append metadata to csv
        with open(meta_file, 'a') as f:
            csv_writer = writer(f)
            csv_writer.writerow([id, description, sequence])

# upload metadata to cloud storage
os.system(f"gcloud storage cp {meta_file} gs://ek990/swiss-prot-esm-1280/sequece_metadata.csv")