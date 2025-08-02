import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from hsg.cisplatinRNA.CNNhead import CNNHead
from hsg.featureanalysis.regelementcorr import read_bed_file, get_sequences_from_dataframe, gcloud_upload
from hsg.stattools.features import get_latent_model

from tap import tapify
from dotenv import load_dotenv
from tqdm import tqdm
from tempfile import TemporaryFile
import os

load_dotenv()

### Functions for data preparation and model training

def prepare_data(cisplatin_positive, cisplatin_negative):
    positive_df = read_bed_file(cisplatin_positive)
    negative_df = read_bed_file(cisplatin_negative)

    positive_sequences = get_sequences_from_dataframe(positive_df)
    negative_sequences = get_sequences_from_dataframe(negative_df)

    data = [(seq, 1) for seq in positive_sequences] + [(seq, 0) for seq in negative_sequences]
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)

    return train_data, validation_data, test_data


def train(upstream_model, prediction_head, train, validate, test, condition:str["embeddings", "features"], epochs=100, batch_size=32, learning_rate=0.001, output_dir=None):
    

    print("Saving results...")

    # upload to GCS
    if output_dir.startswith("gs://"):
        with TemporaryFile() as temp_file:
            torch.save(prediction_head, temp_file)
            temp_file.seek(0)
            gcloud_upload(
                temp_file, 
                bucket_name=output_dir.split("/")[2], 
                destination_blob_name='/'.join(output_dir.split("/")[3:])                      
            )

    else:
        # save to local directory
        if os.path.exists(output_dir):
            torch.save(prediction_head, os.path.join(output_dir, f'{condition}.pt'))
        else:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(prediction_head, os.path.join(output_dir, f'{condition}.pt'))

    print("=== Done ===")
    print(f"Results saved to {output_dir}")


def evaluate(model, test_data):
    pass


### Compose everything into a main function

def main(cisplatin_positive, cisplatin_negative, layer_idx, exp_factor=8, epochs=100, batch_size=32, learning_rate=0.001, sae_dir=None, output_path=None):
    """
    Conduct training and evaluation of a CNN model for RNA sequence classification.
    """
    # set default paths
    if sae_dir is None:
        sae_path = f"/home/ek224/Hidden-State-Genomics/checkpoints/hidden-state-genomics/ef{exp_factor}/sae/layer_{layer_idx}.pt"
    else:
        sae_path = f"{sae_dir}/layer_{layer_idx}.pt"

    if output_path is None:
        output_path = f"gs://hidden-state-genomics/cisplatinCNNheads/ef{exp_factor}/layer_{layer_idx}/"
    
    # Load data
    train_data, validation_data, test_data = prepare_data(cisplatin_positive, cisplatin_negative)
    # Initialize models
    upstream_model = get_latent_model(os.environ["NT_MODEL"], layer_idx, sae_path=sae_path)
    prediction_head = CNNHead(...)

    # Train model
    train(upstream_model, prediction_head, train_data, validation_data, test_data)