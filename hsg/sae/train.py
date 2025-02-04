from tap import tapify
from pyspark.sql import SparkSession, DataFrame
import torch
import torch.nn as nn
from hsg.sae.dictionary import AutoEncoder
from google.cloud import storage

import os

# load environment variables
from dotenv import load_dotenv
load_dotenv()


### HELPER FUNCTIONS ###

def load_data(data_path: str) -> DataFrame:
    """
    Loads SAE training data from a given csv file via spark.

    Args:
        data_path: The path to the csv file containing the training data.
    Returns:
        A DataFrame containing the training data.
    """
    spark = SparkSession.builder.appName("SAE").getOrCreate()
    data = spark.read.csv(data_path, header=True)
    return data


def get_model(activation_dim: int, dict_size: int) -> AutoEncoder:
    """
    Instantiates the model and moves it to the GPU if available.
    """
    model = AutoEncoder(activation_dim=activation_dim, dict_size=dict_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


### MAIN FUNCTION ###

def main(
    input_size: int,
    feature_size: int,
    data_path: str = os.environ["GCLOUD_BUCKET"],
    epochs: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 0.01,
):
    """ 

    Train a given SAE model on extracted embeddings.

    Args:
        input_size: The size of the input data.
        feature_size: The size of the feature embeddings.
        data_path: The path to the training data.
        epochs: The number of epochs to train the model.
        batch_size: The number of samples per batch.
        learning_rate: The learning rate for the optimizer.

    Returns:
        A trained SAE model for extracting features from hidden_statess.

    """
    num_layers = 0

    if data_path.startswith("gs://"):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(data_path.split("/")[2])
        num_layers = bucket.list_blobs()
    else:
        num_layers = len(os.listdir(data_path))
        
    for layer in num_layers:

        autoencoder = get_model(activation_dim=input_size, dict_size=feature_size)
        print(autoencoder)
        embed_df = load_data(data_path=data_path)
        print(embed_df.head())


if __name__ == "__main__":
    tapify(main)