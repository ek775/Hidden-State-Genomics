from tap import tapify
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType
import torch
import torch.nn as nn
from hsg.sae.dictionary import AutoEncoder
from google.cloud import storage
import google.auth

import os

# load environment variables
from dotenv import load_dotenv
load_dotenv()

# google authentication for hadoop
credentials, project = google.auth.default()


### HELPER FUNCTIONS ###
def config_spark() -> SparkSession:
    """
    Configures the spark session with the necessary settings for GCS. 
    
    Note that spark does not support native gcloud sdk authentication, so we need to provide the credentials manually.
    This function utilizes the python gcloud sdk to pass the appropriate credentials to the spark session for security.
    """
    spark = SparkSession.builder\
        .master("local")\
        .appName("SAE")\
        .config("spark.jars", "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar")\
        .getOrCreate()
    

    spark._jsc.hadoopConfiguration().set('fs.AbstractFileSystem.gs.impl', 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS')
    spark._jsc.hadoopConfiguration().set('fs.gs.impl', 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem')
    spark._jsc.hadoopConfiguration().set('fs.gs.project.id', project)
    spark._jsc.hadoopConfiguration().set('fs.gs.auth.client.id', credentials.client_id)
    spark._jsc.hadoopConfiguration().set('fs.gs.auth.client.secret', credentials.client_secret)
    spark._jsc.hadoopConfiguration().set('fs.gs.auth.refresh.token', credentials.refresh_token)

    spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

    return spark


def load_data(data_path: str, spark: SparkSession, hidden_size: int) -> DataFrame:
    """
    Loads SAE training data from a given csv file via spark.

    Args:
        data_path: The path to the csv file containing the training data.
        spark: The spark session to use for loading the data.
        hidden_size: Size of hidden layer for constructing schema.
    Returns:
        A DataFrame containing the training data.
    """
    # define schema
#    hidden_states = [StructField(str(i), FloatType()) for i in range(hidden_size)]
#    meta_fields = [StructField("token", StringType()), StructField("id", StringType())]
#    schema = StructType(hidden_states + meta_fields)
    # load data
    print(f"Loading data from {data_path}")
    data = spark.read.csv(data_path, header=True)
    print("=== DONE ===")
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
    hidden_layers = []

    spark_config = config_spark()

    if data_path.startswith("gs://"):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(data_path.split("/")[2])
        blobs = bucket.list_blobs()
        hidden_layers = [blob.name for blob in blobs]
    else:
        hidden_layers = [os.listdir(data_path)]
        
    for layer in hidden_layers:

        autoencoder = get_model(activation_dim=input_size, dict_size=feature_size)
        print(autoencoder)
        
        embed_df = load_data(data_path=f"{data_path}/{layer}", spark=spark_config, hidden_size=input_size)

        count = 0 
        for row in embed_df.rdd.toLocalIterator():
            if count == 0:
                print(row)
                count += 1
            else:
                count += 1
        print(count)


if __name__ == "__main__":
    tapify(main)