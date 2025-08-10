# helper gui functions for visualizations

# import libraries
import numpy as np
import streamlit as st
from biocommons.seqrepo import SeqRepo

from google.cloud import storage

from hsg.pipelines.hidden_state import load_model
from hsg.stattools.features import LatentModel
from hsg.sae.dictionary import AutoEncoder

import tempfile
import os

from dotenv import load_dotenv
load_dotenv()

### OBJECTS ###
@st.cache_resource
class GCSHandler:
    """
    Object class for handling SAE retrieval from Google Cloud Storage.
    """
    def __init__(self):
        self.client = storage.Client()
        self.bucket = storage.Bucket(self.client, name="hidden-state-genomics")

    def fetchsae(self, expansion:int, layer:int) -> storage.Blob:
        """
        Retrieve sae state dict from gcs cloud as blob.
        """
        path = f"ef{expansion}/sae/layer_{layer}.pt"
        try:
            blob = self.bucket.get_blob(blob_name=path)
            return blob
        except Exception as e:
            print(e)
            return None
        
@st.cache_resource
class DataHandler:
    """
    Handles loading and configuration of model + sae pipelines
    """
    def __init__(self):
        self.seqrepo = SeqRepo(os.environ["SEQREPO_PATH"])
        self.parentmodel = None
        self.tokenizer = None
        self.device = None
        self.latentmodel = None

    def config_parentmodel(self, path:str = os.environ["NT_MODEL"]) -> None:
        """
        load a parent model from huggingface and assign it to self.parentmodel
        """
        self.parentmodel, self.tokenizer, self.device = load_model(path)

    def config_latentmodel(self, blob: storage.Blob):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            self.sae = AutoEncoder
            blob.download_to_file(temp_file)
            self.sae.from_pretrained(temp_file.name, device=self.device)



# prevents script output when importing functions
if __name__ == "__main__":
    pass