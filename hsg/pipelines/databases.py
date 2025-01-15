"""
Functions / Pipelines for writing raw nucleotide transformer embeddings to BigQuery with associated metadata.
"""

from google.cloud import bigquery
import pandas as pd


class BigQueryManager():
    """
    TODO: Class for managing BigQuery operations. Implements basic CRUD operations and functions as an SAE training/result storage pipeline.
    """

    def __init__(self):
        self.client = bigquery.Client()

    ### Table Methods ###
    def create_table(self):
        pass
    
    def read_table(self):
        pass

    def update_table(self):
        pass

    def delete_table(self):
        pass

    ### Row Methods ###
    def create_row(self):
        pass
    
    def read_row(self):
        pass

    def update_row(self):
        pass

    def delete_row(self):
        pass