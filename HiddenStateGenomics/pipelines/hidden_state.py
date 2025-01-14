"""
Extract hidden states from NT language model and store in datasets for SAE training.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from HiddenStateGenomics.pipelines.variantmap import DNAVariantProcessor

def load_model(model_name: str):
    """
    Returns pretrained model and associated tokenizer for extracting hidden states.
    """
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def find_variant_ids(filepath: str) -> list[str]:
    """
    TODO: extract correct column from clingen vs clinvar datasets.
    """
    data = pd.read_csv(filepath, header="infer", sep="\t")

    return data["#Variation"].tolist()

def extract_hidden_states(
        model_name: str,
        csv_data_path: str,
        output_path: str,
    ):
    """
    TODO:
    Extracts hidden states from the given language model from hugging face, and stores them in a dataset for SAE training.

    Args:
        model_name (str): Name of the pretrained model.
        csv_data_path (str): Path to the csv file containing the variant data.
        output_path (str): Path to store the hidden states dataset.
    
    Returns:
        CSV files containing hidden states for training sequences on a per token basis. Each layer of the encoder will have its own file

    """
    model, tokenizer = load_model(model_name)
    variant_ids: list[str] = find_variant_ids(csv_data_path)
    variant_processor = DNAVariantProcessor()


if __name__ == "__main__":
    from tap import tapify
    tapify(extract_hidden_states)

