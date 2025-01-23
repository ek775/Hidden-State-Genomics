"""
Extract hidden states from NT language model and store in datasets for SAE training.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from hsg.pipelines.variantmap import DNAVariantProcessor
from hgvs.sequencevariant import SequenceVariant
from tqdm import tqdm

# load environment variables
from dotenv import load_dotenv
load_dotenv()

def load_model(model_name: str):
    """
    Returns pretrained model and associated tokenizer for extracting hidden states.
    """
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model.to(device)
    except torch.OutOfMemoryError:
        print(f"Model is too large for current memory constraints on {device}.")
        print("Attempting to use CPU instead...")
        device = torch.device("cpu")
        model.to(device)
    
    return model, tokenizer, device


def find_variant_ids(filepath: str) -> list[str]:
    """
    Extract correct column from clingen vs clinvar datasets.
    """
    data = pd.read_csv(filepath, header="infer", sep="\t")
    columns = data.columns

    if "#Variation" in columns:
        print("Extracting ClinGen variant IDs")
        return set(data["#Variation"].tolist())
    
    elif "Name" in columns:
        print("Extracting ClinVar variant IDs")
        return set(data["Name"].tolist())
    
    else:
        raise ValueError("Could not find variant ID column in dataset")


def package_hidden_state_data(hidden_states: torch.Tensor, variant_name: str, variant_sequence: int) -> pd.DataFrame:
    """
    Package hidden states and metadata to be written to disk.
    """
    # convert to dataframe
    activation_array = hidden_states.detach().numpy()
    activations = pd.DataFrame(activation_array)

    # align 6-mer tokens with hidden states
    token_seq = ["<start>"]
    token_seq.extend([variant_sequence[i:i+6] for i in range(0, len(variant_sequence), 6)])
    token_seq.append("<end>")
    token_seq = token_seq[:len(activations)]
    # drop masked / padding tokens
    activations.drop(activations.tail(len(activations)-len(token_seq)).index, axis=0, inplace=True)
    # finalize the dataframe
    activations["variant_sequence"] = token_seq
    activations["variant_name"] = variant_name

    return activations

    # debug
    print(activations.head())
    print(activations.describe())
    print(activations[-5:])
    print(activations.shape)
    exit()


def extract_hidden_states(
        model_name: str = os.environ["NT_MODEL"],
        csv_data_path: str = os.environ["CLIN_VAR_CSV"],
        output_dir: str = os.environ["GCLOUD_BUCKET"],
    ) -> None:

    """
    Extracts hidden states from the given language model from hugging face, by default nucleotide transformer 2.5b.

    TODO: Fix variant mapping -- using refseq data for now to develop/debug pipeline.

    Args:
        model_name (str): Name of the pretrained model.
        csv_data_path (str): Path to the csv file containing the variant data.
    
    Returns:
        CSV files for each layer of the model containing hidden states at a per-token level for use in training SAEs.

    """
    model, tokenizer, device = load_model(model_name)
    print(f"Using device: {device}")
    variant_processor = DNAVariantProcessor()
    
    print("Loading Data...")
    variant_ids: list[str] = find_variant_ids(csv_data_path)

    print("--- Processing Variants ---")
    for variant_id in tqdm(variant_ids):
        variant_name: str = variant_processor.clean_hgvs(variant_id)
        variant_obj: SequenceVariant = variant_processor.parse_variant(variant_name, return_exceptions=False)

        if variant_obj is None:
            continue

        variant_sequence: str = variant_processor.retrieve_refseq(variant_obj) # TODO: Change to retrieve variant sequence

        if variant_sequence is None:
            continue
        
        # tokenize sequence
        tokenized_sequence = tokenizer.encode_plus(
            variant_sequence, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True,
            max_length = tokenizer.model_max_length
        )["input_ids"]
        
        mask = tokenized_sequence != tokenizer.pad_token_id

        # send data to device
        tokenized_sequence = tokenized_sequence.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            output = model(
                tokenized_sequence,
                attention_mask=mask,
                encoder_attention_mask=mask,
                output_hidden_states=True
            )

        for i, layer_act in enumerate(output['hidden_states']):
            # copy tensor to cpu
            layer_act: torch.Tensor = layer_act.cpu()
            # process data
            dataframe: pd.DataFrame = package_hidden_state_data(torch.squeeze(layer_act, 0), variant_name, variant_sequence)
            # write file to GCS bucket
            layer_id: str = f"{model.esm.encoder.layer[i].__class__.__name__}[{i}]"
            layer_path: str = f"{output_dir}/{layer_id}/"
            # TODO: convert dataframe to spark dataframe for writing to GCS,
            #       repartition once full dataset written to shuffle data


    print("Done!")


if __name__ == "__main__":
    from tap import tapify
    tapify(extract_hidden_states)

