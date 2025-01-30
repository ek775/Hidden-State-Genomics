"""
Extract hidden states from NT language model and store in datasets for SAE training.
"""

import os
#import sys
from google.cloud import storage
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from hsg.pipelines.variantmap import DNAVariantProcessor
from hgvs.sequencevariant import SequenceVariant
from biocommons.seqrepo import SeqRepo
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

# for debugging and optimization
import time
import logging

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# load environment variables
from dotenv import load_dotenv
load_dotenv()


### core functions ###
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


def package_hidden_state_data(hidden_states: torch.Tensor, variant_name: str, sequence: int) -> pd.DataFrame:
    """
    Package hidden states and metadata to be written to disk.
    """
    # convert to dataframe
    activation_array = hidden_states.detach().numpy()
    activations = pd.DataFrame(activation_array)

    # align 6-mer tokens with hidden states
    token_seq = ["<start>"]
    token_seq.extend([sequence[i:i+6] for i in range(0, len(sequence), 6)])
    token_seq.append("<end>")
    token_seq = token_seq[:len(activations)]
    # drop masked / padding tokens
    activations.drop(activations.tail(len(activations)-len(token_seq)).index, axis=0, inplace=True)
    # finalize the dataframe
    activations["sequence"] = token_seq
    activations["variant_name"] = variant_name

    return activations

    # debug
    print(activations.head())
    print(activations.describe())
    print(activations[-5:])
    print(activations.shape)
    exit()


def create_dataset(dataframe: pd.DataFrame, address: str) -> None:
    """
    Create a new dataset.
    """
    start = time.perf_counter()

    # if writing to bucket, handle rejection from too many simultaneous requests
    try:
        dataframe.to_csv(address, index=False)
    except Exception as e:
        return e

    stop = time.perf_counter()
#    print(f"Time to write file: {stop - start}")


def extend_dataset(dataframe: pd.DataFrame, address: str) -> None:
    """
    Append new data to existing dataset.
    """
    start = time.perf_counter()

    # if writing to bucket, handle rejection from too many simultaneous requests
    try:
        dataframe.to_csv(address, mode="a", header=False, index=False)
    except Exception as e:
        return e

    stop = time.perf_counter()
#    print(f"Time to write file: {stop - start}")


def get_unique_refseqs(csv_data_path, variant_processor: DNAVariantProcessor) -> list[str]:
    """
    Extract unique refseq accessions from variant ids.
    """
    cache_location = os.environ["REFSEQ_CACHE"]

    if os.path.exists(cache_location):
        print(f"Loading RefSeq Accessions from Cache: {cache_location}")
        unique_refseqs = []
        with open(cache_location, "r") as file:
            for line in file:
                unique_refseqs.append(line.strip())
        return unique_refseqs
    
    else:
        print("No RefSeq Cache Found -> Extracting RefSeq Accessions from Variant IDs")
        variant_ids: set[str] = find_variant_ids(filepath=csv_data_path)
    
        print(f"Finding Unique RefSeq Accessions...")
        unique_refseqs = []
        for id in tqdm(variant_ids):
            variant_name: str = variant_processor.clean_hgvs(id)
            variant_obj: SequenceVariant = variant_processor.parse_variant(variant_name, return_exceptions=False)
            if variant_obj is not None:
                unique_refseqs.append(variant_obj.ac)
            else:
                continue
        unique_refseqs = set(unique_refseqs)
        print(f"Found Unique RefSeq Accessions: {len(unique_refseqs)}")

        # cache refseq loci for future use
        with open(cache_location, "w") as file:
            for accession in unique_refseqs:
                file.write(f"{accession}\n")

        return unique_refseqs


def gcs_or_local(address: str) -> str:
    """
    Check if address is a GCS bucket path or local path.
    """
    if address.startswith("gs://"):
        return "gcs"
    else:
        return "local"

def gcs_file_exists(address: str) -> bool:
    """
    Check if file exists in GCS bucket.
    """
    storage_client = storage.Client()
    stats = storage.Blob.from_string(address).exists(storage_client)
    return stats


### main function ###
def extract_hidden_states(
        model_name: str = os.environ["NT_MODEL"],
        csv_data_path: str = os.environ["CLIN_VAR_CSV"],
        output_dir: str = os.environ["GCLOUD_BUCKET"],
        batch_size: int = 100,
        num_workers: int = (cpu_count()-1),
    ) -> None:

    """
    Extracts hidden states from the given language model from hugging face. Defaults are taken
    from the .env file as environment variables.

    Args:
        model_name (str): Name of the pretrained model.
        csv_data_path (str): Path to the csv file containing the variant data.
        output_dir (str): Path to the GCS bucket where the hidden states will be stored.
        batch_size (int): Number of loci to process in a single batch.
        num_workers (int): Number of parallel processes to use for writing to disk.
    
    Returns:
        CSV files for each layer of the model containing hidden states at a per-token level for use in training SAEs.

    """
    # load model
    model, tokenizer, device = load_model(model_name)

    print("=============================================================")
    print(f"Extracting hidden states from model: {model_name}")
    print(f"Using device: {device}")
    print(f"Writing Embeddings to: {output_dir}")
    print("--- Parameters ---")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Workers: {num_workers}")
    print("=============================================================")

    # hgvs utilities
    variant_processor = DNAVariantProcessor()
    seqrepo = SeqRepo(os.environ["SEQREPO_PATH"])
    
    # find refseq loci from cache or from variant dataset
    unique_refseqs = get_unique_refseqs(csv_data_path, variant_processor)

    print("--- Processing Sequences ---")

    # write to disk / bucket in batches
    count = 0
    batch: list[list[pd.DataFrame]] = []
    batch_number = 0
    total_processed_accessions = 0

    for n, accession in tqdm(enumerate(unique_refseqs)):

        sequence = ""
        try:
            seq_proxy = seqrepo[f"refseq:{accession}"]
            sequence = seq_proxy.__str__()
        except KeyError:
            print(f"Could not find refseq for accession: {accession}")
            continue
        
        # tokenize sequence
        tokenized_sequence = tokenizer.encode_plus(
            sequence, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True,
            max_length = tokenizer.model_max_length
        )["input_ids"]
        
        # mask padding tokens
        mask = tokenized_sequence != tokenizer.pad_token_id

        # send data to device
        tokenized_sequence = tokenized_sequence.to(device)
        mask = mask.to(device)

        # extract hidden states
        with torch.no_grad():
            output = model(
                tokenized_sequence,
                attention_mask=mask,
                encoder_attention_mask=mask,
                output_hidden_states=True
            )

        total_processed_accessions += 1 # keep track of how many loci we've embedded

        # stage embeddings as batches in memory
        for i, layer_act in enumerate(output['hidden_states']):
            # copy tensor to cpu
            layer_act: torch.Tensor = layer_act.cpu()
            # process data
            dataframe: pd.DataFrame = package_hidden_state_data(torch.squeeze(layer_act, 0), accession, sequence)
            # merge with batch
            if count == 0:
                batch.append([dataframe])
            else:
                batch[i].append(dataframe)

        # flush staged embeddings to disk
        if count == batch_size or n == (len(unique_refseqs)-1):

            batch_number += 1
            print(f"\n--- Flushing Batch {batch_number} to Disk ---")

            # enable multiprocessing to alleviate I/O bottleneck
            start = time.perf_counter()
            process_pool = Pool(processes=num_workers)
            results = []

            for i, df_list in enumerate(batch):
                # get file labels
                if i in range(len(model.esm.encoder.layer)):
                    layer_id: str = f"{model.esm.encoder.layer[i].__class__.__name__}[{i}]"
                else:
                    layer_id: str = f"MLP_out[{i}]"

                layer_path: str = f"{output_dir}/{layer_id}.csv"

                # merge dataframes
                dataframe = pd.concat(df_list, axis=0, ignore_index=True)

                # check local output or gcs output, apply appropriate utils to create vs extend dataset
                file_exists = False
                if gcs_or_local(output_dir) == "gcs":
#                    print("executing gcs file check")
                    file_exists = gcs_file_exists(layer_path)
                elif gcs_or_local(output_dir) == "local":
#                    print("executing local file check")
                    file_exists = os.path.exists(layer_path)
                else:
                    raise ValueError("Something went wrong with the output directory. This script currently supports GCS and local paths only.")

                # flush to disk
#                print(file_exists)
                if file_exists == False:
#                    print(f"starting dataset creation process {i}")
                    result = process_pool.apply_async(create_dataset, args=(dataframe, layer_path))
                    results.append(result)
                else:
#                    print(f"starting dataset extension process {i}")
                    result = process_pool.apply_async(extend_dataset, args=(dataframe, layer_path))
                    results.append(result)

            # wait for threads to finish and close pool
            process_pool.close()
            process_pool.join()

            stop = time.perf_counter()
#            print(f"Time to write batch: {stop - start}s")

            # reset count, batch
            batch = []
            count = 0
#            exit() # debug

        else:
            count += 1
            continue

    print("=============================================================")
    print(f"Embeddings from {model_name} extracted successfully.")
    print(f"Total of [{total_processed_accessions}] refseq loci processed.")
    print("=============================================================")


# run module as main with tap cmd-line framework
if __name__ == "__main__":
    from tap import tapify
    tapify(extract_hidden_states)

