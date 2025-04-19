print("Importing Goodies...")

from tap import tapify
from tqdm import tqdm
from dotenv import load_dotenv

import torch
import pandas as pd
import numpy as np
from biocommons.seqrepo import SeqRepo

from hsg.stattools.features import get_latent_model
from hsg.stattools.crosscorr import xcorr_pearson
from scipy.signal import correlate

import os, difflib, warnings
warnings.filterwarnings("ignore")

##################################################################################################################################
# functions
###################################################################################################################################
def read_bed_file(bed_file: str) -> pd.DataFrame:
    """
    Read a BED file and return a DataFrame.
    
    Args:
        bed_file (str): Path to the BED file.
        
    Returns:
        pd.DataFrame: DataFrame containing the BED file data.
    """
    # Read the BED file line by line - cleaner skipping of header lines which may vary
    with open(bed_file, 'r') as f:

        rows = []
        columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", 
                   "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
        
        # Read each line in the file
        # readlines() is used to get total lines for tqdm, but if we have memory issues from 
        # that temporary object we will likely have bigger issues later anyway
        for line in tqdm(f.readlines()):

            # filter header lines
            if not line.startswith("chr"):
                continue
            # Split the line into columns
            items = line.strip().split('\t')

            rows.append(items)

        # if the rows are ragged, truncate the rows to the shortest length
        min_length = min(len(row) for row in rows)
        rows = [row[:min_length] for row in rows]

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows, columns=columns[:min_length])

        # Convert the columns to appropriate data types
        try:
            df['chrom'] = df['chrom'].astype(str)
            df['chromStart'] = df['chromStart'].astype(int)
            df['chromEnd'] = df['chromEnd'].astype(int)
            df['name'] = df['name'].astype(str)
            df['score'] = df['score'].astype(int)
            df['strand'] = df['strand'].astype(str)
            df['thickStart'] = df['thickStart'].astype(int)
            df['thickEnd'] = df['thickEnd'].astype(int)

            df['itemRgb'] = df['itemRgb'].astype(str)
            df['itemRgb'] = df['itemRgb'].apply(lambda x: list(map(int, x.split(','))) if x != '.' else [])

            df['blockCount'] = df['blockCount'].astype(int)

            df['blockSizes'] = df['blockSizes'].astype(str)
            df['blockSizes'] = df['blockSizes'].apply(lambda x: list(map(int, x.split(','))) if x != '.' else [])

            df['blockStarts'] = df['blockStarts'].astype(str)
            df['blockStarts'] = df['blockStarts'].apply(lambda x: list(map(int, x.split(','))) if x != '.' else [])

        except KeyError:
            # Most bed files do not use all 12 columns
            pass


        return df


def get_sequences_from_dataframe(df: pd.DataFrame, seqrepo: SeqRepo, pad_size: int) -> list[str]:
    """
    Get sequences from a DataFrame using SeqRepo.
    
    Args:
        df (pd.DataFrame): DataFrame containing the BED file data.
        seqrepo (SeqRepo): SeqRepo object for fetching sequences.
        
    Returns:
        list: List of sequences.
    """
    sequences = []

    for _, row in df.iterrows():

        chrom = row['chrom']
        start = row['chromStart'] - pad_size # grabbing a bit of additional sequence so we have 0s and 1s in the annotation vector
        end = row['chromEnd'] + pad_size

        # Sometimes bed files will reference non-standard chromosome locations not known to seqrepo
        # Since we're grouping by track name, we can still use the other tracks
        try:
            seq = seqrepo.fetch(namespace="GRCh38", alias=chrom, start=start, end=end)
            sequences.append(seq)
        except:
            print(f"Error fetching sequence for {chrom}:{start}-{end}")
            continue

    return sequences


def generate_annotation_vectors(seq_list: list[str], pad_size: int) -> list[np.array]:
    """
    Creates binary annotation vectors for each sequence in the list. A value of 1 indicates 
    the presence of a feature, while 0 indicates its absence.

    Args:
        seq_list (list[str]): List of sequences.
        pad_size (int): Size of the padding to be added to each sequence.
    Returns:
        list[np.array]: List of binary annotation vectors.
    """
    vec_list = []
    for seq in seq_list:
        # front pad
        front_pad = np.zeros(pad_size)
        # back pad
        back_pad = np.zeros(pad_size)
        # sequence
        seq_array = np.ones(len(seq[pad_size:-pad_size]))
        # combine & check length
        vec = np.concatenate((front_pad, seq_array, back_pad))
        assert len(vec) == len(seq), f"Vector length {len(vec)} does not match sequence length {len(seq)}"
        vec_list.append(vec)
    return vec_list


def find_chunky_index(idx: int, chunky: list[str]) -> int:
    """
    Figure out what token our index corresponds to in the original sequence. 
    """
    count = 0
    for i, c in enumerate(chunky):
        count += len(c)
        if count > idx:
            return i+1 # insert method inserts before given index
    # if we get here, the index is out of bounds
    return -1

def generate_feature_vectors(seq_list: list[str], max_context: int, model) -> list[np.array]:
    """
    Ingest a list of sequences and return a list of feature matrices for each sequence. Resulting matrices
    are of shape (sequence_length, hidden_size) where hidden_size is the size of the latent representation
    of the model.
    """

    feature_vectors = []

    for seq in seq_list:
        # split the sequence into chunks no more than max_context
        chunks = [seq[i:i+max_context] for i in range(0, len(seq), max_context)]

        # get the latent representation for each chunk
        ls_of_feat_arrays: list[torch.Tensor] = []
        ls_of_tokens: list[str] = []
        
        for c in chunks:

            # run model forward pass on data
            with torch.no_grad():
                features, tokens = model(c, return_tokens=True)

            # remove special tokens
            tokens = [t for t in tokens if t not in ["<cls>", "<pad>"]]

            ls_of_feat_arrays.append(features[1:]) # remove the first token which is the <cls> token
            ls_of_tokens.extend(tokens)

        # concatenate the results
        feat_array = torch.cat(ls_of_feat_arrays, dim=0)
        reassembled_chunks = "".join(ls_of_tokens)
        
        # if the reassembled sequence does not match the original, it is likely due to unknown tokens
        # being marked as padding tokens and getting filtered out in our results. To align the feature matrices
        # with the annotations, we use difflib to fill "Z" bases and insert zero vectors for that position
        if reassembled_chunks != seq:
            comparison = difflib.SequenceMatcher(None, reassembled_chunks, seq)
            for tag, i1, i2, j1, j2 in comparison.get_opcodes():
                if tag == 'replace':
                    gap = j2-i2
                    feat_array = torch.cat(
                        (feat_array[:i1], torch.zeros((1, feat_array.size()[1]), device=model.device), feat_array[j1:]), 
                        dim=0
                    )
                    
                    gap_token = "Z" * gap
                    gap_index = find_chunky_index(i1, ls_of_tokens)
                    ls_of_tokens.insert(gap_index, gap_token)
        
        # extend the token level feature activations to provide feature values for each base and 
        # align with the annotation vectors
        exp_features = []
        for i, j in enumerate(feat_array):
            exp_features.extend([j for _ in range(len(ls_of_tokens[i]))])
        feat_array = torch.stack(exp_features)

        # convert to numpy array
        feat_array = feat_array.cpu().numpy()
        feature_vectors.append(feat_array)

        # clear cuda cache
        if model.device != "cpu":
            torch.cuda.empty_cache()

    return feature_vectors

def binarize_features(feat_array: np.array, threshold: float = 1.0) -> np.array:
    """
    Binarize the feature array based on a threshold (numpy).
    
    Args:
        feat_array (np.array): Feature array.
        threshold (float): Threshold for binarization.
        
    Returns:
        np.array: Binarized feature array.
    """
    # binarize the features
    feat_array = np.where(feat_array > threshold, 1, 0)
    return feat_array

#################################################################################################################################
# main
#################################################################################################################################
def main(
        layer_idx: int = 23, 
        exp_factor: int = 8,
        sae_dir: str = None, 
        data_path: str = "./data/Annotation Data/RefSeqFuncElemsGRCh38.bed", 
        output_dir: str = None
        ) -> None:

    if sae_dir is None:
        sae_path = f"/home/ek224/Hidden-State-Genomics/checkpoints/hidden-state-genomics/ef{exp_factor}/sae/layer_{layer_idx}.pt"
    else:
        sae_path = f"{sae_dir}/layer_{layer_idx}.pt"

    if output_dir is None:
        output_dir = f"./data/feature_annotation/ef{exp_factor}/layer_{layer_idx}"

    # check pathing
    if not os.path.isfile(sae_path):
        raise FileNotFoundError(f"SAE file not found: {sae_path}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Loading Utilities...")

    seqrepo = SeqRepo(os.environ["SEQREPO_PATH"])
    model = get_latent_model(os.environ["NT_MODEL"], layer_idx=layer_idx, sae_path=sae_path)

    # load data
    print("Reading Data from BED file...")
    data = read_bed_file(data_path)
    data = data.groupby("name")

    # calculate correlations for each annotation
    print("=== Finding Correlations ===")

    for group, df in data:

        print(f"--- {group} ---")

        # get sequences described in bed data, pad some so vector not just 1s
        print("Gathering sequences...")
        seq_list = get_sequences_from_dataframe(df, seqrepo, pad_size=100)
        del df

        # get the annotation vectors
        print("Generating annotation vectors...")
        annotations = generate_annotation_vectors(seq_list, pad_size=100)
        annotations = np.concatenate(annotations, axis=0)

        # get the feature vectors from NTv2 & calculate pearson correlation against annotations
        # due to high dimensionality, and resource constraints, we will chunk the sequences according to 
        # the max_context of the model and calculate an average pearson R value across the track
        print("Segmenting data...")
        group_chunks = []
        for seq in seq_list:
            chunks = [seq[i:i+6000] for i in range(0, len(seq), 6000)]
            group_chunks.extend(chunks)
        del seq_list
        
        # get the latent representation for each chunk, calculate correlations
        print("Calculating track v feature correlations...")
        num = 0
        for chunk in tqdm(group_chunks):
            # not using enumerate so that the tqdm progress bar shows up
            num += 1
            # chunk annotation
            chunk_annotations = annotations[:len(chunk)]
            annotations = annotations[len(chunk):]

            # run model forward pass on data
            with torch.no_grad():
                features, tokens = model(chunk, return_tokens=True)
            # remove special tokens
            tokens = [t for t in tokens if t not in ["<cls>", "<pad>"]]
            # remove the first feature vector which is the <cls> token
            features = features[1:]
            # raise error here if we did something dumb
            if features.size()[0] != len(tokens):
                    raise ValueError(f"Feature size {features.size()[0]} does not match token size {len(tokens)}")
            # if the reassembled sequence does not match the original, it is likely due to unknown tokens
            # being marked as padding tokens and getting filtered out in our results. To align the feature matrices
            # with the annotations, we use difflib to fill "Z" bases and insert zero vectors for that position
            reassembled_chunk = "".join(tokens)
            if reassembled_chunk != chunk:
#                print(f"filling gap... {reassembled_chunk} != {chunk}")
                comparison = difflib.SequenceMatcher(None, reassembled_chunk, chunk)
                for tag, i1, i2, j1, j2 in comparison.get_opcodes():
#                    print(f"tag: {tag}, i1: {i1}, i2: {i2}, j1: {j1}, j2: {j2}")
                    if tag == 'replace':
                        gap = j2-i2
                        features= torch.cat(
                            (features[:i1], torch.zeros((1, features.size()[1]), device=model.device), features[j1:]), 
                            dim=0
                        )
                    
                        gap_token = "Z" * gap
                        gap_index = find_chunky_index(i1, tokens)
                        tokens.insert(gap_index, gap_token)
                    if tag == 'insert':
                        gap = j2-i2
                        features = torch.cat(
                            (features, torch.zeros((1, features.size()[1]), device=model.device)),
                            dim=0
                        )
                        gap_token = "Z" * gap
                        gap_index = find_chunky_index(i1, tokens)
                        tokens.insert(gap_index, gap_token)
            # extend the token level feature activations to provide feature values for each base and
            # align with the annotation vectors
            exp_features = []
            for i, j in enumerate(features):
                exp_features.extend([j for _ in range(len(tokens[i]))])
            features = torch.stack(exp_features)
            # convert to numpy array
            features = features.cpu().numpy()
            # clear cuda cache
            if model.device != "cpu":
                torch.cuda.empty_cache()
            
            # iterate through transpose to get features as x, sequence idx as y
            pearson_scores = []
            xcorr_arrays = []
            for i, feat in enumerate(features.T):
                # binarize the features
                feat = binarize_features(feat, threshold=1.0)
                # calculate the cross correlation
                xcorr = correlate(chunk_annotations, feat)
                xcorr_arrays.append(xcorr)
                # get the pearson correlation
                pearson = xcorr_pearson(feat, chunk_annotations)
                pearson_scores.append(pearson)

            # convert to numpy arrays
            xcorr_arrays = np.array(xcorr_arrays)
            pearson_scores = np.array(pearson_scores)
            # flush to disk
            if not os.path.exists(f"{output_dir}/{group}"):
                os.makedirs(f"{output_dir}/{group}", exist_ok=True)
            np.savez(f"{output_dir}/{group}/{num}.npz", pearson=pearson_scores, xcorr=xcorr_arrays)

        # end of per chunk loop
    # end of per group loop

    print("=== Done ===")
    print(f"Results saved to {output_dir}")


###############################################################################
# run module
###############################################################################
if __name__ == "__main__":
    load_dotenv()
    tapify(main)